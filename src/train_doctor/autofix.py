"""`train-doctor fix`: profile, try the applicable changes, prove the winners.

``fix`` is the headline, one-command path. It profiles the unchanged training
command, turns the ranked findings into concrete candidate changes it can apply
on its own, measures each with the paired, noise-aware ``compare`` (so the loss
check and stall handling come for free), keeps only the clear winners, tries the
combination of winners, and ends with a before/after number and a reversible
patch. It never reports a speedup ``compare`` did not call ``faster`` with the
loss within tolerance.

Candidates are applied by injecting ``TRAIN_DOCTOR_FIX_*`` settings into the
candidate run (see ``_hook/td_hook.py``), so the same unchanged command is the
baseline and the candidate; only the injected settings differ. Changes that
cannot be applied without editing the user's source (device moves, mixed
precision, torch.compile, batch size, removing sync points) are reported as
suggestions, not measured, so the output stays honest about what was proven.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from train_doctor import api, card, patcher, runner
from train_doctor.redact import dump_json


@dataclass
class Candidate:
    id: str
    title: str
    from_finding: str
    env: dict[str, str]  # TRAIN_DOCTOR_FIX_* settings injected into the candidate run
    edits: list[dict]  # source edits for the patch (see patcher.py)
    risk: str
    describe: str
    loss_sensitive: bool = False


_GENERIC_STEMS = {"train", "main", "run", "__main__", "train_net", "trainer"}


def _label_for(argv: list[str]) -> str:
    for a in argv:
        if a.endswith(".py"):
            p = Path(a)
            # train.py in a project directory reads better as the directory name
            if p.stem in _GENERIC_STEMS and p.parent.name:
                return p.parent.name
            return p.stem
    for a in argv:
        if not a.startswith("-") and "/" not in a:
            return a
    return "fix"


def candidates_from_findings(ev: dict, findings: list[dict], budget: int) -> tuple[list[Candidate], list[dict]]:
    """Map ranked findings to the changes ``fix`` can apply itself, in finding order.

    Returns ``(candidates, suggestions)``: candidates are measured; suggestions are
    findings whose fix needs a source change ``fix`` will not make blindly.
    """
    device = (ev.get("config") or {}).get("device")
    cands: list[Candidate] = []
    suggestions: list[dict] = []
    seen_ids: set[str] = set()
    covers_pin = False
    for f in findings:
        fid = f["id"]
        nums = f.get("numbers") or {}
        if fid == "dataloader_bound":
            k = int(nums.get("suggested_workers") or 4)
            # num_workers alone is loss-neutral: with persistent_workers=False (the torch default) the
            # shuffle RNG advances exactly as it does single-process. persistent_workers=True would avoid
            # re-spawning workers each epoch but changes the per-epoch shuffle order, so it is left to the
            # user (see the suggestion below) rather than bundled into a "loss unchanged" guarantee.
            env = {"TRAIN_DOCTOR_FIX_NUM_WORKERS": str(k)}
            kwargs = {"num_workers": k}
            if device == "cuda":
                env["TRAIN_DOCTOR_FIX_PIN_MEMORY"] = "1"
                kwargs["pin_memory"] = True
                covers_pin = True
            cands.append(
                Candidate(
                    id="dataloader_workers",
                    title=f"DataLoader workers (num_workers={k})",
                    from_finding=fid,
                    env=env,
                    edits=[{"kind": "dataloader_kwargs", "kwargs": kwargs}],
                    risk=f["risk"],
                    describe=f"num_workers={k}" + (", pin_memory=True" if device == "cuda" else ""),
                )
            )
            suggestions.append(
                {
                    "id": "persistent_workers",
                    "title": "persistent_workers=True",
                    "suggestion": (
                        "If num_workers wins, persistent_workers=True also avoids re-spawning the workers each epoch. "
                        "It is not applied automatically because it changes the per-epoch shuffle order (a different, "
                        "statistically equivalent trajectory), which the loss check would flag."
                    ),
                    "risk": "low",
                    "reason": "changes the shuffle order across epochs",
                }
            )
        elif fid == "pin_memory_off" and not covers_pin:
            cands.append(
                Candidate(
                    id="pin_memory",
                    title="pin_memory=True",
                    from_finding=fid,
                    env={"TRAIN_DOCTOR_FIX_PIN_MEMORY": "1"},
                    edits=[{"kind": "dataloader_kwargs", "kwargs": {"pin_memory": True}}],
                    risk=f["risk"],
                    describe="pin_memory=True on the DataLoader",
                )
            )
        elif fid == "cudnn_benchmark_off":
            cands.append(
                Candidate(
                    id="cudnn_benchmark",
                    title="cudnn.benchmark=True",
                    from_finding=fid,
                    env={"TRAIN_DOCTOR_FIX_CUDNN_BENCHMARK": "1"},
                    edits=[{"kind": "line_after_import_torch", "line": "torch.backends.cudnn.benchmark = True"}],
                    risk=f["risk"],
                    describe="torch.backends.cudnn.benchmark = True",
                )
            )
        else:
            suggestions.append(
                {"id": fid, "title": f["title"], "suggestion": f["suggestion"], "risk": f["risk"], "reason": "needs a source change fix does not make on its own"}
            )
        seen_ids.add(fid)
    return cands[:budget], suggestions


def _headline(label: str, res: dict, machine: dict) -> str:
    ra = res["ratio"]
    unit = "samples/s" if res["metric"] == "samples_per_s" else "steps/s"
    tp = res["throughput"]
    before, after = tp["baseline"]["median"], tp["candidate"]["median"]
    loss = (res.get("loss_check") or {}).get("status", "unchecked")
    loss_phrase = "loss unchanged" if loss in ("identical", "within tolerance") else f"loss {loss}"
    return (
        f"{label}: {before:,.1f} -> {after:,.1f} {unit}, {ra['point']:.2f}x faster "
        f"(95% interval {ra['low']:.2f} to {ra['high']:.2f}), {loss_phrase}."
    )


def fix(
    command: str | list[str],
    *,
    budget: int = 4,
    keep: bool = False,
    install_missing: bool = False,
    cwd: str = ".",
    repeats: int = 5,
    warmup_runs: int = 1,
    warmup_steps: int = 5,
    steps: int = 50,
    seconds: float = 2.0,
    timeout: float = 600.0,
    seed: int = 0,
    loss_tol: float = 0.05,
    min_effect: float = 0.02,
    out_dir: str | Path = api.DEFAULT_OUT,
    make_card: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Profile ``command``, measure each applicable change, and keep the proven winners."""
    argv = api.as_argv(command)

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    argv, interp_note = runner.resolve_target_interpreter(argv, cwd)
    if interp_note:
        say(interp_note)
    label = _label_for(argv)

    def do_profile() -> dict:
        return api.profile(argv, cwd=cwd, warmup_steps=warmup_steps, steps=steps, seconds=seconds, timeout=timeout, profiler_steps=0, out_dir=out_dir, label=f"{label}-fix-profile")

    say(f"Profiling {shlex.join(argv)}")
    prof = do_profile()
    out: dict = {"kind": "fix", "label": label, "command": argv, "cwd": cwd, "profile_dir": prof["run_dir"], "candidates": [], "suggestions": [], "proven": None, "machine": prof["evidence"].get("machine", {})}
    if interp_note:
        out["interpreter_note"] = interp_note
    if prof.get("error"):
        missing = prof["error"].get("missing_module")
        if missing and install_missing and (os.sep in argv[0] or argv[0] in ("python", "python3")):
            ok, msg = runner.install_missing(argv[0], missing, cwd)
            say(msg)
            out["install"] = msg
            if ok:
                prof = do_profile()
                out["profile_dir"] = prof["run_dir"]
        if prof.get("error"):
            out["error"] = prof["error"]
            fixcmd = f"{argv[0]} -m pip install {missing}" if missing else None
            out["headline"] = (
                f"{label}: could not profile the command. {prof['error'].get('hint')}"
                + (f" To let fix install it, re-run with --install-missing (it would run `{fixcmd}`)." if missing and not install_missing else "")
            )
            _write(out_dir, out, label)
            return out

    findings = prof["findings"]
    cands, suggestions = candidates_from_findings(prof["evidence"], findings, budget)
    out["suggestions"] = suggestions
    if not cands:
        out["headline"] = f"{label}: no proven win. No change fix can apply on its own addressed the top findings."
        out["next_step"] = prof["next_step"]
        _write(out_dir, out, label)
        return out

    results: list[dict] = []
    winners: list[tuple[Candidate, dict]] = []
    for c in cands:
        say(f"Trying {c.title}")
        res = api.compare(
            argv, argv, repeats=repeats, warmup_runs=warmup_runs, cwd=cwd, warmup_steps=warmup_steps, steps=steps, seconds=seconds,
            timeout=timeout, seed=seed, loss_tol=loss_tol, min_effect=min_effect, out_dir=out_dir, candidate_env=c.env, label=f"{label}-fix-{c.id}",
            progress=(lambda d, t, m: say("  " + m)) if progress else None,
        )
        entry = _result_entry(c, res)
        results.append(entry)
        say(f"  {c.title}: {entry['decision']} ({entry.get('reason')})")
        if res.get("decision") == "keep":
            winners.append((c, res))
    out["candidates"] = results

    combined = None
    if len(winners) >= 2:
        env: dict[str, str] = {}
        for c, _ in winners:
            env.update(c.env)
        say("Trying the winners together")
        res = api.compare(
            argv, argv, repeats=repeats, warmup_runs=warmup_runs, cwd=cwd, warmup_steps=warmup_steps, steps=steps, seconds=seconds,
            timeout=timeout, seed=seed, loss_tol=loss_tol, min_effect=min_effect, out_dir=out_dir, candidate_env=env, label=f"{label}-fix-combined",
            progress=(lambda d, t, m: say("  " + m)) if progress else None,
        )
        combo_edits = [e for c, _ in winners for e in c.edits]
        combo = Candidate(id="combined", title="all winners together", from_finding="+".join(c.id for c, _ in winners), env=env, edits=combo_edits, risk="low", describe=", ".join(c.describe for c, _ in winners))
        entry = _result_entry(combo, res)
        entry["combined_of"] = [c.id for c, _ in winners]
        out["combined"] = entry
        say(f"  combined: {entry['decision']} ({entry.get('reason')})")
        if res.get("decision") == "keep":
            combined = (combo, res)

    # best proven change: the kept result with the largest median pair ratio
    kept = [(c, r) for c, r in winners]
    if combined:
        kept.append(combined)
    if kept:
        best_c, best_r = max(kept, key=lambda cr: cr[1]["ratio"]["point"])
        out["proven"] = {"candidate": best_c.id, "describe": best_c.describe, "compare_dir": best_r["root"], "ratio": best_r["ratio"], "loss_check": best_r.get("loss_check")}
        out["headline"] = _headline(label, best_r, out["machine"])
        patch_info = patcher.make_patch(argv, cwd, best_c.edits, Path(out_dir))
        out["patch"] = patch_info
        if keep and patch_info.get("patch_path") and patch_info.get("target"):
            out["applied"] = patcher.apply_patch(patch_info)
        if make_card:
            try:
                out["card"] = card.render_card(label, best_r, out["machine"], argv, Path(out_dir))
            except Exception as e:
                out["card"] = {"error": repr(e)}
    else:
        tried = "; ".join(f"{e['candidate']}: {e['decision']}" for e in results)
        out["headline"] = f"{label}: no proven win. Tried {len(results)} change(s), none kept ({tried})."

    _write(out_dir, out, label)
    return out


def _result_entry(c: Candidate, res: dict) -> dict:
    return {
        "candidate": c.id,
        "title": c.title,
        "from_finding": c.from_finding,
        "describe": c.describe,
        "env": c.env,
        "decision": res.get("decision"),
        "verdict": res.get("verdict"),
        "reason": res.get("reason"),
        "ratio": res.get("ratio"),
        "loss_check": {k: (res.get("loss_check") or {}).get(k) for k in ("status", "mean_rel_diff", "final_rel_diff")},
        "compare_dir": res.get("root"),
    }


def _write(out_dir: str | Path, out: dict, label: str) -> None:
    if isinstance(out.get("patch"), dict):
        out["patch"].pop("_new_text", None)
    d = api.new_run_dir(out_dir, "fix", label)
    out["fix_dir"] = str(d)
    dump_json(d / "fix.json", out)
    (d / "fix.md").write_text(render_fix_md(out))


def render_fix_md(out: dict) -> str:
    L: list[str] = []
    label = out.get("label", "fix")
    L.append(f"# train-doctor fix: {label}\n")
    L.append(f"Command: `{shlex.join(out.get('command') or [])}`\n")
    L.append(f"**{out.get('headline', '')}**\n")
    if out.get("error"):
        L.append(f"The command could not be profiled: {out['error'].get('hint')}\n")
        return "\n".join(L)
    if out.get("proven"):
        p = out["proven"]
        L.append("## Proven change\n")
        L.append(f"- {p['describe']} (from candidate `{p['candidate']}`)")
        lc = p.get("loss_check") or {}
        L.append(f"- Loss check: {lc.get('status')}")
        ra = p["ratio"]
        L.append(f"- Speed: {ra['point']:.2f}x (95% interval {ra['low']:.2f} to {ra['high']:.2f}), measured over {ra['n_pairs']} pairs")
        L.append(f"- Evidence: `{p['compare_dir']}/report.md`")
        if out.get("patch"):
            pt = out["patch"]
            if pt.get("patch_path"):
                L.append(f"- Patch: `{pt['patch_path']}`" + (f" (applied to `{pt.get('target')}`)" if out.get("applied", {}).get("applied") else " (apply with `git apply` or `patch -p0`; reverse with `-R`)"))
            else:
                L.append(f"- Patch: not generated automatically ({pt.get('reason')}). Make this edit by hand: {p['describe']}.")
        L.append("")
    L.append("## Candidates tried\n")
    rows = list(out.get("candidates") or [])
    if out.get("combined"):
        rows.append(out["combined"])
    if not rows:
        L.append("None: no change fix can apply on its own addressed the top findings. See the suggestions below.\n")
        if out.get("suggestions"):
            L.append("## Suggested, not applied\n")
            L.append("These findings need a source change fix does not make on its own:\n")
            for s in out["suggestions"]:
                L.append(f"- [{s['id']}] {s['title']} (risk {s['risk']}): {s['suggestion']}")
            L.append("")
        return "\n".join(L)
    L.append("| Candidate | Decision | Ratio (95% interval) | Loss | Why |\n|---|---|---|---|---|")
    for e in rows:
        ra = e.get("ratio")
        rs = f"{ra['point']:.2f}x ({ra['low']:.2f} to {ra['high']:.2f})" if ra else "n/a"
        lc = (e.get("loss_check") or {}).get("status", "n/a")
        L.append(f"| {e['title']} | {e['decision']} | {rs} | {lc} | {e.get('reason')} |")
    L.append("")
    if out.get("suggestions"):
        L.append("## Suggested, not applied\n")
        L.append("These findings need a source change fix does not make on its own:\n")
        for s in out["suggestions"]:
            L.append(f"- [{s['id']}] {s['title']} (risk {s['risk']}): {s['suggestion']}")
        L.append("")
    return "\n".join(L)
