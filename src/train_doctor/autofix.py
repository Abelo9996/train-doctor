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

from train_doctor import agentdrv, api, card, patcher, runner
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


def _label_for(argv: list[str], cwd: str = ".") -> str:
    for a in argv:
        if a.endswith(".py"):
            p = Path(a)
            # train.py / main.py in a project directory reads better as the directory name
            if p.stem in _GENERIC_STEMS:
                parent = p.parent.name or Path(cwd, a).resolve().parent.name
                if parent and parent not in (".", ""):
                    return parent
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
                {
                    "id": fid,
                    "title": f["title"],
                    "suggestion": f["suggestion"],
                    "risk": f["risk"],
                    "reason": "needs a source change fix does not make on its own",
                }
            )
        seen_ids.add(fid)
    return cands[:budget], suggestions


@dataclass
class AgentCand:
    id: str
    title: str
    instruction: str
    from_finding: str
    risk: str


def agent_candidates_from_findings(ev: dict, findings: list[dict], limit: int = 3) -> list[AgentCand]:
    """Source-edit changes an agent can implement, mapped from the findings, in rank order.

    These are the changes ``fix`` will not make blindly (mixed precision, torch.compile, a device
    move, removing per-step syncs); ``fix --agent`` asks the user's agent CLI to make one at a time.
    """
    device = (ev.get("config") or {}).get("device")
    tmeta = (ev.get("config") or {}).get("torch") or {}
    out: list[AgentCand] = []
    for f in findings:
        fid = f["id"]
        if fid == "no_mixed_precision":
            if device == "cuda":
                cap = tmeta.get("cuda_capability") or [0, 0]
                dtype = "bfloat16" if cap[0] >= 8 else "float16"
            else:
                dtype = "float16"
            instr = (
                f"enable automatic mixed precision by wrapping the model forward pass and the loss computation inside the "
                f"training step with `with torch.autocast(device_type='{device}', dtype=torch.{dtype}):`."
            )
            if device == "cuda" and dtype == "float16":
                instr += " Also scale the loss for the backward pass with torch.cuda.amp.GradScaler."
            out.append(AgentCand("amp", f"mixed precision (autocast {dtype})", instr, fid, f["risk"]))
        elif fid == "no_torch_compile":
            out.append(
                AgentCand(
                    "torch_compile",
                    "torch.compile(model)",
                    "after the model is created and moved to its device, compile it with `model = torch.compile(model)` and train with the compiled model.",
                    fid,
                    f["risk"],
                )
            )
        elif fid == "host_sync_in_loop":
            out.append(
                AgentCand(
                    "remove_sync",
                    "keep metrics on device, read them less often",
                    "stop reading device tensors every step with .item()/.cpu()/float(): accumulate the running loss (and any metrics) on the device and only move it to Python with .item() when logging, every 50 steps.",
                    fid,
                    f["risk"],
                )
            )
        elif fid == "cpu_with_accelerator_available":
            avail = (f.get("numbers") or {}).get("available")
            out.append(
                AgentCand(
                    "device_move",
                    f"move training to {avail}",
                    f"change the device selection so the model and every input batch are moved to torch.device('{avail}') instead of cpu.",
                    fid,
                    f["risk"],
                )
            )
    # try the changes least likely to move the loss first (same math), then the numerics-changing ones
    safety = {"remove_sync": 0, "torch_compile": 1, "amp": 2, "device_move": 3}
    out.sort(key=lambda c: safety.get(c.id, 9))
    return out[:limit]


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
    agent: str | None = None,
    yes: bool = False,
    agent_timeout: float = 420.0,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Profile ``command``, measure each applicable change, and keep the proven winners.

    With ``agent`` set to ``"claude"`` or ``"codex"``, the source-edit findings (mixed precision,
    torch.compile, a device move, removing syncs) are also tried by asking that agent CLI to make one
    change at a time in a scratch copy, each proven with ``compare``. Without ``yes`` this only prints
    the plan and cost and spends nothing.
    """
    argv = api.as_argv(command)

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    argv, interp_note = runner.resolve_target_interpreter(argv, cwd)
    if interp_note:
        say(interp_note)
    label = _label_for(argv, cwd)

    def do_profile() -> dict:
        return api.profile(
            argv,
            cwd=cwd,
            warmup_steps=warmup_steps,
            steps=steps,
            seconds=seconds,
            timeout=timeout,
            profiler_steps=0,
            out_dir=out_dir,
            label=f"{label}-fix-profile",
        )

    say(f"Profiling {shlex.join(argv)}")
    prof = do_profile()
    out: dict = {
        "kind": "fix",
        "label": label,
        "command": argv,
        "cwd": cwd,
        "profile_dir": prof["run_dir"],
        "candidates": [],
        "suggestions": [],
        "proven": None,
        "machine": prof["evidence"].get("machine", {}),
    }
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
            out["headline"] = f"{label}: could not profile the command. {prof['error'].get('hint')}" + (
                f" To let fix install it, re-run with --install-missing (it would run `{fixcmd}`)."
                if missing and not install_missing
                else ""
            )
            _write(out_dir, out, label)
            return out

    findings = prof["findings"]
    cands, suggestions = candidates_from_findings(prof["evidence"], findings, budget)
    out["suggestions"] = suggestions
    agent_cands = agent_candidates_from_findings(prof["evidence"], findings, limit=budget) if agent else []
    cmp = dict(
        repeats=repeats,
        warmup_runs=warmup_runs,
        warmup_steps=warmup_steps,
        steps=steps,
        seconds=seconds,
        timeout=timeout,
        seed=seed,
        loss_tol=loss_tol,
        min_effect=min_effect,
        out_dir=out_dir,
    )
    prog = (lambda d, t, m: say("  " + m)) if progress else None

    # plan-only: --agent without --yes prints what it would do and spends nothing
    if agent and not yes:
        out["mode"] = "plan"
        out["agent"] = agent
        out["agent_plan"] = {
            "injection_candidates": [{"id": c.id, "describe": c.describe} for c in cands],
            "agent_candidates": [
                {"id": a.id, "title": a.title, "from_finding": a.from_finding, "risk": a.risk, "instruction": a.instruction}
                for a in agent_cands
            ],
        }
        n = len(agent_cands)
        out["headline"] = (
            f"{label}: plan only (no spend). Would try {len(cands)} injected change(s) and ask {agent} for "
            f"{n} source change(s), one at a time, each proven with a paired compare. Re-run with --yes to spend."
        )
        out["cost_note"] = (
            f"Cost: {n} headless {agent} session(s) (roughly a few cents to about $0.30 each, model dependent) "
            f"plus {len(cands) + n + (1 if n else 0)} paired compare(s), which have no agent cost. Nothing was run."
        )
        _write(out_dir, out, label)
        return out

    if not cands and not agent_cands:
        out["headline"] = f"{label}: no proven win. No change fix can apply on its own addressed the top findings."
        out["next_step"] = prof["next_step"]
        _write(out_dir, out, label)
        return out

    # --- injected changes (no source edit, no agent) ---
    results: list[dict] = []
    winners: list[tuple[Candidate, dict]] = []
    for c in cands:
        say(f"Trying {c.title}")
        res = api.compare(argv, argv, cwd=cwd, candidate_env=c.env, label=f"{label}-fix-{c.id}", progress=prog, **cmp)
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
        res = api.compare(argv, argv, cwd=cwd, candidate_env=env, label=f"{label}-fix-combined", progress=prog, **cmp)
        combo = Candidate(
            id="combined",
            title="all winners together",
            from_finding="+".join(c.id for c, _ in winners),
            env=env,
            edits=[e for c, _ in winners for e in c.edits],
            risk="low",
            describe=", ".join(c.describe for c, _ in winners),
        )
        entry = _result_entry(combo, res)
        entry["combined_of"] = [c.id for c, _ in winners]
        out["combined"] = entry
        say(f"  combined: {entry['decision']} ({entry.get('reason')})")
        if res.get("decision") == "keep":
            combined = (combo, res)

    kept = list(winners) + ([combined] if combined else [])
    inj_best = max(kept, key=lambda cr: cr[1]["ratio"]["point"]) if kept else None
    inj_edits = inj_best[0].edits if inj_best else []

    # --- agent-driven source changes, stacked on top of any injected winners ---
    agent_out = None
    if agent and yes and agent_cands:
        agent_out = _run_agent_stack(argv, cwd, label, agent, agent_cands, inj_edits, say, cmp, agent_timeout, prog)
        out["agent"] = agent
        out["agent_results"] = agent_out.get("results", [])
        if agent_out.get("error"):
            out["agent_error"] = agent_out["error"]

    _finalize(out, argv, cwd, label, inj_best, agent_out, keep, make_card, out_dir, say, cmp, prog)
    _write(out_dir, out, label)
    return out


def _swap_script(argv: list[str], path) -> list[str]:
    swapped = False
    cmd = []
    for a in argv:
        if not swapped and a.endswith(".py"):
            cmd.append(str(path))
            swapped = True
        else:
            cmd.append(a)
    return cmd


def _run_agent_stack(argv, cwd, label, agent, agent_cands, inj_edits, say, cmp, agent_timeout, prog) -> dict:
    """Ask the agent for one source edit at a time in a scratch copy, keeping only proven winners.

    The base scratch starts as the original plus any injected winners expressed as source edits, so
    agent changes stack on top. Each candidate is compared against the current accepted base; a reject
    leaves the base untouched (the edit is reverted).
    """
    import shutil

    script = patcher._target_file(argv, cwd)
    if script is None:
        return {"error": "could not find the .py script to edit", "results": [], "winners": [], "final_text": None}
    original_text = script.read_text()
    base_text, _ = patcher.apply_edits_to_text(original_text, inj_edits)
    base_path = script.with_name(f"{script.stem}.td-base.py")
    trial_path = script.with_name(f"{script.stem}.td-trial.py")
    base_path.write_text(base_text)
    results: list[dict] = []
    winners: list = []
    for ac in agent_cands:
        shutil.copyfile(base_path, trial_path)
        say(f"Asking {agent} to apply: {ac.title}")
        ed = agentdrv.edit_file(agent, trial_path, ac.instruction, cwd, agent_timeout)
        entry = {
            "candidate": ac.id,
            "title": ac.title,
            "from_finding": ac.from_finding,
            "instruction": ac.instruction,
            "agent": {k: ed.get(k) for k in ("ok", "changed", "error", "returncode")},
        }
        if not ed.get("changed"):
            entry["decision"] = "skipped"
            entry["reason"] = ed.get("error") or "the agent made no change"
            results.append(entry)
            say(f"  {ac.title}: skipped ({entry['reason']})")
            continue
        res = api.compare(
            _swap_script(argv, base_path), _swap_script(argv, trial_path), cwd=cwd, label=f"{label}-agent-{ac.id}", progress=prog, **cmp
        )
        entry.update(
            decision=res.get("decision"),
            verdict=res.get("verdict"),
            reason=res.get("reason"),
            ratio=res.get("ratio"),
            loss_check={k: (res.get("loss_check") or {}).get(k) for k in ("status", "mean_rel_diff", "final_rel_diff")},
            compare_dir=res.get("root"),
        )
        results.append(entry)
        if res.get("decision") == "keep":
            shutil.copyfile(trial_path, base_path)
            winners.append(ac)
            say(f"  {ac.title}: keep")
        else:
            say(f"  {ac.title}: {res.get('decision')} (reverted)")
    trial_path.unlink(missing_ok=True)
    return {
        "results": results,
        "winners": winners,
        "base_path": base_path,
        "script": script,
        "original_text": original_text,
        "final_text": base_path.read_text(),
    }


def _finalize(out, argv, cwd, label, inj_best, agent_out, keep, make_card, out_dir, say, cmp, prog) -> None:
    """Pick the proven result (agent-combined, else injected best, else none) and write patch and card."""
    machine = out["machine"]
    changed = (
        bool(agent_out)
        and not agent_out.get("error")
        and agent_out.get("final_text")
        and agent_out["final_text"] != agent_out["original_text"]
    )
    confirmed = None
    if changed:
        say("Confirming the combined change against the original")
        res = api.compare(
            argv, _swap_script(argv, agent_out["base_path"]), cwd=cwd, label=f"{label}-combined-confirm", progress=prog, **cmp
        )
        out["combined_confirm"] = {
            "decision": res.get("decision"),
            "reason": res.get("reason"),
            "ratio": res.get("ratio"),
            "loss_check": {k: (res.get("loss_check") or {}).get(k) for k in ("status", "mean_rel_diff", "final_rel_diff")},
            "compare_dir": res.get("root"),
        }
        if res.get("decision") == "keep":
            confirmed = res

    # the proven source text is captured in the patch; the scratch file is no longer needed
    if agent_out and agent_out.get("base_path"):
        Path(agent_out["base_path"]).unlink(missing_ok=True)
    describe_agent = ", ".join(w.title for w in (agent_out["winners"] if agent_out else []))

    if confirmed:
        inj_desc = inj_best[0].describe if inj_best else ""
        describe = ", ".join(x for x in (inj_desc, describe_agent) if x)
        out["proven"] = {
            "candidate": "combined",
            "describe": describe,
            "compare_dir": confirmed["root"],
            "ratio": confirmed["ratio"],
            "loss_check": confirmed.get("loss_check"),
        }
        out["headline"] = _headline(label, confirmed, machine)
        patch = patcher.diff_text(agent_out["script"], agent_out["original_text"], agent_out["final_text"], Path(out_dir))
        out["patch"] = patch
        if keep and patch.get("patch_path"):
            out["applied"] = patcher.apply_patch(patch)
        _card(out, label, confirmed, machine, argv, out_dir, make_card)
        return

    if inj_best:
        best_c, best_r = inj_best
        out["proven"] = {
            "candidate": best_c.id,
            "describe": best_c.describe,
            "compare_dir": best_r["root"],
            "ratio": best_r["ratio"],
            "loss_check": best_r.get("loss_check"),
        }
        out["headline"] = _headline(label, best_r, machine)
        patch = patcher.make_patch(argv, cwd, best_c.edits, Path(out_dir))
        out["patch"] = patch
        if keep and patch.get("patch_path") and patch.get("target"):
            out["applied"] = patcher.apply_patch(patch)
        _card(out, label, best_r, machine, argv, out_dir, make_card)
        if agent_out and agent_out.get("winners"):
            out["headline"] += f" ({describe_agent} passed on its own but the combined confirm did not clear the threshold.)"
        return

    tried = [f"{e['candidate']}: {e['decision']}" for e in (out.get("candidates") or [])]
    tried += [f"{e['candidate']}: {e.get('decision')}" for e in (out.get("agent_results") or [])]
    detail = f" ({'; '.join(tried)})" if tried else ""
    out["headline"] = f"{label}: no proven win.{detail}"


def _card(out, label, res, machine, argv, out_dir, make_card) -> None:
    if not make_card:
        return
    try:
        out["card"] = card.render_card(label, res, machine, argv, Path(out_dir))
    except Exception as e:
        out["card"] = {"error": repr(e)}


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


def _candidate_table(rows: list[dict]) -> list[str]:
    L = ["| Candidate | Decision | Ratio (95% interval) | Loss | Why |", "|---|---|---|---|---|"]
    for e in rows:
        ra = e.get("ratio")
        rs = f"{ra['point']:.2f}x ({ra['low']:.2f} to {ra['high']:.2f})" if ra else "n/a"
        lc = (e.get("loss_check") or {}).get("status", "n/a")
        L.append(f"| {e['title']} | {e.get('decision')} | {rs} | {lc} | {e.get('reason')} |")
    return L


def render_fix_md(out: dict) -> str:
    L: list[str] = []
    label = out.get("label", "fix")
    L.append(f"# train-doctor fix: {label}\n")
    L.append(f"Command: `{shlex.join(out.get('command') or [])}`\n")
    L.append(f"**{out.get('headline', '')}**\n")
    if out.get("error"):
        L.append(f"The command could not be profiled: {out['error'].get('hint')}\n")
        return "\n".join(L)

    if out.get("mode") == "plan":
        plan = out.get("agent_plan") or {}
        L.append(out.get("cost_note", ""))
        L.append("\n## Would inject (no agent)\n")
        for c in plan.get("injection_candidates") or ["(none)"]:
            L.append(f"- {c['describe'] if isinstance(c, dict) else c}")
        L.append(f"\n## Would ask {out.get('agent')} to make, one at a time\n")
        for a in plan.get("agent_candidates") or []:
            L.append(f"- [{a['id']}] {a['title']} (from `{a['from_finding']}`, risk {a['risk']}): {a['instruction']}")
        if not plan.get("agent_candidates"):
            L.append("- (none)")
        L.append("\nRe-run with `--yes` to spend.")
        return "\n".join(L)

    if out.get("proven"):
        p = out["proven"]
        L.append("## Proven change\n")
        L.append(f"- {p['describe']} (from `{p['candidate']}`)")
        lc = p.get("loss_check") or {}
        L.append(f"- Loss check: {lc.get('status')}")
        ra = p["ratio"]
        L.append(f"- Speed: {ra['point']:.2f}x (95% interval {ra['low']:.2f} to {ra['high']:.2f}), measured over {ra['n_pairs']} pairs")
        L.append(f"- Evidence: `{p['compare_dir']}/report.md`")
        if out.get("patch"):
            pt = out["patch"]
            if pt.get("patch_path"):
                L.append(
                    f"- Patch: `{pt['patch_path']}`"
                    + (
                        f" (applied to `{pt.get('target')}`)"
                        if out.get("applied", {}).get("applied")
                        else " (apply with `git apply` or `patch -p0`; reverse with `-R`)"
                    )
                )
            else:
                L.append(f"- Patch: not generated automatically ({pt.get('reason')}). Make this edit by hand: {p['describe']}.")
        L.append("")

    rows = list(out.get("candidates") or [])
    if out.get("combined"):
        rows.append(out["combined"])
    L.append("## Injected candidates tried\n")
    if rows:
        L += _candidate_table(rows)
    else:
        L.append("None applicable: the top findings need source changes.")
    L.append("")

    if out.get("agent") and out.get("agent_results") is not None:
        L.append(f"## Source changes by {out['agent']} (one at a time)\n")
        ar = out.get("agent_results") or []
        if ar:
            L += _candidate_table(ar)
        else:
            L.append("None applicable.")
        cc = out.get("combined_confirm")
        if cc and cc.get("ratio"):
            ra = cc["ratio"]
            L.append(
                f"\nCombined vs original: {cc['decision']}, {ra['point']:.2f}x ({ra['low']:.2f} to {ra['high']:.2f}), loss {(cc.get('loss_check') or {}).get('status')}."
            )
        L.append("")

    if out.get("suggestions"):
        L.append("## Suggested, not applied\n")
        L.append("These findings need a source change fix does not make on its own:\n")
        for s in out["suggestions"]:
            L.append(f"- [{s['id']}] {s['title']} (risk {s['risk']}): {s['suggestion']}")
        L.append("")
    return "\n".join(L)
