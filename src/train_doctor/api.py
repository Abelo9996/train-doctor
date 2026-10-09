"""Core operations shared by the CLI and the MCP server.

``profile`` runs one command and diagnoses it. ``compare`` runs two commands
as back-to-back pairs and decides whether the candidate is
faster without breaking training. ``report`` renders Markdown and JSON for a
finished run directory.
"""

from __future__ import annotations

import datetime as _dt
import random
import shlex
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from train_doctor import lossdiff, report
from train_doctor.evidence import write_evidence
from train_doctor.machine import command_provenance, machine_info
from train_doctor.redact import dump_json
from train_doctor.rules import diagnose as run_rules
from train_doctor.runner import CommandError, RunConfig, not_found_hint, resolve_executable, run
from train_doctor.stats import describe, median, paired_analysis, stdev, verdict

DEFAULT_OUT = ".train-doctor/runs"


def as_argv(cmd: str | list[str]) -> list[str]:
    if isinstance(cmd, str):
        return shlex.split(cmd)
    return list(cmd)


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in label.lower()).strip("-")[:40]


def new_run_dir(out_dir: str | Path, kind: str, label: str | None = None) -> Path:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if label:
        kind = f"{kind}-{_slug(label)}"
    d = base / f"{stamp}-{kind}"
    i = 1
    while d.exists():
        i += 1
        d = base / f"{stamp}-{kind}-{i}"
    d.mkdir(parents=True)
    return d


def latest_run(out_dir: str | Path, kind: str | None = None) -> Path | None:
    base = Path(out_dir)
    if not base.exists():
        return None
    runs = sorted(p for p in base.iterdir() if p.is_dir() and (kind is None or p.name.split("-")[2:3] == [kind]))
    return runs[-1] if runs else None


def resolve_run(run_dir: str | Path | None, out_dir: str | Path = DEFAULT_OUT, kind: str | None = None) -> Path:
    if run_dir:
        p = Path(run_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"run directory not found: {p}")
        return p
    p = latest_run(out_dir, kind)
    if p is None:
        raise FileNotFoundError(f"no {kind or ''} runs under {out_dir}; run `train-doctor profile -- <command>` first")
    return p


def _single_run(cfg: RunConfig, run_dir: Path, machine: dict, kind: str, label: str | None = None) -> dict:
    meta = run(cfg, run_dir)
    meta["kind"] = kind
    meta["label"] = label
    meta["machine"] = machine
    meta["provenance"] = command_provenance(cfg.cmd, cfg.cwd)
    dump_json(run_dir / "run.json", meta)
    return write_evidence(run_dir)


# --------------------------------------------------------------------- profile
def profile(
    cmd: str | list[str],
    *,
    cwd: str = ".",
    warmup_steps: int = 5,
    steps: int = 50,
    seconds: float = 2.0,
    timeout: float = 600.0,
    profiler_steps: int = 5,
    py_spy: bool = False,
    seed: int | None = None,
    out_dir: str | Path = DEFAULT_OUT,
    echo: bool = False,
    label: str | None = None,
) -> dict:
    argv = as_argv(cmd)
    run_dir = new_run_dir(out_dir, "profile", label)
    cfg = RunConfig(
        cmd=argv,
        cwd=cwd,
        warmup_steps=warmup_steps,
        steps=steps,
        seconds=seconds,
        timeout=timeout,
        profiler_steps=profiler_steps,
        py_spy=py_spy,
        seed=seed,
        echo=echo,
    )
    try:
        ev = _single_run(cfg, run_dir, machine_info(), "profile", label)
    except CommandError:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    findings = [f.to_dict() for f in run_rules(ev)]
    dump_json(run_dir / "findings.json", findings)
    report.write(run_dir)
    out = {"run_dir": str(run_dir), "evidence": ev, "findings": findings, "next_step": profile_next_step(argv, ev, findings)}
    if ev.get("failure"):
        out["error"] = ev["failure"]
    return out


def profile_next_step(argv: list[str], ev: dict, findings: list[dict]) -> str:
    """One sentence that tells a person or an agent what to do after a profile."""
    if ev.get("failure"):
        return "The command failed before the first step: " + ev["failure"]["hint"]
    if ev.get("source") == "none":
        return (
            "No training steps were seen. Make sure the loop calls optimizer.step(), or add "
            "train_doctor.step(samples=..., loss=...) at the end of each step, or print a line containing 'loss' each step."
        )
    safe = [f for f in findings if f["risk"] != "high"]
    if not safe:
        return "No rule fired: the loop has no obvious waste the rules can see. Further gains need model or algorithm changes."
    top = safe[0]
    cmd = shlex.join(argv)
    change = top["suggestion"].split(". ")[0].rstrip(".")
    return (
        f"Make one change for [{top['id']}]: {change[:1].lower()}{change[1:]}. Do it behind a flag or in a copy so the "
        f"original still runs, then measure it with compare (CLI: train-doctor compare --baseline {shlex.quote(cmd)} "
        f'--candidate "<changed command>" --label {top["id"]}). Keep it only if the decision is keep.'
    )


def diagnose(run_dir: str | Path | None = None, out_dir: str | Path = DEFAULT_OUT) -> dict:
    d = resolve_run(run_dir, out_dir, "profile")
    ev = write_evidence(d)
    findings = [f.to_dict() for f in run_rules(ev)]
    dump_json(d / "findings.json", findings)
    return {"run_dir": str(d), "findings": findings, "limits": ev.get("limits", []), "source": ev.get("source")}


# --------------------------------------------------------------------- compare
def _metric_value(ev: dict, metric: str) -> float | None:
    tp = ev.get("throughput") or {}
    v = tp.get(metric)
    return float(v) if v else None


def _usable(ev: dict) -> str | None:
    """None if a run's throughput can be used, else the reason it can't."""
    if not (ev.get("throughput") or {}).get("steps_per_s"):
        return "no throughput"
    st = ev.get("steps") or {}
    if ev.get("source") == "hook" and st.get("end_reason") != "window_complete":
        if st.get("end_reason") == "process_exit" and (ev.get("run") or {}).get("exit_code") == 0:
            return (
                f"the script finished on its own after {st.get('total')} steps, before the measurement window closed; "
                "let it run more steps (for example a larger max-steps or epochs setting, the same in both commands) or lower steps/seconds"
            )
        return f"the measurement window did not complete ({st.get('end_reason')})"
    return None


def compare(
    baseline: str | list[str],
    candidate: str | list[str],
    *,
    repeats: int = 5,
    warmup_runs: int = 1,
    cwd: str = ".",
    warmup_steps: int = 5,
    steps: int = 50,
    seconds: float = 2.0,
    timeout: float = 600.0,
    seed: int = 0,
    loss_tol: float = 0.05,
    min_effect: float = 0.02,
    order_seed: int = 0,
    out_dir: str | Path = DEFAULT_OUT,
    echo: bool = False,
    label: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Run baseline and candidate as ``repeats`` back-to-back pairs and judge the candidate.

    The order inside a pair alternates (baseline first, then candidate first, ...), so slow
    drift in machine speed cancels out across neighboring pairs. Each pair gives one
    candidate/baseline throughput ratio; the verdict comes from the median pair ratio and its
    bootstrap interval, with stalled pairs set aside and reported (see ``stats.paired_analysis``).
    ``progress(done, total, message)`` is called after every run.
    """
    if repeats < 2:
        raise ValueError("repeats must be at least 2 to estimate spread")
    arms = {"baseline": as_argv(baseline), "candidate": as_argv(candidate)}
    for arm, argv in arms.items():
        if not argv or resolve_executable(argv, cwd) is None:
            raise CommandError(f"{arm} command not found: {argv[:1]!r}.{not_found_hint(argv)}")
    root = new_run_dir(out_dir, "compare", label)
    machine = machine_info()

    def cfg_for(arm: str) -> RunConfig:
        return RunConfig(
            cmd=arms[arm],
            cwd=cwd,
            warmup_steps=warmup_steps,
            steps=steps,
            seconds=seconds,
            timeout=timeout,
            profiler_steps=0,
            seed=seed,
            echo=echo,
        )

    # order_seed only picks which arm goes first in the first pair; after that the order alternates
    first = ["baseline", "candidate"] if random.Random(order_seed).random() >= 0.5 else ["candidate", "baseline"]
    schedule: list[tuple[str, str, int]] = []  # (phase, arm, index)
    for i in range(warmup_runs):
        schedule += [("warmup", arm, i) for arm in (first if i % 2 == 0 else first[::-1])]
    for i in range(repeats):
        schedule += [("measure", arm, i) for arm in (first if i % 2 == 0 else first[::-1])]

    runs: list[dict] = []
    started = time.monotonic()
    for order, (phase, arm, idx) in enumerate(schedule):
        sub = root / ("warmup" if phase == "warmup" else arm) / (f"{arm}-{idx}" if phase == "warmup" else f"rep-{idx}")
        t0 = time.monotonic()
        ev = _single_run(cfg_for(arm), sub, machine, "compare-run")
        runs.append({"order": order, "phase": phase, "arm": arm, "index": idx, "dir": str(sub.relative_to(root)), "evidence": ev})
        if progress:
            progress(order + 1, len(schedule), _progress_line(order, schedule, ev, time.monotonic() - t0, time.monotonic() - started))

    measured = [r for r in runs if r["phase"] == "measure"]
    errors = []
    for r in measured:
        why = _usable(r["evidence"])
        if why:
            errors.append(
                f"{r['arm']} repeat {r['index']}: {why} (exit code {r['evidence']['run'].get('exit_code')}); see {r['dir']}/stderr.log"
            )
    by_pair: dict[int, dict[str, dict]] = {}
    for r in measured:
        if _usable(r["evidence"]) is None:
            by_pair.setdefault(r["index"], {})[r["arm"]] = r
    pairs = [by_pair[i] for i in sorted(by_pair) if len(by_pair[i]) == 2]
    ok = {arm: [p[arm] for p in pairs] for arm in arms}

    both_samples = bool(pairs) and all(_metric_value(r["evidence"], "samples_per_s") for arm in arms for r in ok[arm])
    metric = "samples_per_s" if both_samples else "steps_per_s"
    values = {arm: [_metric_value(r["evidence"], metric) for r in ok[arm]] for arm in arms}

    result: dict = {
        "kind": "compare",
        "label": label,
        "root": str(root),
        "commands": {arm: arms[arm] for arm in arms},
        "settings": {
            "repeats": repeats,
            "warmup_runs": warmup_runs,
            "warmup_steps": warmup_steps,
            "steps": steps,
            "seconds": seconds,
            "seed": seed,
            "loss_tol": loss_tol,
            "min_effect": min_effect,
            "order_seed": order_seed,
            "design": "paired: each repeat runs baseline and candidate back to back, order alternating between pairs",
            "interval": (
                "95% percentile bootstrap of the median pair ratio (candidate / baseline), 10000 resamples, seed 0; "
                "stalled pairs set aside first (a run under half its arm's median and beyond 4 MADs, or under a quarter of it), at most 1 per 4 pairs"
            ),
        },
        "machine": machine,
        "metric": metric,
        "runs": [
            {k: v for k, v in r.items() if k != "evidence"}
            | {
                "value": _metric_value(r["evidence"], metric),
                "exit_code": r["evidence"]["run"].get("exit_code"),
                "source": r["evidence"].get("source"),
                "time_to_first_step_s": r["evidence"]["run"].get("time_to_first_step_s"),
                "sys_cpu_pct_mean": (r["evidence"].get("resources") or {}).get("sys_cpu_pct_mean"),
                "load_avg_1m": r["evidence"]["run"].get("load_avg_1m"),
            }
            for r in runs
        ],
        "errors": errors,
        "limits": [],
    }
    if metric == "steps_per_s":
        result["limits"].append(
            "Samples per step were not detected in every run, so throughput is compared in steps per second. That is only fair if both commands do the same work per step."
        )
    if len(pairs) < 2:
        result["verdict"] = "error"
        result["decision"] = "inconclusive"
        result["reason"] = "fewer than 2 pairs where both runs succeeded"
        result["next_step"] = (
            "Most repeats failed. Read the errors and the stderr.log files they point to, fix the command, and compare again."
        )
        _write_compare(root, result)
        return result

    b, c = values["baseline"], values["candidate"]
    pa = paired_analysis(b, c)
    for p, pr in zip(pa["pairs"], pairs, strict=True):
        p["pair"] = pr["baseline"]["index"]
    ra = pa["ratio"]
    point, lo, hi = ra["point"], ra["low"], ra["high"]
    v = verdict(lo, hi, min_effect)
    result["throughput"] = {"baseline": describe(b), "candidate": describe(c)}
    result["ratio"] = {"point": point, "low": lo, "high": hi, "n_pairs": ra["n_pairs"]}
    if "ratio_all_pairs" in pa:
        result["ratio_all_pairs"] = pa["ratio_all_pairs"]
    result["pairs"] = pa["pairs"]
    result["wins"] = {"candidate_faster": pa["wins"], "pairs": pa["n_pairs"]}
    result["wilcoxon"] = pa["wilcoxon"]
    result["stalls"] = {
        "pairs": [p["pair"] for p in pa["pairs"] if p["stalled"]],
        "set_aside": [p["pair"] for p in pa["pairs"] if p["excluded"]],
        "max_set_aside": pa["max_excluded"],
    }
    result["verdict"] = v

    # loss trajectories: first successful repeat of each arm, plus the baseline's own repeat-to-repeat difference
    loss = lossdiff.diff(ok["baseline"][0]["evidence"], ok["candidate"][0]["evidence"], loss_tol)
    if len(ok["baseline"]) >= 2:
        floor = lossdiff.diff(ok["baseline"][0]["evidence"], ok["baseline"][1]["evidence"], loss_tol)
        loss["baseline_self_diff"] = {k: floor.get(k) for k in ("status", "max_rel_diff", "mean_rel_diff", "final_rel_diff")}
    result["loss_check"] = loss

    wins = f"won {pa['wins']} of {pa['n_pairs']} pairs"
    if v == "faster" and loss["status"] in ("identical", "within tolerance"):
        decision = "keep"
        reason = (
            f"candidate is faster ({point:.2f}x, 95% interval {lo:.2f} to {hi:.2f}, {wins}) and the loss trajectory is {loss['status']}"
        )
    elif v == "faster" and loss["status"] == "unavailable":
        decision = "inconclusive"
        reason = f"candidate is faster ({point:.2f}x, {wins}) but no loss values were recorded, so training equivalence is unchecked"
    elif v == "faster":
        decision = "reject"
        reason = f"candidate is faster ({point:.2f}x) but the loss trajectory is outside the {loss_tol:.0%} tolerance (mean relative difference {loss['mean_rel_diff']:.1%})"
    elif v == "slower":
        decision = "reject"
        reason = f"candidate is slower ({point:.2f}x, 95% interval {lo:.2f} to {hi:.2f}, {wins})"
    else:
        decision = "inconclusive"
        reason = f"no clear difference: the 95% interval {lo:.2f} to {hi:.2f} does not clear the {min_effect:.0%} threshold on either side ({wins})"
    result["decision"] = decision
    result["reason"] = reason

    busy = False
    unit = "samples/s" if metric == "samples_per_s" else "steps/s"
    meds = {arm: median(values[arm]) for arm in arms}
    for p in pa["pairs"]:
        for arm in p["stalled"]:
            p.setdefault("stall_notes", []).append(
                f"{arm} repeat {p['pair']} ran at {p[arm]:.4g} {unit}, {meds[arm] / p[arm]:.1f} times slower than that arm's median {meds[arm]:.4g}"
            )
    set_aside = [p for p in pa["pairs"] if p["excluded"]]
    if set_aside:
        notes = "; ".join(n for p in set_aside for n in p["stall_notes"])
        al = pa["ratio_all_pairs"]
        result["limits"].append(
            f"Set aside {len(set_aside)} stalled pair(s) out of {len(pa['pairs'])}: {notes}. Every run in an arm is the same command, "
            f"so a gap that large is the machine (another process, swapping, sleep), not the change. With all pairs the interval "
            f"would be {al['low']:.2f} to {al['high']:.2f}."
        )
        if any("candidate" in p["stalled"] for p in set_aside) and v == "faster":
            result["limits"].append(
                "A set-aside stall was in a candidate run. If the candidate stalls again when you rerun, the change itself may cause it "
                "(for example recompiling or starting workers), so don't keep calling it noise."
            )
    kept_stalls = [p for p in pa["pairs"] if p["stalled"] and not p["excluded"]]
    if kept_stalls:
        busy = True
        notes = "; ".join(n for p in kept_stalls for n in p["stall_notes"])
        result["limits"].append(
            f"{len(kept_stalls)} of {len(pa['pairs'])} pairs look stalled ({notes}). That is more than the "
            f"{pa['max_excluded']} that can be set aside with {len(pa['pairs'])} pairs, so they stay in and widen the interval."
        )
    kept_logs = [p["log_ratio"] for p in pa["pairs"] if not p["excluded"]]
    if len(kept_logs) >= 3 and stdev(kept_logs) > 0.1:
        kr = [p["ratio"] for p in pa["pairs"] if not p["excluded"]]
        result["limits"].append(
            f"Pair ratios vary a lot (from {min(kr):.2f} to {max(kr):.2f}), so the machine's speed changed between pairs; close other apps for a firmer answer."
        )
    loads = [r["evidence"]["run"].get("load_avg_1m") for r in measured]
    loads = [x for x in loads if x is not None]
    cores = machine.get("cores_logical")
    if loads and cores and max(loads) > cores:
        busy = True
        result["limits"].append(
            f"The machine was busy: the 1-minute load average reached {max(loads):.1f} on {cores} logical cores during the runs, "
            "so other processes competed for CPU. Pairing cancels slow stretches that hit both runs of a pair, not ones that hit only one. "
            "Rerun when the machine is quieter for a firmer answer."
        )
    ttfs = {arm: [r["evidence"]["run"].get("time_to_first_step_s") for r in ok[arm]] for arm in arms}
    if all(all(x is not None for x in v) and v for v in ttfs.values()):
        result["time_to_first_step_s"] = {arm: median(v) for arm, v in ttfs.items()}
        extra = result["time_to_first_step_s"]["candidate"] - result["time_to_first_step_s"]["baseline"]
        if extra > 1.0:
            result["limits"].append(
                f"The candidate takes {extra:.1f} s longer to reach its first step (median). That one-time cost is outside the throughput window; weigh it against the run length."
            )
        stalled = [
            f"{arm} repeat {r['index']}"
            for arm in arms
            for r in ok[arm]
            if r["evidence"]["run"]["time_to_first_step_s"] > max(3 * result["time_to_first_step_s"][arm], 5.0)
        ]
        if stalled:
            busy = True
            result["limits"].append(
                f"{', '.join(stalled)} took over 3 times the usual time to reach the first step, a sign the machine was stalled "
                "by other work. Rerun when the machine is quieter if the verdict matters."
            )
    if machine.get("gpu_backend") == "apple":
        result["limits"].append(
            "Apple laptops change clock speeds with temperature and power state; pairing and alternating the order reduce but don't remove that drift."
        )
    result["next_step"] = compare_next_step(decision, v, loss["status"], len(pa["pairs"]), busy)
    _write_compare(root, result)
    return result


def _progress_line(order: int, schedule: list[tuple[str, str, int]], ev: dict, took: float, elapsed: float) -> str:
    phase, arm, idx = schedule[order]
    total = len(schedule)
    measured = sum(1 for s in schedule if s[0] == "measure") // 2
    where = f"warmup {arm}" if phase == "warmup" else f"pair {idx + 1} of {measured}, {arm}"
    tp = ev.get("throughput") or {}
    if tp.get("samples_per_s"):
        got = f"{tp['samples_per_s']:.1f} samples/s"
    elif tp.get("steps_per_s"):
        got = f"{tp['steps_per_s']:.2f} steps/s"
    else:
        got = "no throughput (see its stderr.log)"
    left = total - order - 1
    eta = f", about {elapsed / (order + 1) * left:.0f} s left" if left else ", done"
    return f"[{order + 1}/{total}] {where}: {got} in {took:.1f} s{eta}"


def compare_next_step(decision: str, verdict_: str, loss_status: str, n: int, busy: bool = False) -> str:
    if decision == "keep":
        return "Keep the change. Use the candidate as the new baseline and profile it again: the bottleneck usually moves."
    if decision == "reject" and verdict_ == "slower":
        return "Revert the change: the candidate is slower."
    if decision == "reject":
        return (
            "Revert the change, or keep it only if you accept different training: the loss trajectory moved beyond the "
            "tolerance. Say so if you report it."
        )
    if verdict_ == "faster" and loss_status == "unavailable":
        return (
            "Speed looks better but training equivalence is unchecked. Make the script report its loss (print a line "
            "containing 'loss', or train_doctor.step(loss=...)) and compare again."
        )
    if busy:
        return (
            "No clear difference, and other work on the machine disturbed the runs (see limits). More repeats rarely fix "
            "that: rerun this compare when the machine is quieter, or drop the change. Don't report it as faster."
        )
    if n >= 9:
        return (
            f"No clear difference with {n} repeats per arm. Drop the change, or accept that any gain is smaller than "
            "this machine's run-to-run noise. Don't report it as faster."
        )
    return (
        "No clear difference. Either drop the change, or compare again with repeats=9 (and seconds=4) "
        "if you expect a small but real gain. Don't report it as faster."
    )


def _write_compare(root: Path, result: dict) -> None:
    dump_json(root / "compare.json", result)
    report.write(root)


# ---------------------------------------------------------------------- report
def make_report(run_dir: str | Path | None = None, out_dir: str | Path = DEFAULT_OUT) -> dict:
    d = resolve_run(run_dir, out_dir)
    paths = report.write(d)
    return {"run_dir": str(d), **{k: str(v) for k, v in paths.items()}}
