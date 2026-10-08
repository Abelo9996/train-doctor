"""Core operations shared by the CLI and the MCP server.

``profile`` runs one command and diagnoses it. ``compare`` runs two commands
several times in interleaved order and decides whether the candidate is
faster without breaking training. ``report`` renders Markdown and JSON for a
finished run directory.
"""

from __future__ import annotations

import datetime as _dt
import random
import shlex
from pathlib import Path

from train_doctor import lossdiff, report
from train_doctor.evidence import write_evidence
from train_doctor.machine import command_provenance, machine_info
from train_doctor.redact import dump_json
from train_doctor.rules import diagnose as run_rules
from train_doctor.runner import RunConfig, run
from train_doctor.stats import bootstrap_ratio_ci, describe, mann_whitney_u, verdict

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
    ev = _single_run(cfg, run_dir, machine_info(), "profile", label)
    findings = [f.to_dict() for f in run_rules(ev)]
    dump_json(run_dir / "findings.json", findings)
    report.write(run_dir)
    return {"run_dir": str(run_dir), "evidence": ev, "findings": findings}


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
    if ev.get("source") == "hook" and (ev.get("steps") or {}).get("end_reason") != "window_complete":
        return f"the measurement window did not complete ({(ev.get('steps') or {}).get('end_reason')})"
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
    progress=None,
) -> dict:
    """Run both commands ``repeats`` times each in randomized pair order and judge the candidate."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2 to estimate spread")
    arms = {"baseline": as_argv(baseline), "candidate": as_argv(candidate)}
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

    rng = random.Random(order_seed)
    schedule: list[tuple[str, str, int]] = []  # (phase, arm, index)
    for i in range(warmup_runs):
        schedule += [("warmup", "baseline", i), ("warmup", "candidate", i)]
    for i in range(repeats):
        pair = ["baseline", "candidate"]
        if rng.random() < 0.5:
            pair.reverse()
        schedule += [("measure", pair[0], i), ("measure", pair[1], i)]

    runs: list[dict] = []
    for order, (phase, arm, idx) in enumerate(schedule):
        sub = root / ("warmup" if phase == "warmup" else arm) / (f"{arm}-{idx}" if phase == "warmup" else f"rep-{idx}")
        if progress:
            progress(f"[{order + 1}/{len(schedule)}] {phase} {arm} #{idx}")
        ev = _single_run(cfg_for(arm), sub, machine, "compare-run")
        runs.append({"order": order, "phase": phase, "arm": arm, "index": idx, "dir": str(sub.relative_to(root)), "evidence": ev})

    measured = [r for r in runs if r["phase"] == "measure"]
    ok = {arm: [r for r in measured if r["arm"] == arm and _usable(r["evidence"]) is None] for arm in arms}
    errors = []
    for r in measured:
        why = _usable(r["evidence"])
        if why:
            errors.append(
                f"{r['arm']} repeat {r['index']}: {why} (exit code {r['evidence']['run'].get('exit_code')}); see {r['dir']}/stderr.log"
            )

    both_samples = all(_metric_value(r["evidence"], "samples_per_s") for arm in arms for r in ok[arm]) and all(ok.values())
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
            "interval": "95% percentile bootstrap of the ratio of medians, 10000 resamples, seed 0",
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
    if any(len(values[a]) < 2 for a in arms):
        result["verdict"] = "error"
        result["decision"] = "inconclusive"
        result["reason"] = "fewer than 2 successful repeats in an arm"
        _write_compare(root, result)
        return result

    b, c = values["baseline"], values["candidate"]
    point, lo, hi = bootstrap_ratio_ci(b, c)
    u, p = mann_whitney_u(b, c)
    v = verdict(lo, hi, min_effect)
    result["throughput"] = {"baseline": describe(b), "candidate": describe(c)}
    result["ratio"] = {"point": point, "low": lo, "high": hi}
    result["mann_whitney"] = {"u": u, "p_two_sided": p, "n_baseline": len(b), "n_candidate": len(c)}
    result["verdict"] = v

    # loss trajectories: first successful repeat of each arm, plus the baseline's own repeat-to-repeat difference
    loss = lossdiff.diff(ok["baseline"][0]["evidence"], ok["candidate"][0]["evidence"], loss_tol)
    if len(ok["baseline"]) >= 2:
        floor = lossdiff.diff(ok["baseline"][0]["evidence"], ok["baseline"][1]["evidence"], loss_tol)
        loss["baseline_self_diff"] = {k: floor.get(k) for k in ("status", "max_rel_diff", "mean_rel_diff", "final_rel_diff")}
    result["loss_check"] = loss

    if v == "faster" and loss["status"] in ("identical", "within tolerance"):
        decision = "keep"
        reason = f"candidate is faster ({point:.2f}x, 95% interval {lo:.2f} to {hi:.2f}) and the loss trajectory is {loss['status']}"
    elif v == "faster" and loss["status"] == "unavailable":
        decision = "inconclusive"
        reason = f"candidate is faster ({point:.2f}x) but no loss values were recorded, so training equivalence is unchecked"
    elif v == "faster":
        decision = "reject"
        reason = f"candidate is faster ({point:.2f}x) but the loss trajectory is outside the {loss_tol:.0%} tolerance (mean relative difference {loss['mean_rel_diff']:.1%})"
    elif v == "slower":
        decision = "reject"
        reason = f"candidate is slower ({point:.2f}x, 95% interval {lo:.2f} to {hi:.2f})"
    else:
        decision = "inconclusive"
        reason = f"no clear difference: the 95% interval {lo:.2f} to {hi:.2f} does not clear the {min_effect:.0%} threshold on either side"
    result["decision"] = decision
    result["reason"] = reason
    spread = max(result["throughput"]["baseline"].get("cv", 0), result["throughput"]["candidate"].get("cv", 0))
    if spread > 0.1:
        result["limits"].append(
            f"Run-to-run spread is high (coefficient of variation up to {spread:.0%}); close other apps or raise --repeats."
        )
    ttfs = {arm: [r["evidence"]["run"].get("time_to_first_step_s") for r in ok[arm]] for arm in arms}
    if all(all(x is not None for x in v) and v for v in ttfs.values()):
        from train_doctor.stats import median

        result["time_to_first_step_s"] = {arm: median(v) for arm, v in ttfs.items()}
        extra = result["time_to_first_step_s"]["candidate"] - result["time_to_first_step_s"]["baseline"]
        if extra > 1.0:
            result["limits"].append(
                f"The candidate takes {extra:.1f} s longer to reach its first step (median). That one-time cost is outside the throughput window; weigh it against the run length."
            )
    if machine.get("gpu_backend") == "apple":
        result["limits"].append(
            "Apple laptops change clock speeds with temperature and power state; interleaving reduces but doesn't remove that drift."
        )
    _write_compare(root, result)
    return result


def _write_compare(root: Path, result: dict) -> None:
    dump_json(root / "compare.json", result)
    report.write(root)


# ---------------------------------------------------------------------- report
def make_report(run_dir: str | Path | None = None, out_dir: str | Path = DEFAULT_OUT) -> dict:
    d = resolve_run(run_dir, out_dir)
    paths = report.write(d)
    return {"run_dir": str(d), **{k: str(v) for k, v in paths.items()}}
