"""Render a run directory as report.md (for people) and report.json (for agents)."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from train_doctor import __version__
from train_doctor.redact import dump_json, redact_text


def _fmt(x, digits: int = 1, unit: str = "") -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:,.{digits}f}{unit}"
    return f"{x}{unit}"


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _mib(b) -> str:
    return "n/a" if not b else f"{b / 1048576:,.0f} MiB"


def _autocast(entries) -> str:
    if not entries:
        return "never entered"
    on = [f"{e[0]} {e[1]}" for e in entries if len(e) < 3 or e[2]]
    off = [f"{e[0]} {e[1]}" for e in entries if len(e) >= 3 and not e[2]]
    parts = []
    if on:
        parts.append("enabled (" + ", ".join(on) + ")")
    if off:
        parts.append("entered with enabled=False (" + ", ".join(off) + ")")
    return "; ".join(parts)


def _cmd(argv: list[str]) -> str:
    return shlex.join(argv) if argv else ""


def write(run_dir: str | Path) -> dict:
    d = Path(run_dir)
    if (d / "compare.json").exists():
        data = json.loads((d / "compare.json").read_text())
        md = render_compare(data)
        bundle = {"train_doctor": __version__, **data}
    else:
        ev = json.loads((d / "evidence.json").read_text())
        findings = json.loads((d / "findings.json").read_text()) if (d / "findings.json").exists() else []
        md = render_profile(ev, findings, d)
        bundle = {
            "train_doctor": __version__,
            "kind": "profile",
            "run_dir": str(d),
            "findings": findings,
            "evidence": ev,
            "files": sorted(p.name for p in d.iterdir() if p.is_file()),
        }
    (d / "report.md").write_text(redact_text(md))
    dump_json(d / "report.json", bundle)
    return {"markdown": d / "report.md", "json": d / "report.json"}


# --------------------------------------------------------------------- profile
def render_profile(ev: dict, findings: list[dict], d: Path) -> str:
    run = ev.get("run") or {}
    m = ev.get("machine") or {}
    cfg = ev.get("config") or {}
    t = cfg.get("torch") or {}
    st = ev.get("steps") or {}
    stm = ev.get("step_time_ms") or {}
    tp = ev.get("throughput") or {}
    win = ev.get("window") or {}
    L: list[str] = []
    L.append("# train-doctor profile" + (f": {run['label']}" if run.get("label") else "") + "\n")
    L.append(f"Command: `{_cmd(run.get('cmd') or [])}`  ")
    L.append(
        f"Started {run.get('started_at')}, wall time {_fmt(run.get('wall_s'), 1, ' s')}, exit code {run.get('exit_code')}, 1-minute load average at start {_fmt(run.get('load_avg_1m'), 2)}.  "
    )
    L.append(f"Machine: {m.get('cpu')}, {m.get('cores_logical')} logical cores, {_mib(m.get('ram_bytes'))} RAM, {m.get('os')}.  ")
    if t:
        L.append(f"Python {cfg.get('python')}, torch {t.get('version')}, device `{cfg.get('device')}`.")
    L.append("")
    L.append("## Summary\n")
    L.append("| | |\n|---|---|")
    L.append(
        f"| Steps seen | {st.get('total', 0)} ({st.get('warmup', 0)} warmup, {st.get('measured', 0)} measured, {st.get('profiled', 0)} profiled) |"
    )
    L.append(f"| Step boundary | {st.get('boundary', 'n/a')} |")
    if stm.get("n"):
        L.append(
            f"| Step time, median (p10 to p90) | {_fmt(stm.get('median'), 2)} ms ({_fmt(stm.get('p10'), 2)} to {_fmt(stm.get('p90'), 2)}) |"
        )
        L.append(f"| Step time spread (CV) | {_pct(stm.get('cv'))} |")
    if tp.get("samples_per_s"):
        L.append(f"| Throughput | {_fmt(tp['samples_per_s'], 1)} samples/s ({_fmt(tp.get('steps_per_s'), 2)} steps/s) |")
    elif tp.get("steps_per_s"):
        L.append(f"| Throughput | {_fmt(tp['steps_per_s'], 2)} steps/s |")
    if win:
        L.append(f"| Window | {win.get('steps')} steps in {_fmt(win.get('duration_s'), 3)} s; {win.get('basis')} |")
    L.append("")

    split = ev.get("split")
    if split:
        ps = ev.get("per_step") or {}
        stepms = stm.get("mean") or 0.0
        L.append("## Where step time goes\n")
        L.append("| Part | Share | ms per step |\n|---|---:|---:|")
        for key, label in [
            ("data_wait", "Waiting for the DataLoader"),
            ("host_sync", "Blocked in host/device sync"),
            ("checkpoint_save", "torch.save"),
            ("log_write", "Writing stdout/stderr"),
            ("other", "Other (forward, backward, optimizer, Python)"),
        ]:
            L.append(f"| {label} | {_pct(split.get(key))} | {_fmt((split.get(key) or 0) * stepms, 2)} |")
        L.append(
            f"\n{split.get('basis')}. Per step: {_fmt(ps.get('samples'), 0)} samples, {_fmt(ps.get('sync_calls'), 1)} device syncs, {_fmt(ps.get('backward_calls'), 1)} backward calls, {_fmt(ps.get('log_lines'), 1)} log lines.\n"
        )

    L.append("## Findings\n")
    if not findings:
        L.append("No rule fired." + (" (Rules need step-level evidence from the hook.)" if ev.get("source") != "hook" else "") + "\n")
    for i, f in enumerate(findings, 1):
        L.append(f"### {i}. {f['title']}\n")
        L.append(f"- Evidence: {f['evidence']}")
        L.append(f"- Change: {f['suggestion']}")
        L.append(f"- Expected effect: {f['expected_effect']}")
        L.append(f"- Risk to results ({f['risk']}): {f['risk_note']}")
        L.append(f"- Rank score {f['score']:.3f} ({f['score_basis']}). Rule id `{f['id']}`.\n")

    if cfg.get("dataloaders") is not None or cfg.get("params"):
        L.append("## Configuration seen\n")
        for dl in cfg.get("dataloaders") or []:
            L.append(
                f"- DataLoader over `{dl.get('dataset')}` ({dl.get('len')} items): batch_size={dl.get('batch_size')}, num_workers={dl.get('num_workers')}, pin_memory={dl.get('pin_memory')}, persistent_workers={dl.get('persistent_workers')}"
            )
        if cfg.get("params"):
            L.append(
                f"- Optimizer {cfg.get('optimizer')} over {cfg.get('params'):,} parameters ({', '.join(cfg.get('param_dtypes') or [])})"
            )
        L.append(
            f"- autocast: {_autocast(cfg.get('autocast'))}; GradScaler: {bool(cfg.get('grad_scaler'))}; torch.compile calls: {len(cfg.get('compile_calls') or [])}; DDP: {bool(cfg.get('ddp'))}; conv modules: {cfg.get('conv_modules')}"
        )
        if t.get("cuda_available"):
            L.append(
                f"- CUDA device {t.get('cuda_device')} (capability {t.get('cuda_capability')}), cudnn.benchmark={t.get('cudnn_benchmark')}"
            )
        L.append(f"- torch threads: {t.get('num_threads')}; process start to first step: {_fmt(run.get('time_to_first_step_s'), 2, ' s')}")
        L.append("")

    res = ev.get("resources") or {}
    if res:
        L.append("## Resources\n")
        L.append(
            f"- Process tree CPU ({res.get('scope')}): mean {_fmt(res.get('proc_cpu_pct_mean'), 0)}%, p90 {_fmt(res.get('proc_cpu_pct_p90'), 0)}% (100% = one core; {res.get('cores_logical')} logical cores), up to {res.get('processes_max')} processes"
        )
        L.append(
            f"- Peak RSS: largest process {_mib(res.get('max_proc_rss_peak'))}, whole tree {_mib(res.get('tree_rss_peak'))} (shared pages counted per process)"
        )
        g = res.get("gpu")
        if g:
            L.append(
                f"- GPU ({g.get('name') or g.get('backend')}): utilization mean {_fmt(g.get('util_pct_mean'), 0)}%, max {_fmt(g.get('util_pct_max'), 0)}% over {g.get('n_samples')} samples, {g.get('util_scope')}"
            )
        dm = ev.get("device_memory") or {}
        if dm:
            used = dm.get("peak_allocated_max") or dm.get("driver_allocated_max") or dm.get("allocated_max")
            L.append(f"- Device memory: {_mib(used)} used of {_mib(dm.get('total'))}")
        L.append("")

    prof = ev.get("profiler")
    if prof:
        L.append(f"## torch.profiler ({prof['profiled_steps']} steps after the measured window)\n")
        if prof.get("sync_ops"):
            L.append(
                "Scalar reads and syncs (the profiler counts CPU and device tensors alike): "
                + ", ".join(f"`{k}` {v['per_step']:.1f}/step" for k, v in prof["sync_ops"].items())
                + "\n"
            )
        ops = prof.get("top_ops", [])[:10]
        dev = any(o["self_device_ms_per_step"] for o in ops)
        L.append(
            "| Op | Calls/step | Self CPU ms/step |"
            + (" Self device ms/step |" if dev else "")
            + "\n|---|---:|---:|"
            + ("---:|" if dev else "")
        )
        for o in ops:
            L.append(
                f"| `{o['name']}` | {o['count_per_step']:.1f} | {o['self_cpu_ms_per_step']:.3f} |"
                + (f" {o['self_device_ms_per_step']:.3f} |" if dev else "")
            )
        if prof.get("trace"):
            L.append(f"\nChrome trace: `{prof['trace']}` (open in https://ui.perfetto.dev).")
        L.append("")
    spy = ev.get("py_spy")
    if spy:
        L.append("## py-spy\n")
        if spy.get("error"):
            L.append(f"py-spy did not produce a profile: `{spy['error'][-300:]}`\n")
        else:
            L.append(f"{spy['total_samples']} samples. Top frames by own time:\n")
            for fr in spy.get("own", [])[:10]:
                L.append(f"- {fr['share'] * 100:.1f}% `{fr['frame']}`")
            L.append("")

    loss = ev.get("loss") or []
    if loss:
        L.append("## Loss\n")
        L.append(f"{len(loss)} values recorded; first {loss[0][1]:.4f} (step {loss[0][0]}), last {loss[-1][1]:.4f} (step {loss[-1][0]}).\n")

    L.append("## Limits\n")
    for lim in ev.get("limits") or ["None recorded."]:
        L.append(f"- {lim}")
    L.append("")
    L.append("## Files\n")
    L.append(
        f"Run directory `{d}`: " + ", ".join(f"`{p.name}`" for p in sorted(d.iterdir()) if p.is_file() and p.name not in ("report.md",))
    )
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------- compare
def render_compare(r: dict) -> str:
    L: list[str] = []
    s = r.get("settings") or {}
    m = r.get("machine") or {}
    unit = "samples/s" if r.get("metric") == "samples_per_s" else "steps/s"
    L.append("# train-doctor compare" + (f": {r['label']}" if r.get("label") else "") + "\n")
    L.append(f"Decision: **{r.get('decision')}**. {str(r.get('reason', '')).capitalize()}.\n")
    tpb = (r.get("throughput") or {}).get("baseline") or {}
    tpc = (r.get("throughput") or {}).get("candidate") or {}
    L.append("| | Baseline | Candidate |\n|---|---|---|")
    L.append(f"| Command | `{_cmd(r['commands']['baseline'])}` | `{_cmd(r['commands']['candidate'])}` |")
    if tpb.get("n"):
        L.append(f"| Median {unit} | {_fmt(tpb.get('median'), 2)} | {_fmt(tpc.get('median'), 2)} |")
        L.append(
            f"| Min to max | {_fmt(tpb.get('min'), 2)} to {_fmt(tpb.get('max'), 2)} | {_fmt(tpc.get('min'), 2)} to {_fmt(tpc.get('max'), 2)} |"
        )
        L.append(f"| Spread (CV) | {_pct(tpb.get('cv'))} | {_pct(tpc.get('cv'))} |")
        L.append(f"| Repeats | {tpb.get('n')} | {tpc.get('n')} |")
        tt = r.get("time_to_first_step_s")
        if tt:
            L.append(f"| Process start to first step (median) | {tt['baseline']:.2f} s | {tt['candidate']:.2f} s |")
    L.append("")
    if r.get("ratio"):
        ra = r["ratio"]
        mw = r.get("mann_whitney") or {}
        L.append(
            f"Speed ratio (candidate / baseline, median {unit}): **{ra['point']:.3f}x**, 95% interval {ra['low']:.3f} to {ra['high']:.3f}. Verdict: **{r.get('verdict')}** (threshold: interval must clear 1 +/- {s.get('min_effect', 0):.0%}). Mann-Whitney U = {mw.get('u')}, two-sided p = {mw.get('p_two_sided', float('nan')):.4f}.\n"
        )
    lc = r.get("loss_check")
    if lc:
        L.append("## Loss check\n")
        if lc.get("status") == "unavailable":
            L.append(f"Unavailable: {lc.get('reason')}.\n")
        else:
            L.append(
                f"Status: **{lc['status']}**. Tolerance {lc['tol']:.0%} on both numbers below; curves smoothed over {lc.get('smoothing')} points "
                f"and aligned on {lc.get('axis')} ({lc.get('points')} common points).\n"
            )
            L.append(
                f"- Curve-level mean relative difference: {_pct(lc.get('mean_rel_diff'))} (largest single point {_pct(lc.get('max_rel_diff'))})."
            )
            L.append(
                f"- Last {lc.get('final_points', 10)} points: baseline {lc.get('baseline_final', 0):.4f}, candidate {lc.get('candidate_final', 0):.4f}, "
                f"relative difference {_pct(lc.get('final_rel_diff'))}."
            )
            sd = lc.get("baseline_self_diff")
            if sd:
                L.append(
                    f"- For reference, baseline repeat 0 vs repeat 1: {sd.get('status')}, mean relative difference {_pct(sd.get('mean_rel_diff'))}."
                )
            L.append("")
    L.append("## Per-repeat numbers (in run order)\n")
    L.append(
        f"| # | Phase | Arm | Repeat | {unit} | Exit | System CPU % | Load 1m | Directory |\n|---:|---|---|---:|---:|---:|---:|---:|---|"
    )
    for run in r.get("runs", []):
        L.append(
            f"| {run['order'] + 1} | {run['phase']} | {run['arm']} | {run['index']} | {_fmt(run.get('value'), 2)} | {run.get('exit_code')} "
            f"| {_fmt(run.get('sys_cpu_pct_mean'), 0)} | {_fmt(run.get('load_avg_1m'), 1)} | `{run['dir']}` |"
        )
    L.append(
        "\nSystem CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. "
        "Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.\n"
    )
    if r.get("errors"):
        L.append("## Errors\n")
        for e in r["errors"]:
            L.append(f"- {e}")
        L.append("")
    L.append("## Settings\n")
    L.append(
        f"{s.get('repeats')} repeats per arm in randomized pair order (order seed {s.get('order_seed')}), {s.get('warmup_runs')} discarded warmup run(s) per arm, {s.get('warmup_steps')} warmup steps and {s.get('steps') if not s.get('seconds') else str(s.get('seconds')) + ' s'} measured {'steps' if not s.get('seconds') else ''} per run, seed {s.get('seed')} (set before the script's own seeding). Interval: {s.get('interval')}.\n"
    )
    L.append("## Machine\n")
    L.append(
        f"{m.get('cpu')}, {m.get('cores_logical')} logical cores, {_mib(m.get('ram_bytes'))} RAM, {m.get('os')}, GPU telemetry: {m.get('gpu_backend')}. train-doctor {m.get('train_doctor')}.\n"
    )
    L.append("## Limits\n")
    for lim in r.get("limits") or ["None recorded."]:
        L.append(f"- {lim}")
    L.append("")
    return "\n".join(L)
