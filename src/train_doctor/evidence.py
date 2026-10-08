"""Turn the raw files of one run into a single evidence summary.

Inputs (all inside the run directory, all optional except run.json):

* ``events.<pid>.jsonl``: per-step records from the in-process hook
* ``log_losses.jsonl``: loss values parsed from stdout, with timestamps
* ``samples.jsonl``: CPU, memory and GPU samples from the parent
* ``torch_ops.json``: torch.profiler op summary
* ``pyspy.txt``: py-spy raw (collapsed) stacks

Every derived number names the window it was computed over.
"""

from __future__ import annotations

import json
from pathlib import Path

from train_doctor.redact import dump_json
from train_doctor.stats import describe, mean, quantile

SYNC_OPS = {
    "aten::_local_scalar_dense": "scalar read (.item(), float(), bool())",
    "aten::item": "scalar read (.item())",
    "cudaStreamSynchronize": "CUDA stream synchronize",
    "cudaDeviceSynchronize": "CUDA device synchronize",
    "cudaMemcpy": "blocking CUDA memcpy",
}


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a run killed mid-write can leave a partial last line
    return out


def load_events(run_dir: Path) -> tuple[list[dict], int | None]:
    """Events of the process that recorded the most steps (rank 0 or the only trainer)."""
    best: list[dict] = []
    best_pid = None
    best_steps = -1
    for p in sorted(run_dir.glob("events.*.jsonl")):
        evs = _read_jsonl(p)
        n = sum(1 for e in evs if e.get("type") == "step")
        if n > best_steps:
            best, best_steps = evs, n
            try:
                best_pid = int(p.name.split(".")[1])
            except (IndexError, ValueError):
                best_pid = None
    return best, best_pid


def _share(part: float, total: float) -> float:
    return part / total if total > 0 else 0.0


def hook_summary(events: list[dict]) -> dict | None:
    steps = [e for e in events if e.get("type") == "step"]
    if not steps:
        return None
    metas = [e for e in events if e.get("type") == "meta"]
    meta = metas[-1] if metas else {}
    wstart = next((e for e in events if e.get("type") == "window_start"), None)
    wend = next((e for e in events if e.get("type") == "window_end"), None)
    end = next((e for e in events if e.get("type") == "end"), None)
    losses = next((e for e in events if e.get("type") == "losses"), {"values": []})
    measured = [s for s in steps if not s.get("warmup") and not s.get("profiled")]
    profiled = [s for s in steps if s.get("profiled")]

    out: dict = {"source": "hook", "pid": meta.get("pid")}
    out["steps"] = {
        "total": len(steps),
        "warmup": sum(1 for s in steps if s.get("warmup")),
        "measured": len(measured),
        "profiled": len(profiled),
        "boundary": steps[0].get("source"),
        "end_reason": end.get("reason") if end else "no end record (process killed?)",
    }
    dts = [s["dt"] for s in measured]
    out["step_time_ms"] = {k: (v * 1000 if isinstance(v, float) and k not in ("n", "cv") else v) for k, v in describe(dts).items()}

    # throughput over the synchronized window bracket when available
    samples_known = all(s.get("samples") is not None for s in measured) and bool(measured)
    total_samples = sum(s.get("samples") or 0 for s in measured)
    if wstart and wend:
        dur = wend["t"] - wstart["t"]
        basis = "synchronized window: device synced at window start and end"
    else:
        dur = sum(dts)
        basis = "sum of measured step times (window not closed)"
    out["window"] = {
        "duration_s": dur,
        "steps": len(measured),
        "samples": total_samples if samples_known else None,
        "basis": basis,
        "wall_start": wstart.get("wall") if wstart else None,
        "wall_end": wend.get("wall") if wend else None,
    }
    tp: dict = {"steps_per_s": len(measured) / dur if dur > 0 else None}
    if samples_known and dur > 0:
        tp["samples_per_s"] = total_samples / dur
        tp["metric"] = "samples_per_s"
    else:
        tp["metric"] = "steps_per_s"
    out["throughput"] = tp

    total_dt = sum(dts)
    data = sum(s.get("data_wait", 0.0) for s in measured)
    sync = sum(s.get("sync_time", 0.0) for s in measured)
    save = sum(s.get("save_time", 0.0) for s in measured)
    io = sum(s.get("io_time", 0.0) for s in measured)
    other = max(0.0, total_dt - data - sync - save - io)
    out["split"] = {
        "data_wait": _share(data, total_dt),
        "host_sync": _share(sync, total_dt),
        "checkpoint_save": _share(save, total_dt),
        "log_write": _share(io, total_dt),
        "other": _share(other, total_dt),
        "basis": "share of summed step time over measured steps; 'other' is forward, backward, optimizer and Python overhead",
    }
    n = max(1, len(measured))
    getitem_t = sum(s.get("getitem_time", 0.0) for s in measured)
    getitem_n = sum(s.get("getitem_calls", 0) for s in measured)
    out["per_step"] = {
        "samples": total_samples / n if samples_known else None,
        "data_wait_ms": data / n * 1000,
        "sync_calls": sum(s.get("sync_calls", 0) for s in measured) / n,
        "sync_ms": sync / n * 1000,
        "cpu_scalar_reads": sum(s.get("cpu_scalar_calls", 0) for s in measured) / n,
        "backward_calls": sum(s.get("backward_calls", 0) for s in measured) / n,
        "save_calls": sum(s.get("save_calls", 0) for s in measured) / n,
        "log_lines": sum(s.get("io_lines", 0) for s in measured) / n,
        "getitem_calls": getitem_n / n,
    }
    out["data"] = {
        "getitem_share_of_data_wait": _share(getitem_t, data) if getitem_n else None,
        "getitem_ms_per_call": getitem_t / getitem_n * 1000 if getitem_n else None,
        "getitem_measured": bool(getitem_n),
    }
    cfg = {
        k: meta.get(k)
        for k in (
            "device",
            "torch",
            "first_step_wall",
            "dataloaders",
            "autocast",
            "grad_scaler",
            "compile_calls",
            "ddp",
            "ddp_no_sync_calls",
            "conv_modules",
            "params",
            "param_dtypes",
            "optimizer",
            "python",
            "executable",
            "startup_to_first_step_s",
        )
    }
    out["config"] = cfg
    mem = [s["mem"] for s in steps if s.get("mem")]
    if wend and wend.get("mem"):
        mem.append(wend["mem"])
    out["device_memory"] = _mem_summary(mem)
    out["loss"] = [[int(v[0]), float(v[1])] for v in losses.get("values", [])]
    out["samples_by_step"] = [s.get("samples") for s in steps]
    out["profiler_errors"] = [e.get("error") for e in events if e.get("type") == "profiler_error"]
    return out


def _mem_summary(mems: list[dict]) -> dict:
    if not mems:
        return {}
    out: dict = {}
    for key in ("allocated", "peak_allocated", "driver_allocated", "reserved"):
        vals = [m[key] for m in mems if isinstance(m.get(key), int)]
        if vals:
            out[key + "_max"] = max(vals)
    totals = [m["total"] for m in mems if isinstance(m.get("total"), int)]
    if totals:
        out["total"] = totals[-1]
    used = out.get("peak_allocated_max") or out.get("driver_allocated_max") or out.get("allocated_max")
    if used and out.get("total"):
        out["used_fraction"] = used / out["total"]
    return out


def log_summary(run_dir: Path, warmup: int) -> dict | None:
    """Fallback when the hook saw no steps: use loss lines printed to stdout."""
    lines = _read_jsonl(run_dir / "log_losses.jsonl")
    if len(lines) < 3:
        return None
    walls = [ln["wall"] for ln in lines]
    skip = min(warmup, len(walls) - 2)
    dts = [b - a for a, b in zip(walls[skip:], walls[skip + 1 :], strict=False)]
    dur = walls[-1] - walls[skip]
    out: dict = {
        "source": "log_lines",
        "steps": {"total": len(lines), "warmup": skip, "measured": len(dts), "profiled": 0, "boundary": "stdout loss line"},
        "step_time_ms": {k: (v * 1000 if isinstance(v, float) and k not in ("n", "cv") else v) for k, v in describe(dts).items()},
        "window": {
            "duration_s": dur,
            "steps": len(dts),
            "samples": None,
            "basis": "time between printed loss lines",
            "wall_start": walls[skip],
            "wall_end": walls[-1],
        },
        "throughput": {
            "steps_per_s": len(dts) / dur if dur > 0 else None,
            "metric": "steps_per_s",
            "note": "a 'step' here is one printed loss line",
        },
        "loss": [[i, ln["loss"]] for i, ln in enumerate(lines)],
        "config": {},
    }
    return out


def resource_summary(run_dir: Path, wall_start: float | None, wall_end: float | None, cores: int | None) -> dict:
    samples = _read_jsonl(run_dir / "samples.jsonl")
    if wall_start is not None and wall_end is not None:
        inwin = [s for s in samples if wall_start <= s["wall"] <= wall_end]
        scope = "measured window"
    else:
        inwin = samples
        scope = "whole run"
    if len(inwin) < 3:
        inwin = samples
        scope = "whole run (the measured window held fewer than 3 samples)"
    if not inwin:
        return {}
    cpu = [s["proc_cpu_pct"] for s in inwin]
    out: dict = {
        "scope": scope,
        "n_samples": len(inwin),
        "proc_cpu_pct_mean": mean(cpu),
        "proc_cpu_pct_p90": quantile(cpu, 0.9),
        "cores_logical": cores,
        "sys_cpu_pct_mean": mean([s.get("sys_cpu_pct", 0.0) for s in inwin]),
        "processes_max": max(s.get("procs", 0) for s in inwin),
        "tree_rss_peak": max(s.get("tree_rss", 0) for s in samples) if samples else None,
        "max_proc_rss_peak": max(s.get("max_proc_rss", 0) for s in samples) if samples else None,
    }
    gpu = [s["gpu"] for s in inwin if s.get("gpu")]
    if gpu:
        backend = gpu[0]["backend"]
        utils = []
        mem_used = []
        mem_total = None
        for g in gpu:
            gs = g.get("gpus") or []
            if not gs:
                continue
            first = gs[0]
            if first.get("util_pct") is not None:
                utils.append(first["util_pct"])
            if first.get("mem_used_mib") is not None:
                mem_used.append(first["mem_used_mib"])
                mem_total = first.get("mem_total_mib")
        out["gpu"] = {
            "backend": backend,
            "name": (gpu[0].get("gpus") or [{}])[0].get("name"),
            "n_samples": len(utils),
            "util_pct_mean": mean(utils) if utils else None,
            "util_pct_max": max(utils) if utils else None,
            "mem_used_mib_max": max(mem_used) if mem_used else None,
            "mem_total_mib": mem_total,
            "util_scope": "system-wide (ioreg Device Utilization %)" if backend == "apple" else "GPU 0 (nvidia-smi utilization.gpu)",
        }
    return out


def profiler_summary(run_dir: Path) -> dict | None:
    p = run_dir / "torch_ops.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    steps = max(1, int(data.get("profiled_steps", 1)))
    ops = data.get("ops", [])
    sync = {}
    for op in ops:
        for key, label in SYNC_OPS.items():
            if op["name"] == key or op["name"].startswith(key):
                sync[op["name"]] = {
                    "label": label,
                    "per_step": op["count"] / steps,
                    "cpu_total_ms_per_step": op["cpu_total_us"] / steps / 1000,
                }
    top = [
        {
            "name": o["name"],
            "count_per_step": o["count"] / steps,
            "self_cpu_ms_per_step": o["self_cpu_us"] / steps / 1000,
            "self_device_ms_per_step": o["self_device_us"] / steps / 1000,
        }
        for o in ops[:15]
    ]
    return {
        "profiled_steps": steps,
        "sync_ops": sync,
        "top_ops": top,
        "trace": "torch_trace.json.gz" if (run_dir / "torch_trace.json.gz").exists() else None,
    }


def pyspy_summary(run_dir: Path, top: int = 15) -> dict | None:
    p = run_dir / "pyspy.txt"
    if not p.exists():
        log = run_dir / "pyspy.log"
        if log.exists():
            return {"error": log.read_text()[-600:].strip()}
        return None
    return parse_pyspy_raw(p.read_text(errors="replace"), top=top)


def parse_pyspy_raw(text: str, top: int = 15) -> dict:
    """Parse py-spy ``--format raw`` (collapsed stacks: ``f1;f2;f3 COUNT``)."""
    own: dict[str, int] = {}
    incl: dict[str, int] = {}
    total = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or " " not in line:
            continue
        stack, _, count = line.rpartition(" ")
        try:
            c = int(count)
        except ValueError:
            continue
        frames = [f for f in stack.split(";") if f]
        if not frames:
            continue
        total += c
        own[frames[-1]] = own.get(frames[-1], 0) + c
        for f in set(frames):
            incl[f] = incl.get(f, 0) + c
    if total == 0:
        return {"total_samples": 0, "own": [], "inclusive": []}

    def rank(d: dict[str, int]) -> list[dict]:
        return [{"frame": k, "share": v / total, "samples": v} for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:top]]

    return {"total_samples": total, "own": rank(own), "inclusive": rank(incl)}


def summarize(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    events, _ = load_events(run_dir)
    ev = hook_summary(events)
    limits: list[str] = []
    if ev is None:
        ev = log_summary(run_dir, int(run.get("config", {}).get("warmup_steps", 5)))
        if ev is not None:
            limits.append(
                "No optimizer steps or train_doctor.step() markers were seen, so timings come from the times stdout loss lines were printed. Data, sync and I/O splits are unavailable."
            )
    if ev is None:
        ev = {"source": "none", "steps": {"total": 0}, "config": {}, "loss": []}
        limits.append(
            "No steps detected: no optimizer.step() calls, no train_doctor.step() markers and no loss lines on stdout. Only whole-process wall time and resource samples are available."
        )
    machine = run.get("machine", {})
    win = ev.get("window", {})
    ev["resources"] = resource_summary(run_dir, win.get("wall_start"), win.get("wall_end"), machine.get("cores_logical"))
    ev["profiler"] = profiler_summary(run_dir)
    ev["py_spy"] = pyspy_summary(run_dir)
    ev["run"] = {
        k: run.get(k)
        for k in ("cmd", "cwd", "started_at", "wall_s", "exit_code", "timed_out", "env_set", "provenance", "label", "load_avg_1m")
    }
    first = (ev.get("config") or {}).get("first_step_wall")
    if first and run.get("start_wall"):
        ev["run"]["time_to_first_step_s"] = first - run["start_wall"]
    ev["machine"] = machine

    if run.get("timed_out"):
        limits.append(f"The run hit the {run['config'].get('timeout')} s timeout and was stopped.")
    if run.get("exit_code") not in (0, None) and ev.get("steps", {}).get("end_reason") != "window_complete":
        limits.append(f"The command exited with code {run.get('exit_code')} before the window closed; see stderr.log.")
    st = ev.get("steps", {})
    if ev.get("source") == "hook" and st.get("end_reason") not in ("window_complete", None):
        limits.append(
            f"The measurement window did not complete (end: {st.get('end_reason')}) after {st.get('total')} steps; "
            "the numbers cover only the steps that ran. Lower --steps/--seconds or let the script run longer."
        )
    if ev.get("source") == "hook" and st.get("measured", 0) < 10:
        limits.append(f"Only {st.get('measured', 0)} measured steps; step-time percentiles are rough.")
    if ev.get("source") == "hook":
        res_gpu = ev.get("resources", {}).get("gpu")
        dev = (ev.get("config") or {}).get("device")
        if dev == "mps" and res_gpu:
            limits.append("Apple GPU utilization comes from ioreg and is system-wide: other apps using the GPU are included.")
        if dev in ("cuda", "mps") and not res_gpu:
            limits.append("No GPU utilization samples were available.")
        if ev.get("data", {}).get("getitem_measured") is False:
            limits.append("Dataset __getitem__ time is only measured when num_workers=0 (it runs in worker processes otherwise).")
        if ev.get("split", {}).get("host_sync", 0) >= 0.05:
            limits.append(
                "Time inside sync calls includes waiting for queued device work, so it is not all wasted time; removing a sync helps by letting the host queue the next step earlier."
            )
    ev["limits"] = limits
    return ev


def write_evidence(run_dir: Path) -> dict:
    ev = summarize(run_dir)
    dump_json(Path(run_dir) / "evidence.json", ev)
    return ev
