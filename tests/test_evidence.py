"""Evidence summaries from hand-written raw files."""

import json

from train_doctor.evidence import parse_pyspy_raw, summarize


def write_events(d, steps=12, warmup=2, dt=0.1, data=0.04, sync=0.01, samples=32, window=True):
    lines = [
        {
            "type": "meta",
            "pid": 1,
            "device": "cuda",
            "dataloaders": [{"dataset": "DS", "num_workers": 0, "batch_size": samples}],
            "torch": {"version": "2.4"},
        }
    ]
    t = 100.0
    for i in range(steps):
        t += dt
        lines.append(
            {
                "type": "step",
                "i": i,
                "t": t,
                "dt": dt,
                "data_wait": data,
                "samples": samples,
                "sync_calls": 2,
                "sync_time": sync,
                "cpu_scalar_calls": 0,
                "save_calls": 0,
                "save_time": 0.0,
                "io_lines": 1,
                "io_time": 0.001,
                "backward_calls": 1,
                "getitem_calls": samples,
                "getitem_time": data * 0.8,
                "warmup": i < warmup,
                "profiled": False,
            }
        )
        if window and i == warmup - 1:
            lines.append({"type": "window_start", "t": t, "wall": 1000.0})
    if window:
        lines.append({"type": "window_end", "t": t, "wall": 1000.0 + (steps - warmup) * dt, "mem": {"allocated": 100, "total": 1000}})
    lines.append({"type": "losses", "values": [[i, 2.0 - 0.1 * i, 1] for i in range(steps)]})
    lines.append({"type": "end", "reason": "window_complete", "steps": steps})
    (d / "events.1.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    # a DataLoader worker or launcher process that saw nothing
    (d / "events.2.jsonl").write_text(json.dumps({"type": "meta", "pid": 2}) + "\n")


def write_run(d, **extra):
    run = {
        "cmd": ["python", "train.py"],
        "cwd": str(d),
        "exit_code": 0,
        "timed_out": False,
        "config": {"warmup_steps": 2, "timeout": 60},
        "machine": {"cores_logical": 8},
        **extra,
    }
    (d / "run.json").write_text(json.dumps(run))


def test_hook_summary_numbers(tmp_path):
    write_run(tmp_path)
    write_events(tmp_path)
    ev = summarize(tmp_path)
    assert ev["source"] == "hook"
    assert ev["steps"]["measured"] == 10 and ev["steps"]["warmup"] == 2
    assert abs(ev["step_time_ms"]["median"] - 100.0) < 1e-6
    assert abs(ev["throughput"]["samples_per_s"] - 320.0) < 1e-6  # 32 samples / 0.1 s
    assert abs(ev["split"]["data_wait"] - 0.4) < 1e-9
    assert abs(ev["split"]["host_sync"] - 0.1) < 1e-9
    assert abs(ev["data"]["getitem_share_of_data_wait"] - 0.8) < 1e-9
    assert ev["per_step"]["sync_calls"] == 2
    assert ev["device_memory"]["used_fraction"] == 0.1
    assert len(ev["loss"]) == 12 and ev["samples_by_step"][0] == 32
    assert ev["config"]["device"] == "cuda"


def test_truncated_last_line_is_ignored(tmp_path):
    write_run(tmp_path)
    write_events(tmp_path)
    with open(tmp_path / "events.1.jsonl", "a") as f:
        f.write('{"type": "step", "i": 99, "t"')
    assert summarize(tmp_path)["steps"]["total"] == 12


def test_log_line_fallback(tmp_path):
    write_run(tmp_path)
    lines = [{"wall": 1000.0 + 0.5 * i, "loss": 3.0 - i * 0.1} for i in range(10)]
    (tmp_path / "log_losses.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    ev = summarize(tmp_path)
    assert ev["source"] == "log_lines"
    assert abs(ev["throughput"]["steps_per_s"] - 2.0) < 1e-9
    assert any("stdout loss lines" in lim for lim in ev["limits"])


def test_nothing_detected(tmp_path):
    write_run(tmp_path)
    ev = summarize(tmp_path)
    assert ev["source"] == "none"
    assert any("No steps detected" in lim for lim in ev["limits"])


def test_resources_with_gpu_samples_in_window(tmp_path):
    write_run(tmp_path)
    write_events(tmp_path)
    rows = []
    for i in range(10):
        rows.append(
            {
                "wall": 1000.0 + 0.1 * i,
                "proc_cpu_pct": 100.0,
                "procs": 1,
                "tree_rss": 1 << 30,
                "max_proc_rss": 1 << 30,
                "sys_cpu_pct": 20.0,
                "gpu": {
                    "backend": "nvidia",
                    "gpus": [{"name": "Fake GPU", "util_pct": 40.0 + i, "mem_used_mib": 1000.0, "mem_total_mib": 8000.0}],
                },
            }
        )
    rows.append({"wall": 5000.0, "proc_cpu_pct": 0.0, "procs": 1, "tree_rss": 0, "max_proc_rss": 0})  # outside window
    (tmp_path / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    res = summarize(tmp_path)["resources"]
    assert res["scope"] == "measured window"
    assert res["proc_cpu_pct_mean"] == 100.0
    assert res["gpu"]["util_pct_mean"] == 44.5
    assert res["gpu"]["name"] == "Fake GPU"


def test_profiler_summary(tmp_path):
    write_run(tmp_path)
    write_events(tmp_path)
    ops = {
        "profiled_steps": 5,
        "ops": [
            {"name": "aten::_local_scalar_dense", "count": 10, "self_cpu_us": 5000.0, "cpu_total_us": 5000.0, "self_device_us": 0.0},
            {"name": "aten::mm", "count": 15, "self_cpu_us": 100.0, "cpu_total_us": 200.0, "self_device_us": 900.0},
        ],
    }
    (tmp_path / "torch_ops.json").write_text(json.dumps(ops))
    prof = summarize(tmp_path)["profiler"]
    assert prof["sync_ops"]["aten::_local_scalar_dense"]["per_step"] == 2.0
    assert prof["top_ops"][1]["self_device_ms_per_step"] == 0.18


def test_pyspy_parse():
    text = (
        "main (train.py:10);loop (train.py:20);__getitem__ (data.py:5) 70\nmain (train.py:10);loop (train.py:20);forward (model.py:3) 30\n"
    )
    out = parse_pyspy_raw(text)
    assert out["total_samples"] == 100
    assert out["own"][0]["frame"] == "__getitem__ (data.py:5)" and out["own"][0]["share"] == 0.7
    assert out["inclusive"][0]["share"] == 1.0
