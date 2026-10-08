"""Rule tests on hand-written evidence. No torch, no processes."""

import copy

from train_doctor.rules import diagnose


def base_evidence(**over):
    ev = {
        "source": "hook",
        "machine": {"cores_physical": 8, "cores_logical": 8},
        "config": {
            "device": "cuda",
            "torch": {
                "version": "2.4.0",
                "cuda_capability": [8, 0],
                "cudnn_benchmark": True,
                "compile_available": True,
                "cuda_available": True,
            },
            "dataloaders": [{"dataset": "DS", "num_workers": 4, "pin_memory": True, "batch_size": 128}],
            "autocast": [["cuda", "bfloat16", True]],
            "compile_calls": [{"mode": None}],
            "param_dtypes": ["float32"],
            "conv_modules": 0,
            "ddp": False,
            "ddp_no_sync_calls": 0,
        },
        "split": {"data_wait": 0.02, "host_sync": 0.0, "checkpoint_save": 0.0, "log_write": 0.0, "other": 0.98},
        "per_step": {
            "samples": 128,
            "data_wait_ms": 0.5,
            "sync_calls": 0,
            "sync_ms": 0,
            "backward_calls": 1,
            "save_calls": 0,
            "log_lines": 0,
        },
        "data": {"getitem_share_of_data_wait": None, "getitem_ms_per_call": None, "getitem_measured": False},
        "step_time_ms": {"median": 40.0},
        "resources": {"gpu": {"util_pct_mean": 95.0}},
        "device_memory": {"used_fraction": 0.7},
    }
    for k, v in over.items():
        ev[k] = v
    return ev


def ids(ev):
    return [f.id for f in diagnose(ev)]


def test_clean_cuda_run_has_no_findings():
    assert ids(base_evidence()) == []


def test_non_hook_evidence_gives_no_findings():
    assert diagnose({"source": "log_lines"}) == []


def test_dataloader_bound_and_preprocessing():
    ev = base_evidence()
    ev["config"]["dataloaders"][0].update(num_workers=0)
    ev["split"].update(data_wait=0.6, other=0.4)
    ev["per_step"]["data_wait_ms"] = 24.0
    ev["data"] = {"getitem_share_of_data_wait": 0.9, "getitem_ms_per_call": 0.18, "getitem_measured": True}
    found = diagnose(ev)
    got = [f.id for f in found]
    assert got[:2] == ["dataloader_bound", "cpu_bound_preprocessing"]
    dl = found[0]
    assert dl.numbers["suggested_workers"] == 4
    assert "pin_memory=True" in dl.suggestion  # cuda
    assert "2.50x" in dl.expected_effect  # 1 / (1 - 0.6)
    assert dl.risk == "low"


def test_workers_but_still_waiting_is_preprocessing_not_loader():
    ev = base_evidence()
    ev["split"].update(data_wait=0.4, other=0.6)
    ev["resources"]["proc_cpu_pct_mean"] = 750.0
    ev["resources"]["cores_logical"] = 8
    got = ids(ev)
    assert "cpu_bound_preprocessing" in got and "dataloader_bound" not in got


def test_cuda_config_findings():
    ev = base_evidence()
    ev["config"]["autocast"] = []
    ev["config"]["compile_calls"] = []
    ev["config"]["dataloaders"][0]["pin_memory"] = False
    ev["config"]["conv_modules"] = 5
    ev["config"]["torch"]["cudnn_benchmark"] = False
    found = {f.id: f for f in diagnose(ev)}
    assert {"no_mixed_precision", "no_torch_compile", "pin_memory_off", "cudnn_benchmark_off"} <= set(found)
    assert "bfloat16" in found["no_mixed_precision"].suggestion
    assert found["no_mixed_precision"].risk == "medium"


def test_fp16_on_volta_mentions_grad_scaler():
    ev = base_evidence()
    ev["config"]["autocast"] = []
    ev["config"]["torch"]["cuda_capability"] = [7, 0]
    f = next(f for f in diagnose(ev) if f.id == "no_mixed_precision")
    assert "float16" in f.suggestion and "GradScaler" in f.suggestion


def test_no_amp_finding_on_old_gpu_or_cpu():
    ev = base_evidence()
    ev["config"]["autocast"] = []
    ev["config"]["torch"]["cuda_capability"] = [6, 1]
    assert "no_mixed_precision" not in ids(ev)
    ev2 = base_evidence()
    ev2["config"]["autocast"] = []
    ev2["config"]["device"] = "cpu"
    ev2["config"]["torch"]["cuda_available"] = False
    assert "no_mixed_precision" not in ids(ev2)


def test_disabled_autocast_does_not_count_as_used():
    ev = base_evidence()
    ev["config"]["autocast"] = [["cuda", "float16", False]]
    assert "no_mixed_precision" in ids(ev)


def test_host_sync_only_on_accelerators():
    ev = base_evidence()
    ev["split"].update(host_sync=0.3, other=0.68)
    ev["per_step"].update(sync_calls=2, sync_ms=12.0)
    f = next(f for f in diagnose(ev) if f.id == "host_sync_in_loop")
    assert f.risk == "none" and "2.0 blocking reads" in f.evidence
    cpu = copy.deepcopy(ev)
    cpu["config"]["device"] = "cpu"
    assert "host_sync_in_loop" not in ids(cpu)


def test_small_batch_flags_dynamics_risk():
    ev = base_evidence()
    ev["config"]["dataloaders"][0]["batch_size"] = 16
    ev["resources"]["gpu"]["util_pct_mean"] = 20.0
    ev["device_memory"]["used_fraction"] = 0.1
    f = next(f for f in diagnose(ev) if f.id == "small_batch")
    assert f.risk == "high" and "changes-dynamics" in f.tags
    assert "32" in f.suggestion


def test_small_batch_silent_when_memory_is_full():
    ev = base_evidence()
    ev["config"]["dataloaders"][0]["batch_size"] = 16
    ev["resources"]["gpu"]["util_pct_mean"] = 20.0
    ev["device_memory"]["used_fraction"] = 0.9
    assert "small_batch" not in ids(ev)


def test_io_findings():
    ev = base_evidence()
    ev["split"].update(checkpoint_save=0.3, log_write=0.03, other=0.65)
    ev["per_step"].update(save_calls=1.0, log_lines=3.0)
    got = ids(ev)
    assert "checkpoint_in_hot_loop" in got and "logging_in_hot_loop" in got


def test_ddp_accumulation_without_no_sync():
    ev = base_evidence()
    ev["config"]["ddp"] = True
    ev["per_step"]["backward_calls"] = 4.0
    f = next(f for f in diagnose(ev) if f.id == "ddp_accumulation_without_no_sync")
    assert "first 3 micro-steps" in f.suggestion


def test_accumulation_with_headroom():
    ev = base_evidence()
    ev["per_step"]["backward_calls"] = 4.0
    ev["device_memory"]["used_fraction"] = 0.2
    assert "grad_accumulation_small_micro_batch" in ids(ev)


def test_cpu_while_mps_available():
    ev = base_evidence()
    ev["config"]["device"] = "cpu"
    ev["config"]["torch"].update(cuda_available=False, mps_available=True)
    f = next(f for f in diagnose(ev) if f.id == "cpu_with_accelerator_available")
    assert "mps" in f.suggestion


def test_ranking_is_by_score_and_deterministic():
    ev = base_evidence()
    ev["config"]["dataloaders"][0].update(num_workers=0)
    ev["split"].update(data_wait=0.2, checkpoint_save=0.5, other=0.3)
    ev["per_step"]["save_calls"] = 1
    a = [f.id for f in diagnose(ev)]
    b = [f.id for f in diagnose(copy.deepcopy(ev))]
    assert a == b
    assert a[0] == "checkpoint_in_hot_loop"
    scores = [f.score for f in diagnose(ev)]
    assert scores == sorted(scores, reverse=True)
