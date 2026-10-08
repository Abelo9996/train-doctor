"""Deterministic rules that turn evidence into ranked findings.

Each rule reads the evidence dict from ``evidence.summarize`` and either
returns nothing or one ``Finding`` with the numbers it fired on. There is no
model call and no randomness: the same evidence always gives the same list.

Ranking: ``score`` is the measured share of step time a finding addresses
when there is one (for example the data-wait share). Configuration findings
without a measured share get a fixed prior, stated in ``score_basis``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Thresholds, in one place so they're easy to audit.
DATA_WAIT_SHARE = 0.15  # data wait share of step time that counts as loader-bound
GETITEM_DOMINANT = 0.5  # __getitem__ share of data wait that points at preprocessing
SYNC_SHARE = 0.05  # host-sync share of step time worth flagging
SYNC_CALLS = 3.0  # or this many device syncs per step
IO_SHARE = 0.02  # checkpoint or log-write share of step time worth flagging
SMALL_BATCH = 32  # batch sizes at or below this are "small" on an accelerator
LOW_UTIL = 50.0  # GPU utilization (%) below this is "low"
FAST_STEP_MS = 15.0  # steps faster than this are launch-overhead territory
LOW_MEMORY = 0.5  # fraction of device memory in use below which there's headroom


@dataclass
class Finding:
    id: str
    title: str
    score: float
    score_basis: str
    evidence: str
    numbers: dict
    suggestion: str
    expected_effect: str
    risk: str  # none | low | medium | high
    risk_note: str
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _dl(ev: dict) -> dict | None:
    dls = (ev.get("config") or {}).get("dataloaders") or []
    return dls[0] if dls else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _upper_bound(share: float) -> float:
    return 1.0 / (1.0 - share) if share < 0.99 else float("inf")


def _cores(ev: dict) -> int:
    m = ev.get("machine") or {}
    return int(m.get("cores_physical") or m.get("cores_logical") or 4)


def _device(ev: dict) -> str | None:
    return (ev.get("config") or {}).get("device")


def _torch(ev: dict) -> dict:
    return (ev.get("config") or {}).get("torch") or {}


def _gpu_util(ev: dict) -> float | None:
    g = (ev.get("resources") or {}).get("gpu") or {}
    return g.get("util_pct_mean")


def _mem_fraction(ev: dict) -> float | None:
    dm = ev.get("device_memory") or {}
    if dm.get("used_fraction") is not None:
        return dm["used_fraction"]
    g = (ev.get("resources") or {}).get("gpu") or {}
    if g.get("mem_used_mib_max") and g.get("mem_total_mib"):
        return g["mem_used_mib_max"] / g["mem_total_mib"]
    return None


# --------------------------------------------------------------------- rules
def rule_dataloader_bound(ev: dict) -> Finding | None:
    split = ev.get("split") or {}
    dl = _dl(ev)
    share = split.get("data_wait", 0.0)
    if dl is None or share < DATA_WAIT_SHARE or dl.get("num_workers", 0) != 0:
        return None
    k = min(8, max(2, _cores(ev) // 2))
    dev = _device(ev)
    ps = ev.get("per_step") or {}
    sugg = f"Set num_workers={k} and persistent_workers=True on the DataLoader"
    if dev == "cuda":
        sugg += ", plus pin_memory=True and .to(device, non_blocking=True)"
    sugg += '. On macOS and Windows workers start with spawn: keep the training code under `if __name__ == "__main__":` and make sure the dataset pickles.'
    return Finding(
        id="dataloader_bound",
        title="Training waits on the DataLoader (num_workers=0)",
        score=share,
        score_basis="measured data-wait share of step time",
        evidence=(
            f"{_pct(share)} of step time is spent waiting for the next batch "
            f"({ps.get('data_wait_ms', 0):.1f} ms per step) and the DataLoader runs with num_workers=0, "
            "so loading and the training step take turns on one core."
        ),
        numbers={
            "data_wait_share": share,
            "data_wait_ms_per_step": ps.get("data_wait_ms"),
            "num_workers": 0,
            "pin_memory": dl.get("pin_memory"),
            "suggested_workers": k,
        },
        suggestion=sugg,
        expected_effect=f"At most {_upper_bound(share):.2f}x (the step time without any data wait), reached only if loading fully overlaps compute.",
        risk="low",
        risk_note="Deterministic datasets give identical batches. Random augmentation in workers draws from per-worker RNG streams, so exact samples differ (same distribution). Each worker holds a copy of the dataset in memory.",
        tags=["data"],
    )


def rule_cpu_preprocessing(ev: dict) -> Finding | None:
    split = ev.get("split") or {}
    data = ev.get("data") or {}
    dl = _dl(ev)
    share = split.get("data_wait", 0.0)
    if dl is None or share < DATA_WAIT_SHARE:
        return None
    gshare = data.get("getitem_share_of_data_wait")
    workers = dl.get("num_workers", 0)
    if workers == 0 and (gshare is None or gshare < GETITEM_DOMINANT):
        return None
    res = ev.get("resources") or {}
    cpu = res.get("proc_cpu_pct_mean")
    if workers > 0:
        ev_text = f"Even with {workers} workers the step waits for data {_pct(share)} of the time"
        if cpu is not None:
            ev_text += f"; the process tree used {cpu:.0f}% CPU (100% = one core) on {res.get('cores_logical')} logical cores"
        ev_text += "."
    else:
        ev_text = (
            f"{_pct(share)} of step time is data wait and {_pct(gshare)} of that is inside "
            f"{dl.get('dataset')}.__getitem__ ({data.get('getitem_ms_per_call', 0):.2f} ms per sample)."
        )
    return Finding(
        id="cpu_bound_preprocessing",
        title=f"Per-sample preprocessing in {dl.get('dataset')} is expensive",
        score=share * (gshare if gshare is not None else 1.0) * 0.9,
        score_basis="data-wait share times the __getitem__ share of it, times 0.9 so the one-line worker fix ranks first when both apply",
        evidence=ev_text,
        numbers={
            "data_wait_share": share,
            "getitem_share_of_data_wait": gshare,
            "getitem_ms_per_sample": data.get("getitem_ms_per_call"),
            "num_workers": workers,
            "proc_cpu_pct_mean": cpu,
        },
        suggestion=(
            "Make preprocessing cheaper instead of (or as well as) parallelizing it: precompute or cache deterministic "
            "transforms once, move per-element Python loops to vectorized tensor ops on the whole batch (or on the device), "
            "and keep only cheap random augmentation per sample."
        ),
        expected_effect=f"At most {_upper_bound(share):.2f}x if data wait disappears entirely.",
        risk="low",
        risk_note="Only safe if the cached or vectorized path produces the same tensors. compare's loss check catches a transform that changed.",
        tags=["data"],
    )


def rule_pin_memory(ev: dict) -> Finding | None:
    dl = _dl(ev)
    if _device(ev) != "cuda" or dl is None or dl.get("pin_memory"):
        return None
    share = (ev.get("split") or {}).get("data_wait", 0.0)
    return Finding(
        id="pin_memory_off",
        title="pin_memory is off on a CUDA run",
        score=0.03 + 0.1 * share,
        score_basis="prior 0.03 plus 0.1 x data-wait share",
        evidence="The DataLoader feeds a CUDA device with pin_memory=False, so host-to-device copies use pageable memory and can't overlap with compute.",
        numbers={"pin_memory": False, "data_wait_share": share},
        suggestion="Set pin_memory=True on the DataLoader and copy batches with .to(device, non_blocking=True).",
        expected_effect="Faster host-to-device copies and copy/compute overlap; usually a small gain unless batches are large.",
        risk="none",
        risk_note="Uses more page-locked host memory; training math is unchanged.",
        tags=["data", "cuda"],
    )


def rule_host_sync(ev: dict) -> Finding | None:
    dev = _device(ev)
    if dev not in ("cuda", "mps"):
        return None
    ps = ev.get("per_step") or {}
    share = (ev.get("split") or {}).get("host_sync", 0.0)
    calls = ps.get("sync_calls", 0.0)
    if calls < 1 or (share < SYNC_SHARE and calls < SYNC_CALLS):
        return None
    return Finding(
        id="host_sync_in_loop",
        title="Host waits on the device inside every step",
        score=max(0.05, share * 0.5),
        score_basis="half the measured sync share (time inside a sync includes device work that still has to run)",
        evidence=(
            f"{calls:.1f} blocking reads of {dev} tensors per step (.item(), .cpu(), .tolist(), float(), bool() or synchronize()), "
            f"{ps.get('sync_ms', 0):.1f} ms per step blocked in them ({_pct(share)} of step time)."
        ),
        numbers={"sync_calls_per_step": calls, "sync_ms_per_step": ps.get("sync_ms"), "sync_share": share, "device": dev},
        suggestion=(
            "Keep running metrics on the device (for example running_loss += loss.detach()) and read them with .item() "
            "only when you log, every N steps. Avoid float(t), bool(t), .cpu() and printing tensors inside the step."
        ),
        expected_effect=(
            "Lets the host queue step N+1 while the device still runs step N. The gain is at most the host-side time per "
            "step that is currently serialized; large on small models, small on big ones."
        ),
        risk="none",
        risk_note="Training math is unchanged; you log less often.",
        tags=["sync"],
    )


def rule_small_batch(ev: dict) -> Finding | None:
    dev = _device(ev)
    if dev not in ("cuda", "mps"):
        return None
    dl = _dl(ev)
    ps = ev.get("per_step") or {}
    bs = (dl or {}).get("batch_size") or ps.get("samples")
    if not bs or bs > SMALL_BATCH:
        return None
    util = _gpu_util(ev)
    step_ms = (ev.get("step_time_ms") or {}).get("median")
    memf = _mem_fraction(ev)
    low_util = util is not None and util < LOW_UTIL
    fast = step_ms is not None and step_ms < FAST_STEP_MS
    if not (low_util or (util is None and fast)):
        return None
    if memf is not None and memf >= LOW_MEMORY:
        return None
    bits = [f"batch size {bs:.0f}"]
    if util is not None:
        bits.append(f"GPU utilization {util:.0f}% mean")
    if step_ms is not None:
        bits.append(f"median step {step_ms:.1f} ms")
    if memf is not None:
        bits.append(f"{_pct(memf)} of device memory in use")
    return Finding(
        id="small_batch",
        title="Small batch leaves the device underused",
        score=0.3 * (1 - util / 100) if util is not None else 0.1,
        score_basis="0.3 x idle GPU fraction" if util is not None else "prior 0.1 (no utilization reading)",
        evidence=", ".join(bits) + ". Per-step launch overhead dominates at this size.",
        numbers={"batch_size": bs, "gpu_util_pct_mean": util, "step_ms_median": step_ms, "device_memory_fraction": memf},
        suggestion=(
            f"Try batch size {int(bs) * 2} or {int(bs) * 4}. If you change it, scale the learning rate (linear scaling is a "
            "common starting point) or keep the effective batch fixed and compare convergence over a longer run."
        ),
        expected_effect="Higher samples/s when the device has headroom. Steps per second drop; what matters is samples/s and time to reach the target loss.",
        risk="high",
        risk_note="Changes training dynamics: gradient noise, effective learning rate and BatchNorm statistics all change. compare's loss check will likely flag it; judge it on a longer run, not the first steps.",
        tags=["device", "changes-dynamics"],
    )


def rule_mixed_precision(ev: dict) -> Finding | None:
    dev = _device(ev)
    cfg = ev.get("config") or {}
    t = _torch(ev)
    used = [a for a in (cfg.get("autocast") or []) if len(a) < 3 or a[2]]
    dtypes = cfg.get("param_dtypes") or []
    if used or (dtypes and dtypes != ["float32"]):
        return None
    other = (ev.get("split") or {}).get("other", 0.0)
    if dev == "cuda":
        cap = t.get("cuda_capability") or [0, 0]
        if cap[0] < 7:
            return None
        bf16 = cap[0] >= 8
        dtype = "torch.bfloat16" if bf16 else "torch.float16"
        sugg = f"Wrap the forward pass and loss in `with torch.autocast('cuda', dtype={dtype}):`"
        if not bf16:
            sugg += " and scale the loss with torch.amp.GradScaler"
        sugg += "."
        score = 0.25 * other
        note = f"compute capability {cap[0]}.{cap[1]} has tensor cores"
    elif dev == "mps":
        if not t.get("mps_autocast_supported"):
            return None
        sugg = "Wrap the forward pass and loss in `with torch.autocast('mps', dtype=torch.float16):`. Gains on Apple GPUs vary by op, so measure."
        score = 0.1 * other
        note = "this torch build supports autocast on MPS"
    else:
        return None
    return Finding(
        id="no_mixed_precision",
        title="No mixed precision on hardware that supports it",
        score=score,
        score_basis=f"{'0.25' if dev == 'cuda' else '0.1'} x compute ('other') share of step time",
        evidence=f"Device {dev}, {note}; autocast was never entered and parameters are {', '.join(dtypes) or 'float32'}. {_pct(other)} of step time is compute and Python ('other').",
        numbers={"device": dev, "autocast_used": False, "param_dtypes": dtypes, "compute_share": other},
        suggestion=sugg,
        expected_effect="Largest on compute-bound models dominated by matmuls and convolutions; little or none when data loading or sync dominates.",
        risk="medium",
        risk_note="Lower precision changes numerics. bf16 rarely needs loss scaling; fp16 needs GradScaler. Check the loss trajectory with compare.",
        tags=["compute", "numerics"],
    )


def rule_torch_compile(ev: dict) -> Finding | None:
    dev = _device(ev)
    cfg = ev.get("config") or {}
    t = _torch(ev)
    if not t.get("compile_available") or cfg.get("compile_calls"):
        return None
    other = (ev.get("split") or {}).get("other", 0.0)
    prior = {"cuda": 0.15, "cpu": 0.06, "mps": 0.03}.get(dev or "", 0.0)
    if prior == 0.0 or other < 0.3:
        return None
    sugg = "Try `model = torch.compile(model)`."
    if dev == "mps":
        sugg += " Support for MPS is newer than for CUDA and CPU; expect some models to fall back or run slower."
    return Finding(
        id="no_torch_compile",
        title="torch.compile is not used",
        score=prior * other,
        score_basis=f"prior {prior} for {dev} x compute ('other') share",
        evidence=f"torch {t.get('version')} supports torch.compile and it was never called. {_pct(other)} of step time is compute and Python overhead, which is what compilation reduces.",
        numbers={"torch": t.get("version"), "device": dev, "compute_share": other},
        suggestion=sugg + " Compare with more warmup steps (for example --warmup-steps 20): compilation happens in the first steps.",
        expected_effect="Fewer kernel launches and less Python overhead per step. Compilation adds startup time, so it pays off on long runs.",
        risk="low",
        risk_note="Same math in principle, but fused kernels can differ in the last bits, and graph breaks or recompiles can make a run slower. Measure.",
        tags=["compute"],
    )


def rule_io(ev: dict) -> list[Finding]:
    split = ev.get("split") or {}
    ps = ev.get("per_step") or {}
    out = []
    save_share = split.get("checkpoint_save", 0.0)
    if save_share >= IO_SHARE or ps.get("save_calls", 0) >= 0.5:
        out.append(
            Finding(
                id="checkpoint_in_hot_loop",
                title="Checkpoints are written inside the step loop",
                score=save_share,
                score_basis="measured torch.save share of step time",
                evidence=f"{ps.get('save_calls', 0):.2f} torch.save calls per step, {_pct(save_share)} of step time.",
                numbers={"save_calls_per_step": ps.get("save_calls"), "save_share": save_share},
                suggestion="Checkpoint every N steps or minutes instead of every step, and keep only the latest few.",
                expected_effect=f"At most {_upper_bound(save_share):.2f}x from removing checkpoint time entirely.",
                risk="none",
                risk_note="Training is unchanged; you lose more progress if the run crashes between checkpoints.",
                tags=["io"],
            )
        )
    log_share = split.get("log_write", 0.0)
    lines = ps.get("log_lines", 0.0)
    if log_share >= IO_SHARE or lines >= 1.0:
        sync_note = " Printing a metric usually needs .item(), which also forces a device sync." if _device(ev) in ("cuda", "mps") else ""
        out.append(
            Finding(
                id="logging_in_hot_loop",
                title="Logging on every step",
                score=max(log_share, 0.02 if lines >= 1 else 0.0),
                score_basis="measured stdout/stderr write share (floor 0.02 when logging every step)",
                evidence=f"{lines:.1f} lines written to stdout/stderr per step, {_pct(log_share)} of step time spent writing them.{sync_note}",
                numbers={"log_lines_per_step": lines, "log_write_share": log_share},
                suggestion="Log every N steps (for example every 50) and aggregate metrics on the device in between.",
                expected_effect="Small by itself; larger when each log line also forces a device sync.",
                risk="none",
                risk_note="Training is unchanged.",
                tags=["io"],
            )
        )
    return out


def rule_grad_accumulation(ev: dict) -> Finding | None:
    ps = ev.get("per_step") or {}
    cfg = ev.get("config") or {}
    k = ps.get("backward_calls", 0.0)
    if k < 1.5:
        return None
    kk = round(k)
    micro = (ps.get("samples") / k) if ps.get("samples") else None
    if cfg.get("ddp") and not cfg.get("ddp_no_sync_calls"):
        return Finding(
            id="ddp_accumulation_without_no_sync",
            title="Gradient accumulation under DDP without no_sync()",
            score=0.3,
            score_basis="prior 0.3 (gradients are all-reduced on every micro-step)",
            evidence=f"{k:.1f} backward calls per optimizer step under DistributedDataParallel, and model.no_sync() was never used.",
            numbers={"backward_per_step": k, "ddp": True, "no_sync_calls": 0},
            suggestion=f"Wrap the first {kk - 1} micro-steps of each accumulation cycle in `with model.no_sync():` so gradients are all-reduced once per optimizer step.",
            expected_effect=f"Cuts gradient all-reduce traffic by about {kk}x.",
            risk="none",
            risk_note="The accumulated gradient is the same.",
            tags=["distributed"],
        )
    if _device(ev) not in ("cuda", "mps"):
        return None
    memf = _mem_fraction(ev)
    if memf is not None and memf >= LOW_MEMORY:
        return None
    return Finding(
        id="grad_accumulation_small_micro_batch",
        title="Gradient accumulation with a small micro-batch",
        score=0.08,
        score_basis="prior 0.08",
        evidence=f"{k:.1f} backward calls per optimizer step"
        + (f" with micro-batches of about {micro:.0f} samples" if micro else "")
        + (f"; {_pct(memf)} of device memory in use" if memf is not None else "")
        + ".",
        numbers={"backward_per_step": k, "micro_batch": micro, "device_memory_fraction": memf},
        suggestion="If memory allows, raise the micro-batch and cut accumulation steps by the same factor so the effective batch stays the same.",
        expected_effect="Fewer, larger kernels per optimizer step.",
        risk="low",
        risk_note="Same effective batch. BatchNorm statistics see larger micro-batches, so results can shift slightly.",
        tags=["device"],
    )


def rule_cudnn_benchmark(ev: dict) -> Finding | None:
    cfg = ev.get("config") or {}
    t = _torch(ev)
    if _device(ev) != "cuda" or not cfg.get("conv_modules") or t.get("cudnn_benchmark", True):
        return None
    return Finding(
        id="cudnn_benchmark_off",
        title="cuDNN autotuning is off for a convolutional model",
        score=0.04,
        score_basis="prior 0.04",
        evidence=f"{cfg.get('conv_modules')} convolution modules and torch.backends.cudnn.benchmark=False.",
        numbers={"conv_modules": cfg.get("conv_modules"), "cudnn_benchmark": False},
        suggestion="Set torch.backends.cudnn.benchmark = True when input shapes are fixed.",
        expected_effect="cuDNN picks the fastest convolution algorithm per shape after a short search.",
        risk="low",
        risk_note="Algorithm choice can be nondeterministic; slower if input shapes change every step.",
        tags=["cuda", "compute"],
    )


def rule_cpu_with_accelerator(ev: dict) -> Finding | None:
    t = _torch(ev)
    if _device(ev) != "cpu":
        return None
    avail = "cuda" if t.get("cuda_available") else ("mps" if t.get("mps_available") else None)
    if avail is None:
        return None
    return Finding(
        id="cpu_with_accelerator_available",
        title=f"Training runs on CPU while {avail} is available",
        score=0.2,
        score_basis="prior 0.2",
        evidence=f"Model parameters live on cpu; torch reports {avail} as available.",
        numbers={"device": "cpu", "available": avail},
        suggestion=f"Move the model and each batch to torch.device('{avail}').",
        expected_effect="Depends on model size: very small models can be faster on CPU because of launch overhead. Measure with compare.",
        risk="low",
        risk_note="Numerics differ slightly across devices.",
        tags=["device"],
    )


SINGLE_RULES = [
    rule_dataloader_bound,
    rule_cpu_preprocessing,
    rule_pin_memory,
    rule_host_sync,
    rule_small_batch,
    rule_mixed_precision,
    rule_torch_compile,
    rule_grad_accumulation,
    rule_cudnn_benchmark,
    rule_cpu_with_accelerator,
]


def diagnose(ev: dict) -> list[Finding]:
    found: list[Finding] = []
    if ev.get("source") != "hook":
        return found
    for rule in SINGLE_RULES:
        f = rule(ev)
        if f is not None:
            found.append(f)
    found.extend(rule_io(ev))
    found.sort(key=lambda f: (-f.score, f.id))
    return found
