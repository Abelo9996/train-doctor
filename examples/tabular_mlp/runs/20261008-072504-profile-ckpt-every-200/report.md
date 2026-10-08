# train-doctor profile: ckpt-every-200

Command: `python examples/tabular_mlp/train.py --ckpt-every 200`  
Started 2026-10-08T11:25:04+00:00, wall time 6.8 s, exit code 0, 1-minute load average at start 5.33.  
Machine: Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0.  
Python 3.12.13, torch 2.14.1, device `mps`.

## Summary

| | |
|---|---|
| Steps seen | 60 (5 warmup, 50 measured, 5 profiled) |
| Step boundary | optimizer |
| Step time, median (p10 to p90) | 28.57 ms (15.01 to 75.08) |
| Step time spread (CV) | 78.1% |
| Throughput | 6,374.3 samples/s (24.90 steps/s) |
| Window | 50 steps in 2.008 s; synchronized window: device synced at window start and end |

## Where step time goes

| Part | Share | ms per step |
|---|---:|---:|
| Waiting for the DataLoader | 34.3% | 13.70 |
| Blocked in host/device sync | 3.1% | 1.24 |
| torch.save | 0.0% | 0.00 |
| Writing stdout/stderr | 0.0% | 0.00 |
| Other (forward, backward, optimizer, Python) | 62.6% | 25.00 |

share of summed step time over measured steps; 'other' is forward, backward, optimizer and Python overhead. Per step: 256 samples, 0.0 device syncs, 1.0 backward calls, 0.0 log lines.

## Findings

### 1. Training waits on the DataLoader (num_workers=0)

- Evidence: 34% of step time is spent waiting for the next batch (13.7 ms per step) and the DataLoader runs with num_workers=0, so loading and the training step take turns on one core.
- Change: Set num_workers=5 and persistent_workers=True on the DataLoader. On macOS and Windows workers start with spawn: keep the training code under `if __name__ == "__main__":` and make sure the dataset pickles.
- Expected effect: At most 1.52x (the step time without any data wait), reached only if loading fully overlaps compute.
- Risk to results (low): Deterministic datasets give identical batches. Random augmentation in workers draws from per-worker RNG streams, so exact samples differ (same distribution). Each worker holds a copy of the dataset in memory.
- Rank score 0.343 (measured data-wait share of step time). Rule id `dataloader_bound`.

### 2. Per-sample preprocessing in RawRows is expensive

- Evidence: 34% of step time is data wait and 69% of that is inside RawRows.__getitem__ (0.04 ms per sample).
- Change: Make preprocessing cheaper instead of (or as well as) parallelizing it: precompute or cache deterministic transforms once, move per-element Python loops to vectorized tensor ops on the whole batch (or on the device), and keep only cheap random augmentation per sample.
- Expected effect: At most 1.52x if data wait disappears entirely.
- Risk to results (low): Only safe if the cached or vectorized path produces the same tensors. compare's loss check catches a transform that changed.
- Rank score 0.212 (data-wait share times the __getitem__ share of it, times 0.9 so the one-line worker fix ranks first when both apply). Rule id `cpu_bound_preprocessing`.

### 3. No mixed precision on hardware that supports it

- Evidence: Device mps, this torch build supports autocast on MPS; autocast was never entered and parameters are float32. 63% of step time is compute and Python ('other').
- Change: Wrap the forward pass and loss in `with torch.autocast('mps', dtype=torch.float16):`. Gains on Apple GPUs vary by op, so measure.
- Expected effect: Largest on compute-bound models dominated by matmuls and convolutions; little or none when data loading or sync dominates.
- Risk to results (medium): Lower precision changes numerics. bf16 rarely needs loss scaling; fp16 needs GradScaler. Check the loss trajectory with compare.
- Rank score 0.063 (0.1 x compute ('other') share of step time). Rule id `no_mixed_precision`.

### 4. torch.compile is not used

- Evidence: torch 2.14.1 supports torch.compile and it was never called. 63% of step time is compute and Python overhead, which is what compilation reduces.
- Change: Try `model = torch.compile(model)`. Support for MPS is newer than for CUDA and CPU; expect some models to fall back or run slower. Compare with more warmup steps (for example --warmup-steps 20): compilation happens in the first steps.
- Expected effect: Fewer kernel launches and less Python overhead per step. Compilation adds startup time, so it pays off on long runs.
- Risk to results (low): Same math in principle, but fused kernels can differ in the last bits, and graph breaks or recompiles can make a run slower. Measure.
- Rank score 0.019 (prior 0.03 for mps x compute ('other') share). Rule id `no_torch_compile`.

## Configuration seen

- DataLoader over `RawRows` (60000 items): batch_size=256, num_workers=0, pin_memory=False, persistent_workers=False
- Optimizer AdamW over 2,311,170 parameters (float32)
- autocast: entered with enabled=False (mps float16); GradScaler: False; torch.compile calls: 0; DDP: False; conv modules: 0
- torch threads: 4; process start to first step: 3.01 s

## Resources

- Process tree CPU (measured window): mean 45%, p90 87% (100% = one core; 10 logical cores), up to 1 processes
- Peak RSS: largest process 432 MiB, whole tree 432 MiB (shared pages counted per process)
- GPU (Apple M4): utilization mean 100%, max 100% over 2 samples, system-wide (ioreg Device Utilization %)
- Device memory: 91 MiB used of 12,124 MiB

## torch.profiler (5 steps after the measured window)

Scalar reads and syncs (the profiler counts CPU and device tensors alike): `aten::item` 8.0/step, `aten::_local_scalar_dense` 8.0/step

| Op | Calls/step | Self CPU ms/step |
|---|---:|---:|
| `enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__` | 1.0 | 6.562 |
| `aten::copy_` | 42.0 | 3.314 |
| `aten::addcmul_` | 8.0 | 0.310 |
| `aten::linear` | 4.0 | 0.301 |
| `aten::addcdiv_` | 8.0 | 0.290 |
| `aten::mm` | 7.0 | 0.259 |
| `Optimizer.step#AdamW.step` | 1.0 | 0.230 |
| `aten::_log_softmax` | 1.0 | 0.140 |
| `aten::empty` | 516.0 | 0.140 |
| `aten::nll_loss_forward` | 1.0 | 0.131 |

Chrome trace: `torch_trace.json.gz` (open in https://ui.perfetto.dev).

## Loss

60 values recorded; first 0.7010 (step 0), last 0.2965 (step 59).

## Limits

- Apple GPU utilization comes from ioreg and is system-wide: other apps using the GPU are included.

## Files

Run directory `examples/tabular_mlp/runs/20261008-072504-profile-ckpt-every-200`: `events.86802.jsonl`, `evidence.json`, `findings.json`, `log_losses.jsonl`, `run.json`, `samples.jsonl`, `stderr.log`, `stdout.log`, `torch_ops.json`, `torch_trace.json.gz`
