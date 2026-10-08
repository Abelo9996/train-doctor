# train-doctor profile: baseline

Command: `python examples/cnn_images/train.py`  
Started 2026-10-08T11:09:03+00:00, wall time 9.4 s, exit code 0, 1-minute load average at start 7.36.  
Machine: Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0.  
Python 3.12.13, torch 2.14.1, device `mps`.

## Summary

| | |
|---|---|
| Steps seen | 97 (5 warmup, 87 measured, 5 profiled) |
| Step boundary | optimizer |
| Step time, median (p10 to p90) | 20.88 ms (14.32 to 32.84) |
| Step time spread (CV) | 40.8% |
| Throughput | 2,766.8 samples/s (43.23 steps/s) |
| Window | 87 steps in 2.012 s; synchronized window: device synced at window start and end |

## Where step time goes

| Part | Share | ms per step |
|---|---:|---:|
| Waiting for the DataLoader | 39.1% | 9.01 |
| Blocked in host/device sync | 26.4% | 6.08 |
| torch.save | 0.0% | 0.00 |
| Writing stdout/stderr | 0.1% | 0.02 |
| Other (forward, backward, optimizer, Python) | 34.4% | 7.91 |

share of summed step time over measured steps; 'other' is forward, backward, optimizer and Python overhead. Per step: 64 samples, 2.0 device syncs, 1.0 backward calls, 1.0 log lines.

## Findings

### 1. Training waits on the DataLoader (num_workers=0)

- Evidence: 39% of step time is spent waiting for the next batch (9.0 ms per step) and the DataLoader runs with num_workers=0, so loading and the training step take turns on one core.
- Change: Set num_workers=5 and persistent_workers=True on the DataLoader. On macOS and Windows workers start with spawn: keep the training code under `if __name__ == "__main__":` and make sure the dataset pickles.
- Expected effect: At most 1.64x (the step time without any data wait), reached only if loading fully overlaps compute.
- Risk to results (low): Deterministic datasets give identical batches. Random augmentation in workers draws from per-worker RNG streams, so exact samples differ (same distribution). Each worker holds a copy of the dataset in memory.
- Rank score 0.391 (measured data-wait share of step time). Rule id `dataloader_bound`.

### 2. Per-sample preprocessing in SyntheticImages is expensive

- Evidence: 39% of step time is data wait and 91% of that is inside SyntheticImages.__getitem__ (0.13 ms per sample).
- Change: Make preprocessing cheaper instead of (or as well as) parallelizing it: precompute or cache deterministic transforms once, move per-element Python loops to vectorized tensor ops on the whole batch (or on the device), and keep only cheap random augmentation per sample.
- Expected effect: At most 1.64x if data wait disappears entirely.
- Risk to results (low): Only safe if the cached or vectorized path produces the same tensors. compare's loss check catches a transform that changed.
- Rank score 0.319 (data-wait share times the __getitem__ share of it, times 0.9 so the one-line worker fix ranks first when both apply). Rule id `cpu_bound_preprocessing`.

### 3. Host waits on the device inside every step

- Evidence: 2.0 blocking reads of mps tensors per step (.item(), .cpu(), .tolist(), float(), bool() or synchronize()), 6.1 ms per step blocked in them (26% of step time).
- Change: Keep running metrics on the device (for example running_loss += loss.detach()) and read them with .item() only when you log, every N steps. Avoid float(t), bool(t), .cpu() and printing tensors inside the step.
- Expected effect: Lets the host queue step N+1 while the device still runs step N. The gain is at most the host-side time per step that is currently serialized; large on small models, small on big ones.
- Risk to results (none): Training math is unchanged; you log less often.
- Rank score 0.132 (half the measured sync share (time inside a sync includes device work that still has to run)). Rule id `host_sync_in_loop`.

### 4. No mixed precision on hardware that supports it

- Evidence: Device mps, this torch build supports autocast on MPS; autocast was never entered and parameters are float32. 34% of step time is compute and Python ('other').
- Change: Wrap the forward pass and loss in `with torch.autocast('mps', dtype=torch.float16):`. Gains on Apple GPUs vary by op, so measure.
- Expected effect: Largest on compute-bound models dominated by matmuls and convolutions; little or none when data loading or sync dominates.
- Risk to results (medium): Lower precision changes numerics. bf16 rarely needs loss scaling; fp16 needs GradScaler. Check the loss trajectory with compare.
- Rank score 0.034 (0.1 x compute ('other') share of step time). Rule id `no_mixed_precision`.

### 5. Logging on every step

- Evidence: 1.0 lines written to stdout/stderr per step, 0% of step time spent writing them. Printing a metric usually needs .item(), which also forces a device sync.
- Change: Log every N steps (for example every 50) and aggregate metrics on the device in between.
- Expected effect: Small by itself; larger when each log line also forces a device sync.
- Risk to results (none): Training is unchanged.
- Rank score 0.020 (measured stdout/stderr write share (floor 0.02 when logging every step)). Rule id `logging_in_hot_loop`.

### 6. torch.compile is not used

- Evidence: torch 2.14.1 supports torch.compile and it was never called. 34% of step time is compute and Python overhead, which is what compilation reduces.
- Change: Try `model = torch.compile(model)`. Support for MPS is newer than for CUDA and CPU; expect some models to fall back or run slower. Compare with more warmup steps (for example --warmup-steps 20): compilation happens in the first steps.
- Expected effect: Fewer kernel launches and less Python overhead per step. Compilation adds startup time, so it pays off on long runs.
- Risk to results (low): Same math in principle, but fused kernels can differ in the last bits, and graph breaks or recompiles can make a run slower. Measure.
- Rank score 0.010 (prior 0.03 for mps x compute ('other') share). Rule id `no_torch_compile`.

## Configuration seen

- DataLoader over `SyntheticImages` (20000 items): batch_size=64, num_workers=0, pin_memory=False, persistent_workers=False
- Optimizer SGD over 114,186 parameters (float32)
- autocast: entered with enabled=False (mps float16); GradScaler: False; torch.compile calls: 0; DDP: False; conv modules: 3
- torch threads: 4; process start to first step: 4.60 s

## Resources

- Process tree CPU (measured window): mean 70%, p90 73% (100% = one core; 10 logical cores), up to 1 processes
- Peak RSS: largest process 432 MiB, whole tree 432 MiB (shared pages counted per process)
- GPU (Apple M4): utilization mean 74%, max 74% over 2 samples, system-wide (ioreg Device Utilization %)
- Device memory: 93 MiB used of 12,124 MiB

## torch.profiler (5 steps after the measured window)

Scalar reads and syncs (the profiler counts CPU and device tensors alike): `aten::_local_scalar_dense` 66.0/step, `aten::item` 66.0/step

| Op | Calls/step | Self CPU ms/step |
|---|---:|---:|
| `aten::_local_scalar_dense` | 66.0 | 3.659 |
| `enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__` | 1.0 | 2.686 |
| `aten::copy_` | 144.0 | 2.091 |
| `aten::reflection_pad2d` | 64.0 | 2.060 |
| `aten::native_batch_norm` | 3.0 | 1.131 |
| `aten::cat` | 2.0 | 0.855 |
| `aten::_mps_convolution` | 3.0 | 0.533 |
| `aten::native_batch_norm_backward` | 3.0 | 0.504 |
| `aten::div` | 128.0 | 0.417 |
| `aten::mps_convolution_backward` | 3.0 | 0.367 |

Chrome trace: `torch_trace.json.gz` (open in https://ui.perfetto.dev).

## Loss

97 values recorded; first 2.5092 (step 0), last 1.1540 (step 96).

## Limits

- Apple GPU utilization comes from ioreg and is system-wide: other apps using the GPU are included.
- Time inside sync calls includes waiting for queued device work, so it is not all wasted time; removing a sync helps by letting the host queue the next step earlier.

## Files

Run directory `examples/cnn_images/runs/20261008-070903-profile-baseline`: `events.62564.jsonl`, `evidence.json`, `findings.json`, `log_losses.jsonl`, `run.json`, `samples.jsonl`, `stderr.log`, `stdout.log`, `torch_ops.json`, `torch_trace.json.gz`
