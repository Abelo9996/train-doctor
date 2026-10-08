# train-doctor profile: batch-augment

Command: `python examples/cnn_images/train.py --batch-augment`  
Started 2026-10-08T11:13:10+00:00, wall time 7.8 s, exit code 0, 1-minute load average at start 7.40.  
Machine: Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0.  
Python 3.12.13, torch 2.14.1, device `mps`.

## Summary

| | |
|---|---|
| Steps seen | 123 (5 warmup, 113 measured, 5 profiled) |
| Step boundary | optimizer |
| Step time, median (p10 to p90) | 15.38 ms (11.66 to 24.02) |
| Step time spread (CV) | 50.9% |
| Throughput | 3,472.8 samples/s (54.26 steps/s) |
| Window | 113 steps in 2.082 s; synchronized window: device synced at window start and end |

## Where step time goes

| Part | Share | ms per step |
|---|---:|---:|
| Waiting for the DataLoader | 6.0% | 1.06 |
| Blocked in host/device sync | 37.1% | 6.55 |
| torch.save | 0.0% | 0.00 |
| Writing stdout/stderr | 0.1% | 0.01 |
| Other (forward, backward, optimizer, Python) | 56.8% | 10.03 |

share of summed step time over measured steps; 'other' is forward, backward, optimizer and Python overhead. Per step: 64 samples, 2.0 device syncs, 1.0 backward calls, 1.0 log lines.

## Findings

### 1. Host waits on the device inside every step

- Evidence: 2.0 blocking reads of mps tensors per step (.item(), .cpu(), .tolist(), float(), bool() or synchronize()), 6.5 ms per step blocked in them (37% of step time).
- Change: Keep running metrics on the device (for example running_loss += loss.detach()) and read them with .item() only when you log, every N steps. Avoid float(t), bool(t), .cpu() and printing tensors inside the step.
- Expected effect: Lets the host queue step N+1 while the device still runs step N. The gain is at most the host-side time per step that is currently serialized; large on small models, small on big ones.
- Risk to results (none): Training math is unchanged; you log less often.
- Rank score 0.185 (half the measured sync share (time inside a sync includes device work that still has to run)). Rule id `host_sync_in_loop`.

### 2. No mixed precision on hardware that supports it

- Evidence: Device mps, this torch build supports autocast on MPS; autocast was never entered and parameters are float32. 57% of step time is compute and Python ('other').
- Change: Wrap the forward pass and loss in `with torch.autocast('mps', dtype=torch.float16):`. Gains on Apple GPUs vary by op, so measure.
- Expected effect: Largest on compute-bound models dominated by matmuls and convolutions; little or none when data loading or sync dominates.
- Risk to results (medium): Lower precision changes numerics. bf16 rarely needs loss scaling; fp16 needs GradScaler. Check the loss trajectory with compare.
- Rank score 0.057 (0.1 x compute ('other') share of step time). Rule id `no_mixed_precision`.

### 3. Logging on every step

- Evidence: 1.0 lines written to stdout/stderr per step, 0% of step time spent writing them. Printing a metric usually needs .item(), which also forces a device sync.
- Change: Log every N steps (for example every 50) and aggregate metrics on the device in between.
- Expected effect: Small by itself; larger when each log line also forces a device sync.
- Risk to results (none): Training is unchanged.
- Rank score 0.020 (measured stdout/stderr write share (floor 0.02 when logging every step)). Rule id `logging_in_hot_loop`.

### 4. torch.compile is not used

- Evidence: torch 2.14.1 supports torch.compile and it was never called. 57% of step time is compute and Python overhead, which is what compilation reduces.
- Change: Try `model = torch.compile(model)`. Support for MPS is newer than for CUDA and CPU; expect some models to fall back or run slower. Compare with more warmup steps (for example --warmup-steps 20): compilation happens in the first steps.
- Expected effect: Fewer kernel launches and less Python overhead per step. Compilation adds startup time, so it pays off on long runs.
- Risk to results (low): Same math in principle, but fused kernels can differ in the last bits, and graph breaks or recompiles can make a run slower. Measure.
- Rank score 0.017 (prior 0.03 for mps x compute ('other') share). Rule id `no_torch_compile`.

## Configuration seen

- DataLoader over `SyntheticImages` (20000 items): batch_size=64, num_workers=0, pin_memory=False, persistent_workers=False
- Optimizer SGD over 114,186 parameters (float32)
- autocast: entered with enabled=False (mps float16); GradScaler: False; torch.compile calls: 0; DDP: False; conv modules: 3
- torch threads: 4; process start to first step: 3.33 s

## Resources

- Process tree CPU (measured window): mean 58%, p90 68% (100% = one core; 10 logical cores), up to 1 processes
- Peak RSS: largest process 420 MiB, whole tree 420 MiB (shared pages counted per process)
- GPU (Apple M4): utilization mean 84%, max 87% over 2 samples, system-wide (ioreg Device Utilization %)
- Device memory: 93 MiB used of 12,124 MiB

## torch.profiler (5 steps after the measured window)

Scalar reads and syncs (the profiler counts CPU and device tensors alike): `aten::_local_scalar_dense` 2.0/step, `aten::item` 2.0/step

| Op | Calls/step | Self CPU ms/step |
|---|---:|---:|
| `aten::_local_scalar_dense` | 2.0 | 67.230 |
| `aten::copy_` | 21.0 | 19.383 |
| `aten::native_batch_norm` | 3.0 | 1.406 |
| `aten::native_batch_norm_backward` | 3.0 | 0.682 |
| `aten::_mps_convolution` | 3.0 | 0.600 |
| `aten::cat` | 2.0 | 0.512 |
| `enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__` | 1.0 | 0.473 |
| `aten::mps_convolution_backward` | 3.0 | 0.454 |
| `Optimizer.step#SGD.step` | 1.0 | 0.324 |
| `aten::max_pool2d_backward` | 3.0 | 0.302 |

Chrome trace: `torch_trace.json.gz` (open in https://ui.perfetto.dev).

## Loss

123 values recorded; first 2.6227 (step 0), last 0.8361 (step 122).

## Limits

- Apple GPU utilization comes from ioreg and is system-wide: other apps using the GPU are included.
- Time inside sync calls includes waiting for queued device work, so it is not all wasted time; removing a sync helps by letting the host queue the next step earlier.

## Files

Run directory `examples/cnn_images/runs/20261008-071310-profile-batch-augment`: `events.71266.jsonl`, `evidence.json`, `findings.json`, `log_losses.jsonl`, `run.json`, `samples.jsonl`, `stderr.log`, `stdout.log`, `torch_ops.json`, `torch_trace.json.gz`
