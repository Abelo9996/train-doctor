# train-doctor profile: log-every-50

Command: `python examples/cnn_images/train.py --batch-augment --log-every 50`  
Started 2026-10-08T11:15:07+00:00, wall time 6.8 s, exit code 0, 1-minute load average at start 6.84.  
Machine: Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0.  
Python 3.12.13, torch 2.14.1, device `mps`.

## Summary

| | |
|---|---|
| Steps seen | 99 (5 warmup, 89 measured, 5 profiled) |
| Step boundary | optimizer |
| Step time, median (p10 to p90) | 12.43 ms (9.69 to 74.40) |
| Step time spread (CV) | 125.6% |
| Throughput | 2,640.1 samples/s (41.25 steps/s) |
| Window | 89 steps in 2.157 s; synchronized window: device synced at window start and end |

## Where step time goes

| Part | Share | ms per step |
|---|---:|---:|
| Waiting for the DataLoader | 4.0% | 0.94 |
| Blocked in host/device sync | 0.2% | 0.04 |
| torch.save | 0.0% | 0.00 |
| Writing stdout/stderr | 0.0% | 0.00 |
| Other (forward, backward, optimizer, Python) | 95.8% | 22.49 |

share of summed step time over measured steps; 'other' is forward, backward, optimizer and Python overhead. Per step: 64 samples, 0.0 device syncs, 1.0 backward calls, 0.0 log lines.

## Findings

### 1. No mixed precision on hardware that supports it

- Evidence: Device mps, this torch build supports autocast on MPS; autocast was never entered and parameters are float32. 96% of step time is compute and Python ('other').
- Change: Wrap the forward pass and loss in `with torch.autocast('mps', dtype=torch.float16):`. Gains on Apple GPUs vary by op, so measure.
- Expected effect: Largest on compute-bound models dominated by matmuls and convolutions; little or none when data loading or sync dominates.
- Risk to results (medium): Lower precision changes numerics. bf16 rarely needs loss scaling; fp16 needs GradScaler. Check the loss trajectory with compare.
- Rank score 0.096 (0.1 x compute ('other') share of step time). Rule id `no_mixed_precision`.

### 2. torch.compile is not used

- Evidence: torch 2.14.1 supports torch.compile and it was never called. 96% of step time is compute and Python overhead, which is what compilation reduces.
- Change: Try `model = torch.compile(model)`. Support for MPS is newer than for CUDA and CPU; expect some models to fall back or run slower. Compare with more warmup steps (for example --warmup-steps 20): compilation happens in the first steps.
- Expected effect: Fewer kernel launches and less Python overhead per step. Compilation adds startup time, so it pays off on long runs.
- Risk to results (low): Same math in principle, but fused kernels can differ in the last bits, and graph breaks or recompiles can make a run slower. Measure.
- Rank score 0.029 (prior 0.03 for mps x compute ('other') share). Rule id `no_torch_compile`.

## Configuration seen

- DataLoader over `SyntheticImages` (20000 items): batch_size=64, num_workers=0, pin_memory=False, persistent_workers=False
- Optimizer SGD over 114,186 parameters (float32)
- autocast: entered with enabled=False (mps float16); GradScaler: False; torch.compile calls: 0; DDP: False; conv modules: 3
- torch threads: 4; process start to first step: 2.92 s

## Resources

- Process tree CPU (measured window): mean 42%, p90 74% (100% = one core; 10 logical cores), up to 1 processes
- Peak RSS: largest process 439 MiB, whole tree 439 MiB (shared pages counted per process)
- GPU (Apple M4): utilization mean 88%, max 90% over 2 samples, system-wide (ioreg Device Utilization %)
- Device memory: 93 MiB used of 12,124 MiB

## torch.profiler (5 steps after the measured window)

| Op | Calls/step | Self CPU ms/step |
|---|---:|---:|
| `aten::copy_` | 21.0 | 85.449 |
| `aten::native_batch_norm` | 3.0 | 1.466 |
| `aten::native_batch_norm_backward` | 3.0 | 0.875 |
| `aten::_mps_convolution` | 3.0 | 0.712 |
| `aten::mps_convolution_backward` | 3.0 | 0.699 |
| `LogSoftmaxBackward0` | 1.0 | 0.476 |
| `aten::cat` | 2.0 | 0.468 |
| `aten::max_pool2d_backward` | 3.0 | 0.439 |
| `Optimizer.step#SGD.step` | 1.0 | 0.374 |
| `enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__` | 1.0 | 0.314 |

Chrome trace: `torch_trace.json.gz` (open in https://ui.perfetto.dev).

## Loss

99 values recorded; first 2.6227 (step 0), last 0.9619 (step 98).

## Limits

- Apple GPU utilization comes from ioreg and is system-wide: other apps using the GPU are included.

## Files

Run directory `examples/cnn_images/runs/20261008-071507-profile-log-every-50`: `events.73232.jsonl`, `evidence.json`, `findings.json`, `log_losses.jsonl`, `run.json`, `samples.jsonl`, `stderr.log`, `stdout.log`, `torch_ops.json`, `torch_trace.json.gz`
