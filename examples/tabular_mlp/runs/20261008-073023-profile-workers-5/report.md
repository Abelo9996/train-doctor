# train-doctor profile: workers-5

Command: `python examples/tabular_mlp/train.py --ckpt-every 200 --num-workers 5`  
Started 2026-10-08T11:30:23+00:00, wall time 41.5 s, exit code 0, 1-minute load average at start 4.98.  
Machine: Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0.  
Python 3.12.13, torch 2.14.1, device `mps`.

## Summary

| | |
|---|---|
| Steps seen | 206 (5 warmup, 196 measured, 5 profiled) |
| Step boundary | optimizer |
| Step time, median (p10 to p90) | 8.38 ms (5.39 to 14.84) |
| Step time spread (CV) | 148.0% |
| Throughput | 24,201.3 samples/s (94.54 steps/s) |
| Window | 196 steps in 2.073 s; synchronized window: device synced at window start and end |

## Where step time goes

| Part | Share | ms per step |
|---|---:|---:|
| Waiting for the DataLoader | 7.3% | 0.75 |
| Blocked in host/device sync | 2.3% | 0.23 |
| torch.save | 8.8% | 0.91 |
| Writing stdout/stderr | 0.0% | 0.00 |
| Other (forward, backward, optimizer, Python) | 81.6% | 8.46 |

share of summed step time over measured steps; 'other' is forward, backward, optimizer and Python overhead. Per step: 256 samples, 0.1 device syncs, 1.0 backward calls, 0.1 log lines.

## Findings

### 1. Checkpoints are written inside the step loop

- Evidence: 0.01 torch.save calls per step, 9% of step time.
- Change: Checkpoint every N steps or minutes instead of every step, and keep only the latest few.
- Expected effect: At most 1.10x from removing checkpoint time entirely.
- Risk to results (none): Training is unchanged; you lose more progress if the run crashes between checkpoints.
- Rank score 0.088 (measured torch.save share of step time). Rule id `checkpoint_in_hot_loop`.

### 2. No mixed precision on hardware that supports it

- Evidence: Device mps, this torch build supports autocast on MPS; autocast was never entered and parameters are float32. 82% of step time is compute and Python ('other').
- Change: Wrap the forward pass and loss in `with torch.autocast('mps', dtype=torch.float16):`. Gains on Apple GPUs vary by op, so measure.
- Expected effect: Largest on compute-bound models dominated by matmuls and convolutions; little or none when data loading or sync dominates.
- Risk to results (medium): Lower precision changes numerics. bf16 rarely needs loss scaling; fp16 needs GradScaler. Check the loss trajectory with compare.
- Rank score 0.082 (0.1 x compute ('other') share of step time). Rule id `no_mixed_precision`.

### 3. torch.compile is not used

- Evidence: torch 2.14.1 supports torch.compile and it was never called. 82% of step time is compute and Python overhead, which is what compilation reduces.
- Change: Try `model = torch.compile(model)`. Support for MPS is newer than for CUDA and CPU; expect some models to fall back or run slower. Compare with more warmup steps (for example --warmup-steps 20): compilation happens in the first steps.
- Expected effect: Fewer kernel launches and less Python overhead per step. Compilation adds startup time, so it pays off on long runs.
- Risk to results (low): Same math in principle, but fused kernels can differ in the last bits, and graph breaks or recompiles can make a run slower. Measure.
- Rank score 0.024 (prior 0.03 for mps x compute ('other') share). Rule id `no_torch_compile`.

## Configuration seen

- DataLoader over `RawRows` (60000 items): batch_size=256, num_workers=5, pin_memory=False, persistent_workers=True
- Optimizer AdamW over 2,311,170 parameters (float32)
- autocast: entered with enabled=False (mps float16); GradScaler: False; torch.compile calls: 0; DDP: False; conv modules: 0
- torch threads: 4; process start to first step: 10.57 s

## Resources

- Process tree CPU (measured window): mean 195%, p90 216% (100% = one core; 10 logical cores), up to 12 processes
- Peak RSS: largest process 353 MiB, whole tree 1,278 MiB (shared pages counted per process)
- GPU (Apple M4): utilization mean 84%, max 87% over 2 samples, system-wide (ioreg Device Utilization %)
- Device memory: 91 MiB used of 12,124 MiB

## torch.profiler (5 steps after the measured window)

Scalar reads and syncs (the profiler counts CPU and device tensors alike): `aten::item` 8.0/step, `aten::_local_scalar_dense` 8.0/step

| Op | Calls/step | Self CPU ms/step |
|---|---:|---:|
| `aten::copy_` | 42.0 | 36.433 |
| `aten::addcdiv_` | 8.0 | 0.698 |
| `aten::mm` | 7.0 | 0.649 |
| `aten::addcmul_` | 8.0 | 0.648 |
| `Optimizer.step#AdamW.step` | 1.0 | 0.607 |
| `aten::linear` | 4.0 | 0.562 |
| `enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__` | 1.0 | 0.521 |
| `aten::nll_loss_forward` | 1.0 | 0.379 |
| `aten::_log_softmax` | 1.0 | 0.290 |
| `aten::nll_loss_backward` | 1.0 | 0.230 |

Chrome trace: `torch_trace.json.gz` (open in https://ui.perfetto.dev).

## Loss

206 values recorded; first 0.7010 (step 0), last 0.2873 (step 205).

## Limits

- Apple GPU utilization comes from ioreg and is system-wide: other apps using the GPU are included.
- Dataset __getitem__ time is only measured when num_workers=0 (it runs in worker processes otherwise).

## Files

Run directory `examples/tabular_mlp/runs/20261008-073023-profile-workers-5`: `events.93578.jsonl`, `evidence.json`, `findings.json`, `log_losses.jsonl`, `run.json`, `samples.jsonl`, `stderr.log`, `stdout.log`, `torch_ops.json`, `torch_trace.json.gz`
