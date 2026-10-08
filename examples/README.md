# Examples

Two small PyTorch training scripts with realistic slow spots, and the real train-doctor runs on them.
Every directory under `*/runs/` is untouched tool output: `report.md` for reading, `report.json` and
`compare.json` for the numbers, and the raw per-step events, resource samples, logs and profiler output.

Machine: Apple M4 MacBook (10 cores, 16 GB), macOS 26 (Darwin 25.2), Python 3.12.13, torch 2.14.1, device MPS.
Other jobs were running on the machine during these runs (1-minute load average 4.6 to 10.4 at the start
of each run); the per-repeat tables in the compare reports list the load and system CPU of every run.

Each command-line flag on the scripts stands in for one code edit, so a single change can be compared
on its own. [run_validation.sh](run_validation.sh) is the exact sequence of commands.

## cnn_images

A 3-layer CNN on 20,000 synthetic 32x32 images: per-sample crop, flip and normalize in `__getitem__`,
`num_workers=0`, and loss and accuracy read with `.item()` and printed every step.

| Run | What happened |
|---|---|
| `profile-baseline` | 39.1% of step time waiting on the DataLoader (91% of that in `__getitem__`), 26.4% blocked in 2 `.item()` calls per step. Top finding: `dataloader_bound`. |
| `compare-workers-5` | `--num-workers 5`: 1.22x by medians but the interval is 0.66 to 1.68, spread up to 33%. Inconclusive, dropped. |
| `compare-batch-augment` | Next finding, `cpu_bound_preprocessing`: augmentation moved to one vectorized call per batch on the device. 1.28x (1.10 to 1.35), loss within tolerance (1.7% mean difference). Kept. |
| `profile-batch-augment` | Data wait down to 6.0%; host sync now 37.1% of step time. Top finding: `host_sync_in_loop`. |
| `compare-log-every-50` | Metrics kept on the device and read every 50 steps. 1.25x (1.11 to 1.35), loss identical. Kept. |
| `profile-log-every-50` | Host sync 0.2%, compute and Python 95.8%. Top finding: `no_mixed_precision`. |
| `compare-amp` | `--amp` (autocast float16 on MPS): 1.07x (0.81 to 1.28). Inconclusive, dropped. |
| `compare-original-vs-final` | Original vs `--batch-augment --log-every 50`: 1.62x (1.37 to 3.48), loss within tolerance. |

## tabular_mlp

A 4-layer MLP on 60,000 synthetic CSV-like rows, featurized in plain Python inside `__getitem__`
(parse, normalize, one-hot, hashed text buckets), with a `torch.save` checkpoint every 5 steps.

| Run | What happened |
|---|---|
| `profile-baseline` | 52.8% of step time in `torch.save`, 28.2% waiting on the DataLoader. Top finding: `checkpoint_in_hot_loop`. |
| `compare-ckpt-every-200` | Checkpoint every 200 steps: 1.54x by medians, interval 0.62 to 2.67 (spread up to 35%). Inconclusive. |
| `compare-ckpt-every-200-9-repeats` | Same change with 9 repeats: 2.01x (1.54 to 2.52), loss identical. Kept. |
| `profile-ckpt-every-200` | Data wait 34.3% with `num_workers=0`. Top finding: `dataloader_bound`. |
| `compare-workers-5` | `--num-workers 5`: 1.93x (1.54 to 2.86), loss identical. Kept. The candidate needs 6.1 s more to reach its first step (spawned workers import torch); the report states that separately. |
| `profile-workers-5` | Data wait 7.3%; the window caught one checkpoint (8.8%). Next findings: mixed precision, torch.compile. Stopped here. |
| `compare-original-vs-final` | Original vs `--ckpt-every 200 --num-workers 5`: 3.05x (2.15 to 3.40), loss identical. |

Throughput medians differ between compares of the same command (for example the CNN baseline at 2,091
to 2,249 samples/s) because each compare ran at a different time on a shared machine. That is why every
decision comes from baseline and candidate runs interleaved in the same compare, never from numbers
taken at different times.

## Reproduce

```bash
uv venv --python 3.12 && uv pip install -e . torch numpy
PATH="$PWD/.venv/bin:$PATH" sh examples/run_validation.sh
```

Your numbers will differ on a different machine or under different load, and so can the decisions.
