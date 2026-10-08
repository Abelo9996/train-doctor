# train-doctor compare: batch-augment

Decision: **keep**. Candidate is faster (1.28x, 95% interval 1.10 to 1.35) and the loss trajectory is within tolerance.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/cnn_images/train.py` | `python examples/cnn_images/train.py --batch-augment` |
| Median samples/s | 2,232.08 | 2,854.89 |
| Min to max | 2,190.70 to 2,586.20 | 2,753.17 to 3,011.91 |
| Spread (CV) | 7.8% | 3.7% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 3.15 s | 3.05 s |

Speed ratio (candidate / baseline, median samples/s): **1.279x**, 95% interval 1.104 to 1.349. Verdict: **faster** (threshold: interval must clear 1 +/- 2%). Mann-Whitney U = 25.0, two-sided p = 0.0079.

## Loss check

Status: **within tolerance**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (145 common points).

- Curve-level mean relative difference: 1.7% (largest single point 6.5%).
- Last 10 points: baseline 1.1566, candidate 1.1751, relative difference 1.6%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 2,491.46 | 0 | 53 | 5.7 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 2,840.95 | 0 | 49 | 9.0 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 2,232.08 | 0 | 52 | 8.6 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 2,753.17 | 0 | 46 | 8.3 | `candidate/rep-0` |
| 5 | measure | baseline | 1 | 2,586.20 | 0 | 49 | 8.4 | `baseline/rep-1` |
| 6 | measure | candidate | 1 | 2,755.87 | 0 | 48 | 7.9 | `candidate/rep-1` |
| 7 | measure | candidate | 2 | 3,011.91 | 0 | 45 | 8.0 | `candidate/rep-2` |
| 8 | measure | baseline | 2 | 2,190.70 | 0 | 53 | 8.0 | `baseline/rep-2` |
| 9 | measure | candidate | 3 | 2,854.89 | 0 | 47 | 9.0 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 2,200.52 | 0 | 53 | 8.5 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 2,480.20 | 0 | 48 | 8.2 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 2,884.98 | 0 | 49 | 8.1 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm in randomized pair order (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 4.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the ratio of medians, 10000 resamples, seed 0.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.0.

## Limits

- Apple laptops change clock speeds with temperature and power state; interleaving reduces but doesn't remove that drift.
