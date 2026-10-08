# train-doctor compare: log-every-50

Decision: **keep**. Candidate is faster (1.25x, 95% interval 1.11 to 1.35) and the loss trajectory is identical.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/cnn_images/train.py --batch-augment` | `python examples/cnn_images/train.py --batch-augment --log-every 50` |
| Median samples/s | 2,419.93 | 3,028.53 |
| Min to max | 2,245.26 to 2,736.62 | 2,675.69 to 3,087.39 |
| Spread (CV) | 8.9% | 5.7% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 3.25 s | 2.92 s |

Speed ratio (candidate / baseline, median samples/s): **1.251x**, 95% interval 1.105 to 1.350. Verdict: **faster** (threshold: interval must clear 1 +/- 2%). Mann-Whitney U = 24.0, two-sided p = 0.0159.

## Loss check

Status: **identical**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (151 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.0%).
- Last 10 points: baseline 1.0472, candidate 1.0472, relative difference 0.0%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 2,663.53 | 0 | 54 | 7.2 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 2,672.03 | 0 | 49 | 9.3 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 2,245.26 | 0 | 52 | 9.1 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 3,072.64 | 0 | 47 | 9.2 | `candidate/rep-0` |
| 5 | measure | baseline | 1 | 2,286.17 | 0 | 43 | 8.9 | `baseline/rep-1` |
| 6 | measure | candidate | 1 | 3,024.80 | 0 | 45 | 8.3 | `candidate/rep-1` |
| 7 | measure | candidate | 2 | 3,087.39 | 0 | 49 | 8.2 | `candidate/rep-2` |
| 8 | measure | baseline | 2 | 2,736.62 | 0 | 53 | 7.4 | `baseline/rep-2` |
| 9 | measure | candidate | 3 | 3,028.53 | 0 | 37 | 8.1 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 2,419.93 | 0 | 46 | 7.8 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 2,662.98 | 0 | 42 | 7.4 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 2,675.69 | 0 | 39 | 7.2 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm in randomized pair order (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 4.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the ratio of medians, 10000 resamples, seed 0.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.0.

## Limits

- Apple laptops change clock speeds with temperature and power state; interleaving reduces but doesn't remove that drift.
