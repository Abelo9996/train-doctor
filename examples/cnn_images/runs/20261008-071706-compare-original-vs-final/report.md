# train-doctor compare: original-vs-final

Decision: **keep**. Candidate is faster (1.62x, 95% interval 1.37 to 3.48) and the loss trajectory is within tolerance.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/cnn_images/train.py` | `python examples/cnn_images/train.py --batch-augment --log-every 50` |
| Median samples/s | 2,090.67 | 3,380.70 |
| Min to max | 972.27 to 2,423.95 | 3,048.30 to 3,984.94 |
| Spread (CV) | 29.6% | 10.0% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 3.21 s | 3.08 s |

Speed ratio (candidate / baseline, median samples/s): **1.617x**, 95% interval 1.373 to 3.477. Verdict: **faster** (threshold: interval must clear 1 +/- 2%). Mann-Whitney U = 25.0, two-sided p = 0.0079.

## Loss check

Status: **within tolerance**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (66 common points).

- Curve-level mean relative difference: 1.6% (largest single point 5.0%).
- Last 10 points: baseline 1.4670, candidate 1.4284, relative difference 2.6%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 1,799.16 | 0 | 48 | 7.1 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 3,220.00 | 0 | 45 | 6.8 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 972.27 | 0 | 43 | 6.9 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 3,328.45 | 0 | 41 | 8.3 | `candidate/rep-0` |
| 5 | measure | baseline | 1 | 1,873.50 | 0 | 39 | 7.9 | `baseline/rep-1` |
| 6 | measure | candidate | 1 | 3,395.54 | 0 | 42 | 8.0 | `candidate/rep-1` |
| 7 | measure | candidate | 2 | 3,380.70 | 0 | 61 | 7.6 | `candidate/rep-2` |
| 8 | measure | baseline | 2 | 2,090.67 | 0 | 42 | 7.8 | `baseline/rep-2` |
| 9 | measure | candidate | 3 | 3,984.94 | 0 | 51 | 7.1 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 2,266.65 | 0 | 43 | 9.5 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 2,423.95 | 0 | 42 | 8.5 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 3,048.30 | 0 | 45 | 8.2 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm in randomized pair order (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 4.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the ratio of medians, 10000 resamples, seed 0.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.0.

## Limits

- Run-to-run spread is high (coefficient of variation up to 30%); close other apps or raise --repeats.
- Apple laptops change clock speeds with temperature and power state; interleaving reduces but doesn't remove that drift.
