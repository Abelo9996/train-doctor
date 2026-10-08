# train-doctor compare: amp

Decision: **inconclusive**. No clear difference: the 95% interval 0.81 to 1.28 does not clear the 2% threshold on either side.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/cnn_images/train.py --batch-augment --log-every 50` | `python examples/cnn_images/train.py --batch-augment --log-every 50 --amp` |
| Median samples/s | 2,648.25 | 2,842.01 |
| Min to max | 2,378.03 to 2,840.31 | 2,150.54 to 3,259.01 |
| Spread (CV) | 7.2% | 15.2% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 3.69 s | 3.70 s |

Speed ratio (candidate / baseline, median samples/s): **1.073x**, 95% interval 0.812 to 1.277. Verdict: **no clear difference** (threshold: interval must clear 1 +/- 2%). Mann-Whitney U = 17.0, two-sided p = 0.4206.

## Loss check

Status: **within tolerance**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (140 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.2%).
- Last 10 points: baseline 1.2951, candidate 1.2963, relative difference 0.1%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 3,270.88 | 0 | 37 | 6.6 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 2,642.24 | 0 | 55 | 6.7 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 2,378.03 | 0 | 33 | 9.3 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 2,150.54 | 0 | 39 | 8.3 | `candidate/rep-0` |
| 5 | measure | baseline | 1 | 2,840.31 | 0 | 48 | 7.4 | `baseline/rep-1` |
| 6 | measure | candidate | 1 | 3,259.01 | 0 | 45 | 7.3 | `candidate/rep-1` |
| 7 | measure | candidate | 2 | 3,037.09 | 0 | 46 | 7.2 | `candidate/rep-2` |
| 8 | measure | baseline | 2 | 2,691.24 | 0 | 41 | 7.5 | `baseline/rep-2` |
| 9 | measure | candidate | 3 | 2,630.76 | 0 | 50 | 9.1 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 2,449.74 | 0 | 45 | 8.6 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 2,648.25 | 0 | 41 | 8.0 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 2,842.01 | 0 | 41 | 7.8 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm in randomized pair order (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 4.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the ratio of medians, 10000 resamples, seed 0.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.0.

## Limits

- Run-to-run spread is high (coefficient of variation up to 15%); close other apps or raise --repeats.
- Apple laptops change clock speeds with temperature and power state; interleaving reduces but doesn't remove that drift.
