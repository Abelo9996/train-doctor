# train-doctor compare: workers-5

Decision: **inconclusive**. No clear difference: the 95% interval 0.66 to 1.68 does not clear the 2% threshold on either side.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/cnn_images/train.py` | `python examples/cnn_images/train.py --num-workers 5` |
| Median samples/s | 2,249.46 | 2,749.88 |
| Min to max | 2,081.89 to 4,189.56 | 2,315.89 to 3,789.23 |
| Spread (CV) | 33.0% | 21.0% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 2.42 s | 3.85 s |

Speed ratio (candidate / baseline, median samples/s): **1.222x**, 95% interval 0.656 to 1.685. Verdict: **no clear difference** (threshold: interval must clear 1 +/- 2%). Mann-Whitney U = 17.0, two-sided p = 0.4206.

## Loss check

Status: **within tolerance**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (76 common points).

- Curve-level mean relative difference: 1.5% (largest single point 6.7%).
- Last 10 points: baseline 1.3399, candidate 1.3075, relative difference 2.4%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 2,040.00 | 0 | 67 | 7.3 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 1,271.92 | 0 | 75 | 7.2 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 2,249.46 | 0 | 70 | 8.2 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 3,181.91 | 0 | 66 | 8.1 | `candidate/rep-0` |
| 5 | measure | baseline | 1 | 2,081.89 | 0 | 23 | 8.6 | `baseline/rep-1` |
| 6 | measure | candidate | 1 | 2,749.88 | 0 | 47 | 8.0 | `candidate/rep-1` |
| 7 | measure | candidate | 2 | 2,414.79 | 0 | 27 | 8.2 | `candidate/rep-2` |
| 8 | measure | baseline | 2 | 3,156.77 | 0 | 16 | 7.8 | `baseline/rep-2` |
| 9 | measure | candidate | 3 | 3,789.23 | 0 | 36 | 8.7 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 4,189.56 | 0 | 35 | 8.2 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 2,124.76 | 0 | 37 | 7.7 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 2,315.89 | 0 | 37 | 7.5 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm in randomized pair order (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 2.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the ratio of medians, 10000 resamples, seed 0.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.0.

## Limits

- Run-to-run spread is high (coefficient of variation up to 33%); close other apps or raise --repeats.
- The candidate takes 1.4 s longer to reach its first step (median). That one-time cost is outside the throughput window; weigh it against the run length.
- Apple laptops change clock speeds with temperature and power state; interleaving reduces but doesn't remove that drift.
