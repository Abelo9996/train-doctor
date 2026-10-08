# train-doctor compare: workers-5

Decision: **keep**. Candidate is faster (1.93x, 95% interval 1.54 to 2.86) and the loss trajectory is identical.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/tabular_mlp/train.py --ckpt-every 200` | `python examples/tabular_mlp/train.py --ckpt-every 200 --num-workers 5` |
| Median samples/s | 8,900.19 | 17,176.86 |
| Min to max | 6,009.96 to 10,110.09 | 13,995.47 to 18,337.58 |
| Spread (CV) | 18.7% | 12.2% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 3.59 s | 9.73 s |

Speed ratio (candidate / baseline, median samples/s): **1.930x**, 95% interval 1.540 to 2.858. Verdict: **faster** (threshold: interval must clear 1 +/- 2%). Mann-Whitney U = 25.0, two-sided p = 0.0079.

## Loss check

Status: **identical**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (99 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.0%).
- Last 10 points: baseline 0.3249, candidate 0.3249, relative difference 0.0%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 16,739.06 | 0 | 42 | 5.8 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 14,264.64 | 0 | 53 | 5.8 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 6,009.96 | 0 | 46 | 5.2 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 17,441.95 | 0 | 62 | 5.7 | `candidate/rep-0` |
| 5 | measure | baseline | 1 | 10,110.09 | 0 | 49 | 5.5 | `baseline/rep-1` |
| 6 | measure | candidate | 1 | 13,995.47 | 0 | 48 | 5.2 | `candidate/rep-1` |
| 7 | measure | candidate | 2 | 14,264.57 | 0 | 56 | 7.4 | `candidate/rep-2` |
| 8 | measure | baseline | 2 | 9,090.08 | 0 | 49 | 5.6 | `baseline/rep-2` |
| 9 | measure | candidate | 3 | 17,176.86 | 0 | 54 | 5.5 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 7,718.36 | 0 | 41 | 5.3 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 8,900.19 | 0 | 37 | 5.4 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 18,337.58 | 0 | 58 | 5.6 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm in randomized pair order (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 4.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the ratio of medians, 10000 resamples, seed 0.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.0.

## Limits

- Run-to-run spread is high (coefficient of variation up to 19%); close other apps or raise --repeats.
- The candidate takes 6.1 s longer to reach its first step (median). That one-time cost is outside the throughput window; weigh it against the run length.
- Apple laptops change clock speeds with temperature and power state; interleaving reduces but doesn't remove that drift.
