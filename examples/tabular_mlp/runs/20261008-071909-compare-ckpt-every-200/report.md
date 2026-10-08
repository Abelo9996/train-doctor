# train-doctor compare: ckpt-every-200

Decision: **inconclusive**. No clear difference: the 95% interval 0.62 to 2.67 does not clear the 2% threshold on either side.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/tabular_mlp/train.py` | `python examples/tabular_mlp/train.py --ckpt-every 200` |
| Median samples/s | 6,944.05 | 10,680.00 |
| Min to max | 4,524.49 to 8,314.39 | 4,329.40 to 13,583.43 |
| Spread (CV) | 25.8% | 35.3% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 3.48 s | 3.24 s |

Speed ratio (candidate / baseline, median samples/s): **1.538x**, 95% interval 0.623 to 2.672. Verdict: **no clear difference** (threshold: interval must clear 1 +/- 2%). Mann-Whitney U = 20.0, two-sided p = 0.1508.

## Loss check

Status: **identical**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (81 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.0%).
- Last 10 points: baseline 0.3474, candidate 0.3474, relative difference 0.0%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 6,078.33 | 0 | 52 | 7.2 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 10,319.26 | 0 | 62 | 7.1 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 4,818.80 | 0 | 43 | 8.4 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 12,091.62 | 0 | 61 | 7.9 | `candidate/rep-0` |
| 5 | measure | baseline | 1 | 4,524.49 | 0 | 60 | 7.6 | `baseline/rep-1` |
| 6 | measure | candidate | 1 | 10,680.00 | 0 | 47 | 8.3 | `candidate/rep-1` |
| 7 | measure | candidate | 2 | 13,583.43 | 0 | 46 | 8.3 | `candidate/rep-2` |
| 8 | measure | baseline | 2 | 8,314.39 | 0 | 43 | 8.2 | `baseline/rep-2` |
| 9 | measure | candidate | 3 | 4,329.40 | 0 | 43 | 7.4 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 7,323.67 | 0 | 53 | 10.4 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 6,944.05 | 0 | 47 | 9.6 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 9,467.28 | 0 | 44 | 9.5 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm in randomized pair order (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 4.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the ratio of medians, 10000 resamples, seed 0.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.0.

## Limits

- Run-to-run spread is high (coefficient of variation up to 35%); close other apps or raise --repeats.
- Apple laptops change clock speeds with temperature and power state; interleaving reduces but doesn't remove that drift.
