# train-doctor compare: ckpt-every-200-9-repeats

Decision: **keep**. Candidate is faster (2.01x, 95% interval 1.54 to 2.52) and the loss trajectory is identical.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/tabular_mlp/train.py` | `python examples/tabular_mlp/train.py --ckpt-every 200` |
| Median samples/s | 6,902.91 | 13,841.35 |
| Min to max | 4,022.79 to 7,969.68 | 8,473.98 to 15,432.54 |
| Spread (CV) | 19.5% | 18.9% |
| Repeats | 9 | 9 |
| Process start to first step (median) | 3.22 s | 3.02 s |

Speed ratio (candidate / baseline, median samples/s): **2.005x**, 95% interval 1.542 to 2.524. Verdict: **faster** (threshold: interval must clear 1 +/- 2%). Mann-Whitney U = 81.0, two-sided p = 0.0000.

## Loss check

Status: **identical**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (86 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.0%).
- Last 10 points: baseline 0.3527, candidate 0.3527, relative difference 0.0%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 6,484.13 | 0 | 42 | 4.7 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 12,889.34 | 0 | 40 | 4.6 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 5,172.38 | 0 | 51 | 4.6 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 14,513.96 | 0 | 47 | 4.8 | `candidate/rep-0` |
| 5 | measure | baseline | 1 | 7,110.93 | 0 | 39 | 5.0 | `baseline/rep-1` |
| 6 | measure | candidate | 1 | 8,473.98 | 0 | 42 | 5.1 | `candidate/rep-1` |
| 7 | measure | candidate | 2 | 13,841.35 | 0 | 44 | 5.3 | `candidate/rep-2` |
| 8 | measure | baseline | 2 | 6,652.21 | 0 | 45 | 5.5 | `baseline/rep-2` |
| 9 | measure | candidate | 3 | 15,067.76 | 0 | 45 | 5.7 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 6,902.91 | 0 | 58 | 5.6 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 5,749.38 | 0 | 43 | 5.8 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 14,043.95 | 0 | 38 | 5.8 | `candidate/rep-4` |
| 13 | measure | candidate | 5 | 15,432.54 | 0 | 39 | 5.6 | `candidate/rep-5` |
| 14 | measure | baseline | 5 | 6,959.37 | 0 | 49 | 5.8 | `baseline/rep-5` |
| 15 | measure | baseline | 6 | 4,022.79 | 0 | 42 | 6.0 | `baseline/rep-6` |
| 16 | measure | candidate | 6 | 9,633.88 | 0 | 47 | 6.0 | `candidate/rep-6` |
| 17 | measure | candidate | 7 | 11,811.39 | 0 | 58 | 5.7 | `candidate/rep-7` |
| 18 | measure | baseline | 7 | 7,660.13 | 0 | 48 | 5.7 | `baseline/rep-7` |
| 19 | measure | candidate | 8 | 12,786.00 | 0 | 48 | 5.5 | `candidate/rep-8` |
| 20 | measure | baseline | 8 | 7,969.68 | 0 | 46 | 5.5 | `baseline/rep-8` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

9 repeats per arm in randomized pair order (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 4.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the ratio of medians, 10000 resamples, seed 0.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.0.

## Limits

- Run-to-run spread is high (coefficient of variation up to 20%); close other apps or raise --repeats.
- Apple laptops change clock speeds with temperature and power state; interleaving reduces but doesn't remove that drift.
