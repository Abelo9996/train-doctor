# train-doctor compare: original-vs-final

Decision: **keep**. Candidate is faster (3.05x, 95% interval 2.15 to 3.40) and the loss trajectory is identical.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/tabular_mlp/train.py` | `python examples/tabular_mlp/train.py --ckpt-every 200 --num-workers 5` |
| Median samples/s | 5,483.51 | 16,699.97 |
| Min to max | 5,010.20 to 6,685.04 | 12,727.03 to 18,079.38 |
| Spread (CV) | 12.5% | 14.6% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 3.84 s | 10.27 s |

Speed ratio (candidate / baseline, median samples/s): **3.045x**, 95% interval 2.145 to 3.402. Verdict: **faster** (threshold: interval must clear 1 +/- 2%). Mann-Whitney U = 25.0, two-sided p = 0.0079.

## Loss check

Status: **identical**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (111 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.0%).
- Last 10 points: baseline 0.3490, candidate 0.3490, relative difference 0.0%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 7,015.71 | 0 | 48 | 5.0 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 16,932.67 | 0 | 47 | 5.1 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 6,685.04 | 0 | 44 | 5.0 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 14,340.21 | 0 | 51 | 4.9 | `candidate/rep-0` |
| 5 | measure | baseline | 1 | 6,383.75 | 0 | 54 | 6.8 | `baseline/rep-1` |
| 6 | measure | candidate | 1 | 17,877.50 | 0 | 59 | 6.5 | `candidate/rep-1` |
| 7 | measure | candidate | 2 | 12,727.03 | 0 | 36 | 5.5 | `candidate/rep-2` |
| 8 | measure | baseline | 2 | 5,314.59 | 0 | 46 | 5.5 | `baseline/rep-2` |
| 9 | measure | candidate | 3 | 18,079.38 | 0 | 54 | 5.3 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 5,483.51 | 0 | 45 | 4.9 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 5,010.20 | 0 | 47 | 5.4 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 16,699.97 | 0 | 53 | 5.2 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm in randomized pair order (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 4.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the ratio of medians, 10000 resamples, seed 0.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.0.

## Limits

- Run-to-run spread is high (coefficient of variation up to 15%); close other apps or raise --repeats.
- The candidate takes 6.4 s longer to reach its first step (median). That one-time cost is outside the throughput window; weigh it against the run length.
- Apple laptops change clock speeds with temperature and power state; interleaving reduces but doesn't remove that drift.
