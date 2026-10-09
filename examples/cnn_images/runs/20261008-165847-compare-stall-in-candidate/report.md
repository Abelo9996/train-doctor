# train-doctor compare: stall-in-candidate

Decision: **inconclusive**. No clear difference: the 95% interval 0.05 to 1.05 does not clear the 2% threshold on either side (won 1 of 5 pairs).

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/stall_once.py --counter /tmp/td-stall-a-base.count -- examples/cnn_images/train.py` | `python examples/stall_once.py --counter /tmp/td-stall-a-cand.count --on-run 4 --sleep 60 -- examples/cnn_images/train.py --batch-augment` |
| Median samples/s | 1,745.15 | 854.94 |
| Min to max | 561.43 to 2,010.02 | 47.03 to 2,098.61 |
| Spread (CV) | 43.8% | 92.7% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 5.74 s | 6.17 s |

Speed ratio (candidate / baseline samples/s, median over 5 back-to-back pairs): **0.490x**, 95% interval 0.045 to 1.052. Candidate faster in 1 of 5 pairs. Verdict: **no clear difference** (threshold: interval must clear 1 +/- 2%). Wilcoxon signed-rank on the log pair ratios: two-sided p = 0.1250 (the smallest possible with 5 pairs is 0.0625).

## Pairs

| Pair | Baseline samples/s | Candidate samples/s | Ratio | Stall |
|---:|---:|---:|---:|---|
| 0 | 561.43 | 317.78 | 0.566 | baseline run stalled, kept (too many stalls to set aside) |
| 1 | 2,010.02 | 943.49 | 0.469 |  |
| 2 | 1,037.14 | 47.03 | 0.045 | candidate run stalled, kept (too many stalls to set aside) |
| 3 | 1,994.81 | 2,098.61 | 1.052 |  |
| 4 | 1,745.15 | 854.94 | 0.490 |  |

## Loss check

Status: **within tolerance**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (55 common points).

- Curve-level mean relative difference: 1.5% (largest single point 4.5%).
- Last 10 points: baseline 1.5758, candidate 1.5658, relative difference 0.6%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 944.42 | 0 | 50 | 17.7 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 821.66 | 0 | 54 | 20.4 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 561.43 | 0 | 52 | 18.7 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 317.78 | 0 | 73 | 21.2 | `candidate/rep-0` |
| 5 | measure | candidate | 1 | 943.49 | 0 | 55 | 52.9 | `candidate/rep-1` |
| 6 | measure | baseline | 1 | 2,010.02 | 0 | 55 | 45.9 | `baseline/rep-1` |
| 7 | measure | baseline | 2 | 1,037.14 | 0 | 53 | 38.9 | `baseline/rep-2` |
| 8 | measure | candidate | 2 | 47.03 | 0 | 46 | 34.8 | `candidate/rep-2` |
| 9 | measure | candidate | 3 | 2,098.61 | 0 | 49 | 36.5 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 1,994.81 | 0 | 34 | 32.0 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 1,745.15 | 0 | 52 | 29.4 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 854.94 | 0 | 69 | 27.6 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm, paired: each repeat runs baseline and candidate back to back, order alternating between pairs (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 2.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the median pair ratio (candidate / baseline), 10000 resamples, seed 0; stalled pairs set aside first (a run under half its arm's median and beyond 4 MADs, or under a quarter of it), at most 1 per 4 pairs.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.2.

## Limits

- 2 of 5 pairs look stalled (baseline repeat 0 ran at 561.4 samples/s, 3.1 times slower than that arm's median 1745; candidate repeat 2 ran at 47.03 samples/s, 18.2 times slower than that arm's median 854.9). That is more than the 1 that can be set aside with 5 pairs, so they stay in and widen the interval.
- Pair ratios vary a lot (from 0.05 to 1.05), so the machine's speed changed between pairs; close other apps for a firmer answer.
- The machine was busy: the 1-minute load average reached 52.9 on 10 logical cores during the runs, so other processes competed for CPU. Pairing cancels slow stretches that hit both runs of a pair, not ones that hit only one. Rerun when the machine is quieter for a firmer answer.
- candidate repeat 0 took over 3 times the usual time to reach the first step, a sign the machine was stalled by other work. Rerun when the machine is quieter if the verdict matters.
- Apple laptops change clock speeds with temperature and power state; pairing and alternating the order reduce but don't remove that drift.
