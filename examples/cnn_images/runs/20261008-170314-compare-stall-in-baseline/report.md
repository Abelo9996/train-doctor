# train-doctor compare: stall-in-baseline

Decision: **inconclusive**. No clear difference: the 95% interval 0.80 to 4.62 does not clear the 2% threshold on either side (won 2 of 4 pairs).

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/stall_once.py --counter /tmp/td-stall-b-base.count --on-run 4 --sleep 60 -- examples/cnn_images/train.py` | `python examples/stall_once.py --counter /tmp/td-stall-b-cand.count -- examples/cnn_images/train.py --batch-augment` |
| Median samples/s | 546.85 | 1,391.79 |
| Min to max | 48.14 to 1,747.88 | 461.27 to 3,434.80 |
| Spread (CV) | 90.5% | 69.9% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 5.31 s | 5.92 s |

Speed ratio (candidate / baseline samples/s, median over 4 back-to-back pairs): **1.136x**, 95% interval 0.796 to 4.623. Candidate faster in 2 of 4 pairs. Verdict: **no clear difference** (threshold: interval must clear 1 +/- 2%). Wilcoxon signed-rank on the log pair ratios: two-sided p = 0.6250 (the smallest possible with 4 pairs is 0.1250).

With the stalled pairs included, the median pair ratio would be 1.299x and the interval 0.796 to 71.344.

## Pairs

| Pair | Baseline samples/s | Candidate samples/s | Ratio | Stall |
|---:|---:|---:|---:|---|
| 0 | 1,747.88 | 1,391.79 | 0.796 |  |
| 1 | 911.96 | 906.55 | 0.994 |  |
| 2 | 48.14 | 3,434.80 | 71.344 | baseline run stalled, set aside |
| 3 | 355.12 | 461.27 | 1.299 |  |
| 4 | 546.85 | 2,528.14 | 4.623 |  |

## Loss check

Status: **within tolerance**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (55 common points).

- Curve-level mean relative difference: 1.5% (largest single point 4.5%).
- Last 10 points: baseline 1.5758, candidate 1.5658, relative difference 0.6%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 2,146.89 | 0 | 45 | 31.0 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 1,643.25 | 0 | 52 | 26.8 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 1,747.88 | 0 | 39 | 24.4 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 1,391.79 | 0 | 64 | 21.4 | `candidate/rep-0` |
| 5 | measure | candidate | 1 | 906.55 | 0 | 33 | 19.6 | `candidate/rep-1` |
| 6 | measure | baseline | 1 | 911.96 | 0 | 46 | 18.5 | `baseline/rep-1` |
| 7 | measure | baseline | 2 | 48.14 | 0 | 59 | 17.0 | `baseline/rep-2` |
| 8 | measure | candidate | 2 | 3,434.80 | 0 | 55 | 16.8 | `candidate/rep-2` |
| 9 | measure | candidate | 3 | 461.27 | 0 | 54 | 34.6 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 355.12 | 0 | 61 | 28.6 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 546.85 | 0 | 46 | 26.5 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 2,528.14 | 0 | 57 | 21.8 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm, paired: each repeat runs baseline and candidate back to back, order alternating between pairs (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 2.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the median pair ratio (candidate / baseline), 10000 resamples, seed 0; stalled pairs set aside first (a run under half its arm's median and beyond 4 MADs, or under a quarter of it), at most 1 per 4 pairs.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.2.

## Limits

- Set aside 1 stalled pair(s) out of 5: baseline repeat 2 ran at 48.14 samples/s, 11.4 times slower than that arm's median 546.8. Every run in an arm is the same command, so a gap that large is the machine (another process, swapping, sleep), not the change. With all pairs the interval would be 0.80 to 71.34.
- Pair ratios vary a lot (from 0.80 to 4.62), so the machine's speed changed between pairs; close other apps for a firmer answer.
- The machine was busy: the 1-minute load average reached 34.6 on 10 logical cores during the runs, so other processes competed for CPU. Pairing cancels slow stretches that hit both runs of a pair, not ones that hit only one. Rerun when the machine is quieter for a firmer answer.
- candidate repeat 2 took over 3 times the usual time to reach the first step, a sign the machine was stalled by other work. Rerun when the machine is quieter if the verdict matters.
- Apple laptops change clock speeds with temperature and power state; pairing and alternating the order reduce but don't remove that drift.
