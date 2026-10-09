# train-doctor compare: stall-in-baseline

Decision: **keep**. Candidate is faster (1.79x, 95% interval 1.30 to 1.94, won 4 of 4 pairs) and the loss trajectory is identical.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/stall_once.py --counter /tmp/td-stall-g-base.count --on-run 4 --sleep 60 -- examples/sleep_loop/train.py --step-ms 20` | `python examples/stall_once.py --counter /tmp/td-stall-g-cand.count -- examples/sleep_loop/train.py --step-ms 10` |
| Median steps/s | 41.72 | 79.60 |
| Min to max | 0.82 to 42.13 | 54.63 to 81.34 |
| Spread (CV) | 54.5% | 15.5% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 2.04 s | 2.12 s |

Speed ratio (candidate / baseline steps/s, median over 4 back-to-back pairs): **1.786x**, 95% interval 1.297 to 1.943. Candidate faster in 4 of 4 pairs. Verdict: **faster** (threshold: interval must clear 1 +/- 2%). Wilcoxon signed-rank on the log pair ratios: two-sided p = 0.1250 (the smallest possible with 4 pairs is 0.1250).

With the stalled pairs included, the median pair ratio would be 1.904x and the interval 1.297 to 99.602.

## Pairs

| Pair | Baseline steps/s | Candidate steps/s | Ratio | Stall |
|---:|---:|---:|---:|---|
| 0 | 41.80 | 79.60 | 1.904 |  |
| 1 | 41.44 | 80.52 | 1.943 |  |
| 2 | 0.82 | 81.34 | 99.602 | baseline run stalled, set aside |
| 3 | 42.13 | 54.63 | 1.297 |  |
| 4 | 41.72 | 69.92 | 1.676 |  |

## Loss check

Status: **identical**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on step (89 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.0%).
- Last 10 points: baseline 0.4661, candidate 0.4661, relative difference 0.0%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | steps/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 41.40 | 0 | 36 | 6.7 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 77.21 | 0 | 41 | 6.3 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 41.80 | 0 | 20 | 6.6 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 79.60 | 0 | 26 | 6.3 | `candidate/rep-0` |
| 5 | measure | candidate | 1 | 80.52 | 0 | 25 | 6.4 | `candidate/rep-1` |
| 6 | measure | baseline | 1 | 41.44 | 0 | 21 | 6.4 | `baseline/rep-1` |
| 7 | measure | baseline | 2 | 0.82 | 0 | 20 | 6.1 | `baseline/rep-2` |
| 8 | measure | candidate | 2 | 81.34 | 0 | 30 | 4.7 | `candidate/rep-2` |
| 9 | measure | candidate | 3 | 54.63 | 0 | 76 | 4.5 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 42.13 | 0 | 31 | 9.4 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 41.72 | 0 | 10 | 9.0 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 69.92 | 0 | 33 | 8.9 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm, paired: each repeat runs baseline and candidate back to back, order alternating between pairs (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 2.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the median pair ratio (candidate / baseline), 10000 resamples, seed 0; stalled pairs set aside first (a run under half its arm's median and beyond 4 MADs, or under a quarter of it), at most 1 per 4 pairs.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.2.

## Limits

- Samples per step were not detected in every run, so throughput is compared in steps per second. That is only fair if both commands do the same work per step.
- Set aside 1 stalled pair(s) out of 5: baseline repeat 2 ran at 0.8167 steps/s, 51.1 times slower than that arm's median 41.72. Every run in an arm is the same command, so a gap that large is the machine (another process, swapping, sleep), not the change. With all pairs the interval would be 1.30 to 99.60.
- Pair ratios vary a lot (from 1.30 to 1.94), so the machine's speed changed between pairs; close other apps for a firmer answer.
- candidate repeat 3 took over 3 times the usual time to reach the first step, a sign the machine was stalled by other work. Rerun when the machine is quieter if the verdict matters.
- Apple laptops change clock speeds with temperature and power state; pairing and alternating the order reduce but don't remove that drift.
