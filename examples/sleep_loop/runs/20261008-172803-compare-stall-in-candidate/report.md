# train-doctor compare: stall-in-candidate

Decision: **keep**. Candidate is faster (1.92x, 95% interval 1.91 to 1.94, won 4 of 4 pairs) and the loss trajectory is identical.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/stall_once.py --counter /tmp/td-stall-f-base.count -- examples/sleep_loop/train.py --step-ms 20` | `python examples/stall_once.py --counter /tmp/td-stall-f-cand.count --on-run 4 --sleep 60 -- examples/sleep_loop/train.py --step-ms 10` |
| Median steps/s | 41.39 | 80.23 |
| Min to max | 40.50 to 41.93 | 0.82 to 80.74 |
| Spread (CV) | 1.4% | 55.2% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 2.05 s | 2.74 s |

Speed ratio (candidate / baseline steps/s, median over 4 back-to-back pairs): **1.920x**, 95% interval 1.906 to 1.944. Candidate faster in 4 of 4 pairs. Verdict: **faster** (threshold: interval must clear 1 +/- 2%). Wilcoxon signed-rank on the log pair ratios: two-sided p = 0.1250 (the smallest possible with 4 pairs is 0.1250).

With the stalled pairs included, the median pair ratio would be 1.915x and the interval 0.020 to 1.944.

## Pairs

| Pair | Baseline steps/s | Candidate steps/s | Ratio | Stall |
|---:|---:|---:|---:|---|
| 0 | 41.28 | 78.67 | 1.906 |  |
| 1 | 41.93 | 80.74 | 1.926 |  |
| 2 | 40.50 | 0.82 | 0.020 | candidate run stalled, set aside |
| 3 | 41.39 | 80.46 | 1.944 |  |
| 4 | 41.89 | 80.23 | 1.915 |  |

## Loss check

Status: **identical**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on step (88 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.0%).
- Last 10 points: baseline 0.4794, candidate 0.4794, relative difference 0.0%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | steps/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 41.22 | 0 | 39 | 20.7 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 72.87 | 0 | 59 | 18.1 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 41.28 | 0 | 61 | 16.9 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 78.67 | 0 | 48 | 16.1 | `candidate/rep-0` |
| 5 | measure | candidate | 1 | 80.74 | 0 | 39 | 15.1 | `candidate/rep-1` |
| 6 | measure | baseline | 1 | 41.93 | 0 | 45 | 14.4 | `baseline/rep-1` |
| 7 | measure | baseline | 2 | 40.50 | 0 | 39 | 13.7 | `baseline/rep-2` |
| 8 | measure | candidate | 2 | 0.82 | 0 | 33 | 12.4 | `candidate/rep-2` |
| 9 | measure | candidate | 3 | 80.46 | 0 | 24 | 6.9 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 41.39 | 0 | 21 | 7.3 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 41.89 | 0 | 34 | 6.9 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 80.23 | 0 | 25 | 6.9 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm, paired: each repeat runs baseline and candidate back to back, order alternating between pairs (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 2.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the median pair ratio (candidate / baseline), 10000 resamples, seed 0; stalled pairs set aside first (a run under half its arm's median and beyond 4 MADs, or under a quarter of it), at most 1 per 4 pairs.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.2.

## Limits

- Samples per step were not detected in every run, so throughput is compared in steps per second. That is only fair if both commands do the same work per step.
- Set aside 1 stalled pair(s) out of 5: candidate repeat 2 ran at 0.824 steps/s, 97.4 times slower than that arm's median 80.23. Every run in an arm is the same command, so a gap that large is the machine (another process, swapping, sleep), not the change. With all pairs the interval would be 0.02 to 1.94.
- A set-aside stall was in a candidate run. If the candidate stalls again when you rerun, the change itself may cause it (for example recompiling or starting workers), so don't keep calling it noise.
- The machine was busy: the 1-minute load average reached 16.9 on 10 logical cores during the runs, so other processes competed for CPU. Pairing cancels slow stretches that hit both runs of a pair, not ones that hit only one. Rerun when the machine is quieter for a firmer answer.
- Apple laptops change clock speeds with temperature and power state; pairing and alternating the order reduce but don't remove that drift.
