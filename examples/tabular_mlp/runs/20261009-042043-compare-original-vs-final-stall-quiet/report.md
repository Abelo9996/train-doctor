# train-doctor compare: original-vs-final-stall-quiet

Decision: **keep**. Candidate is faster (2.67x, 95% interval 2.28 to 3.66, won 4 of 4 pairs) and the loss trajectory is identical.

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/stall_once.py --counter /tmp/td-stall-i-base.count -- examples/tabular_mlp/train.py` | `python examples/stall_once.py --counter /tmp/td-stall-i-cand.count --on-run 4 --sleep 60 -- examples/tabular_mlp/train.py --ckpt-every 200 --num-workers 5` |
| Median samples/s | 12,307.43 | 33,502.10 |
| Min to max | 9,430.09 to 14,051.33 | 211.32 to 40,929.56 |
| Spread (CV) | 15.0% | 57.2% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 1.91 s | 5.68 s |

Speed ratio (candidate / baseline samples/s, median over 4 back-to-back pairs): **2.670x**, 95% interval 2.283 to 3.656. Candidate faster in 4 of 4 pairs. Verdict: **faster** (threshold: interval must clear 1 +/- 2%). Wilcoxon signed-rank on the log pair ratios: two-sided p = 0.1250 (the smallest possible with 4 pairs is 0.1250).

With the stalled pairs included, the median pair ratio would be 2.384x and the interval 0.022 to 3.656.

## Pairs

| Pair | Baseline samples/s | Candidate samples/s | Ratio | Stall |
|---:|---:|---:|---:|---|
| 0 | 14,051.33 | 33,502.10 | 2.384 |  |
| 1 | 12,307.43 | 36,792.67 | 2.989 |  |
| 2 | 9,430.09 | 211.32 | 0.022 | candidate run stalled, set aside |
| 3 | 11,196.27 | 40,929.56 | 3.656 |  |
| 4 | 13,271.49 | 30,301.31 | 2.283 |  |

## Loss check

Status: **identical**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (116 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.0%).
- Last 10 points: baseline 0.3456, candidate 0.3456, relative difference 0.0%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 13,845.67 | 0 | 27 | 4.0 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 36,490.52 | 0 | 34 | 6.8 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 14,051.33 | 0 | 34 | 5.7 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 33,502.10 | 0 | 49 | 5.5 | `candidate/rep-0` |
| 5 | measure | candidate | 1 | 36,792.67 | 0 | 44 | 5.8 | `candidate/rep-1` |
| 6 | measure | baseline | 1 | 12,307.43 | 0 | 35 | 7.0 | `baseline/rep-1` |
| 7 | measure | baseline | 2 | 9,430.09 | 0 | 53 | 6.9 | `baseline/rep-2` |
| 8 | measure | candidate | 2 | 211.32 | 0 | 33 | 6.8 | `candidate/rep-2` |
| 9 | measure | candidate | 3 | 40,929.56 | 0 | 48 | 7.2 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 11,196.27 | 0 | 38 | 6.4 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 13,271.49 | 0 | 30 | 6.1 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 30,301.31 | 0 | 64 | 6.9 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm, paired: each repeat runs baseline and candidate back to back, order alternating between pairs (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 2.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the median pair ratio (candidate / baseline), 10000 resamples, seed 0; stalled pairs set aside first (a run under half its arm's median and beyond 4 MADs, or under a quarter of it), at most 1 per 4 pairs.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.2.

## Limits

- Set aside 1 stalled pair(s) out of 5: candidate repeat 2 ran at 211.3 samples/s, 158.5 times slower than that arm's median 3.35e+04. Every run in an arm is the same command, so a gap that large is the machine (another process, swapping, sleep), not the change. With all pairs the interval would be 0.02 to 3.66.
- A set-aside stall was in a candidate run. If the candidate stalls again when you rerun, the change itself may cause it (for example recompiling or starting workers), so don't keep calling it noise.
- Pair ratios vary a lot (from 2.28 to 3.66), so the machine's speed changed between pairs; close other apps for a firmer answer.
- The candidate takes 3.8 s longer to reach its first step (median). That one-time cost is outside the throughput window; weigh it against the run length.
- Apple laptops change clock speeds with temperature and power state; pairing and alternating the order reduce but don't remove that drift.
