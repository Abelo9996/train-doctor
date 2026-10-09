# train-doctor compare: stall-in-candidate-quiet

Decision: **inconclusive**. No clear difference: the 95% interval 0.89 to 1.04 does not clear the 2% threshold on either side (won 2 of 4 pairs).

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/stall_once.py --counter /tmp/td-stall-h-base.count -- examples/cnn_images/train.py` | `python examples/stall_once.py --counter /tmp/td-stall-h-cand.count --on-run 4 --sleep 60 -- examples/cnn_images/train.py --batch-augment` |
| Median samples/s | 4,237.29 | 3,806.42 |
| Min to max | 3,606.80 to 4,610.86 | 52.52 to 4,200.48 |
| Spread (CV) | 11.5% | 55.3% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 1.33 s | 1.58 s |

Speed ratio (candidate / baseline samples/s, median over 4 back-to-back pairs): **1.016x**, 95% interval 0.891 to 1.045. Candidate faster in 2 of 4 pairs. Verdict: **no clear difference** (threshold: interval must clear 1 +/- 2%). Wilcoxon signed-rank on the log pair ratios: two-sided p = 1.0000 (the smallest possible with 4 pairs is 0.1250).

With the stalled pairs included, the median pair ratio would be 0.991x and the interval 0.012 to 1.045.

## Pairs

| Pair | Baseline samples/s | Candidate samples/s | Ratio | Stall |
|---:|---:|---:|---:|---|
| 0 | 4,237.29 | 4,200.48 | 0.991 |  |
| 1 | 4,610.86 | 4,106.60 | 0.891 |  |
| 2 | 4,532.71 | 52.52 | 0.012 | candidate run stalled, set aside |
| 3 | 3,606.80 | 3,768.52 | 1.045 |  |
| 4 | 3,658.32 | 3,806.42 | 1.040 |  |

## Loss check

Status: **within tolerance**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (137 common points).

- Curve-level mean relative difference: 1.7% (largest single point 6.3%).
- Last 10 points: baseline 1.2672, candidate 1.2643, relative difference 0.2%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 4,231.51 | 0 | 47 | 4.2 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 3,605.01 | 0 | 54 | 4.2 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 4,237.29 | 0 | 52 | 4.6 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 4,200.48 | 0 | 36 | 4.9 | `candidate/rep-0` |
| 5 | measure | candidate | 1 | 4,106.60 | 0 | 41 | 4.9 | `candidate/rep-1` |
| 6 | measure | baseline | 1 | 4,610.86 | 0 | 34 | 5.2 | `baseline/rep-1` |
| 7 | measure | baseline | 2 | 4,532.71 | 0 | 50 | 5.1 | `baseline/rep-2` |
| 8 | measure | candidate | 2 | 52.52 | 0 | 38 | 5.1 | `candidate/rep-2` |
| 9 | measure | candidate | 3 | 3,768.52 | 0 | 22 | 5.2 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 3,606.80 | 0 | 33 | 4.8 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 3,658.32 | 0 | 34 | 4.5 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 3,806.42 | 0 | 42 | 4.5 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm, paired: each repeat runs baseline and candidate back to back, order alternating between pairs (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 2.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the median pair ratio (candidate / baseline), 10000 resamples, seed 0; stalled pairs set aside first (a run under half its arm's median and beyond 4 MADs, or under a quarter of it), at most 1 per 4 pairs.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.2.

## Limits

- Set aside 1 stalled pair(s) out of 5: candidate repeat 2 ran at 52.52 samples/s, 72.5 times slower than that arm's median 3806. Every run in an arm is the same command, so a gap that large is the machine (another process, swapping, sleep), not the change. With all pairs the interval would be 0.01 to 1.04.
- Apple laptops change clock speeds with temperature and power state; pairing and alternating the order reduce but don't remove that drift.
