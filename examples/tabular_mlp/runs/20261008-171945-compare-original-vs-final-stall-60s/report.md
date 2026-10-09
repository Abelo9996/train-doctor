# train-doctor compare: original-vs-final-stall-60s

Decision: **inconclusive**. No clear difference: the 95% interval 0.87 to 7.31 does not clear the 2% threshold on either side (won 3 of 4 pairs).

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/stall_once.py --counter /tmp/td-stall-e-base.count -- examples/tabular_mlp/train.py` | `python examples/stall_once.py --counter /tmp/td-stall-e-cand.count --on-run 4 --sleep 60 -- examples/tabular_mlp/train.py --ckpt-every 200 --num-workers 5` |
| Median samples/s | 3,732.65 | 6,571.14 |
| Min to max | 2,117.02 to 6,391.94 | 205.30 to 16,763.44 |
| Spread (CV) | 50.3% | 77.5% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 7.13 s | 20.44 s |

Speed ratio (candidate / baseline samples/s, median over 4 back-to-back pairs): **1.736x**, 95% interval 0.873 to 7.310. Candidate faster in 3 of 4 pairs. Verdict: **no clear difference** (threshold: interval must clear 1 +/- 2%). Wilcoxon signed-rank on the log pair ratios: two-sided p = 0.2500 (the smallest possible with 4 pairs is 0.1250).

With the stalled pairs included, the median pair ratio would be 1.713x and the interval 0.097 to 7.310.

## Pairs

| Pair | Baseline samples/s | Candidate samples/s | Ratio | Stall |
|---:|---:|---:|---:|---|
| 0 | 6,391.94 | 10,947.22 | 1.713 |  |
| 1 | 3,732.65 | 6,571.14 | 1.760 |  |
| 2 | 2,117.02 | 205.30 | 0.097 | candidate run stalled, set aside |
| 3 | 2,293.18 | 16,763.44 | 7.310 |  |
| 4 | 6,362.29 | 5,554.68 | 0.873 |  |

## Loss check

Status: **identical**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (56 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.0%).
- Last 10 points: baseline 0.3601, candidate 0.3601, relative difference 0.0%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 7,664.46 | 0 | 28 | 7.3 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 20,918.69 | 0 | 19 | 7.0 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 6,391.94 | 0 | 38 | 14.8 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 10,947.22 | 0 | 39 | 13.7 | `candidate/rep-0` |
| 5 | measure | candidate | 1 | 6,571.14 | 0 | 72 | 12.3 | `candidate/rep-1` |
| 6 | measure | baseline | 1 | 3,732.65 | 0 | 61 | 14.7 | `baseline/rep-1` |
| 7 | measure | baseline | 2 | 2,117.02 | 0 | 44 | 14.1 | `baseline/rep-2` |
| 8 | measure | candidate | 2 | 205.30 | 0 | 49 | 16.1 | `candidate/rep-2` |
| 9 | measure | candidate | 3 | 16,763.44 | 0 | 38 | 9.5 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 2,293.18 | 0 | 41 | 9.1 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 6,362.29 | 0 | 33 | 8.2 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 5,554.68 | 0 | 66 | 8.2 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm, paired: each repeat runs baseline and candidate back to back, order alternating between pairs (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 2.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the median pair ratio (candidate / baseline), 10000 resamples, seed 0; stalled pairs set aside first (a run under half its arm's median and beyond 4 MADs, or under a quarter of it), at most 1 per 4 pairs.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.2.

## Limits

- Set aside 1 stalled pair(s) out of 5: candidate repeat 2 ran at 205.3 samples/s, 32.0 times slower than that arm's median 6571. Every run in an arm is the same command, so a gap that large is the machine (another process, swapping, sleep), not the change. With all pairs the interval would be 0.10 to 7.31.
- Pair ratios vary a lot (from 0.87 to 7.31), so the machine's speed changed between pairs; close other apps for a firmer answer.
- The machine was busy: the 1-minute load average reached 16.1 on 10 logical cores during the runs, so other processes competed for CPU. Pairing cancels slow stretches that hit both runs of a pair, not ones that hit only one. Rerun when the machine is quieter for a firmer answer.
- The candidate takes 13.3 s longer to reach its first step (median). That one-time cost is outside the throughput window; weigh it against the run length.
- Apple laptops change clock speeds with temperature and power state; pairing and alternating the order reduce but don't remove that drift.
