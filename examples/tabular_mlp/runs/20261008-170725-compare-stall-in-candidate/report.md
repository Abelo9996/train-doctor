# train-doctor compare: stall-in-candidate

Decision: **inconclusive**. No clear difference: the 95% interval 0.57 to 1.85 does not clear the 2% threshold on either side (won 2 of 4 pairs).

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/stall_once.py --counter /tmp/td-stall-c-base.count -- examples/tabular_mlp/train.py` | `python examples/stall_once.py --counter /tmp/td-stall-c-cand.count --on-run 4 --sleep 60 -- examples/tabular_mlp/train.py --ckpt-every 200` |
| Median samples/s | 3,586.68 | 2,863.58 |
| Min to max | 1,346.42 to 5,595.66 | 193.75 to 10,374.85 |
| Spread (CV) | 48.2% | 99.8% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 6.50 s | 13.32 s |

Speed ratio (candidate / baseline samples/s, median over 4 back-to-back pairs): **0.963x**, 95% interval 0.568 to 1.854. Candidate faster in 2 of 4 pairs. Verdict: **no clear difference** (threshold: interval must clear 1 +/- 2%). Wilcoxon signed-rank on the log pair ratios: two-sided p = 1.0000 (the smallest possible with 4 pairs is 0.1250).

With the stalled pairs included, the median pair ratio would be 0.798x and the interval 0.144 to 1.854.

## Pairs

| Pair | Baseline samples/s | Candidate samples/s | Ratio | Stall |
|---:|---:|---:|---:|---|
| 0 | 4,981.93 | 2,832.03 | 0.568 |  |
| 1 | 3,586.68 | 2,863.58 | 0.798 |  |
| 2 | 1,346.42 | 193.75 | 0.144 | candidate run stalled, set aside |
| 3 | 2,536.49 | 2,946.34 | 1.162 |  |
| 4 | 5,595.66 | 10,374.85 | 1.854 |  |

## Loss check

Status: **identical**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (55 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.0%).
- Last 10 points: baseline 0.3637, candidate 0.3637, relative difference 0.0%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 2,053.24 | 0 | 54 | 18.2 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 6,256.47 | 0 | 52 | 14.7 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 4,981.93 | 0 | 44 | 13.4 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 2,832.03 | 0 | 60 | 12.8 | `candidate/rep-0` |
| 5 | measure | candidate | 1 | 2,863.58 | 0 | 72 | 23.5 | `candidate/rep-1` |
| 6 | measure | baseline | 1 | 3,586.68 | 0 | 59 | 24.2 | `baseline/rep-1` |
| 7 | measure | baseline | 2 | 1,346.42 | 0 | 74 | 21.7 | `baseline/rep-2` |
| 8 | measure | candidate | 2 | 193.75 | 0 | 46 | 28.0 | `candidate/rep-2` |
| 9 | measure | candidate | 3 | 2,946.34 | 0 | 42 | 17.3 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 2,536.49 | 0 | 38 | 17.8 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 5,595.66 | 0 | 43 | 15.5 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 10,374.85 | 0 | 52 | 14.8 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm, paired: each repeat runs baseline and candidate back to back, order alternating between pairs (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 2.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the median pair ratio (candidate / baseline), 10000 resamples, seed 0; stalled pairs set aside first (a run under half its arm's median and beyond 4 MADs, or under a quarter of it), at most 1 per 4 pairs.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.2.

## Limits

- Set aside 1 stalled pair(s) out of 5: candidate repeat 2 ran at 193.8 samples/s, 14.8 times slower than that arm's median 2864. Every run in an arm is the same command, so a gap that large is the machine (another process, swapping, sleep), not the change. With all pairs the interval would be 0.14 to 1.85.
- Pair ratios vary a lot (from 0.57 to 1.85), so the machine's speed changed between pairs; close other apps for a firmer answer.
- The machine was busy: the 1-minute load average reached 28.0 on 10 logical cores during the runs, so other processes competed for CPU. Pairing cancels slow stretches that hit both runs of a pair, not ones that hit only one. Rerun when the machine is quieter for a firmer answer.
- The candidate takes 6.8 s longer to reach its first step (median). That one-time cost is outside the throughput window; weigh it against the run length.
- Apple laptops change clock speeds with temperature and power state; pairing and alternating the order reduce but don't remove that drift.
