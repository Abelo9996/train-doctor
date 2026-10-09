# train-doctor compare: original-vs-final-stall-in-candidate

Decision: **inconclusive**. No clear difference: the 95% interval 0.94 to 2.37 does not clear the 2% threshold on either side (won 3 of 4 pairs).

| | Baseline | Candidate |
|---|---|---|
| Command | `python examples/stall_once.py --counter /tmp/td-stall-d-base.count -- examples/tabular_mlp/train.py` | `python examples/stall_once.py --counter /tmp/td-stall-d-cand.count --on-run 4 --sleep 120 -- examples/tabular_mlp/train.py --ckpt-every 200 --num-workers 5` |
| Median samples/s | 6,233.95 | 11,900.50 |
| Min to max | 5,027.81 to 10,311.02 | 104.52 to 13,201.33 |
| Spread (CV) | 30.3% | 56.9% |
| Repeats | 5 | 5 |
| Process start to first step (median) | 2.92 s | 10.68 s |

Speed ratio (candidate / baseline samples/s, median over 4 back-to-back pairs): **2.139x**, 95% interval 0.945 to 2.367. Candidate faster in 3 of 4 pairs. Verdict: **no clear difference** (threshold: interval must clear 1 +/- 2%). Wilcoxon signed-rank on the log pair ratios: two-sided p = 0.2500 (the smallest possible with 4 pairs is 0.1250).

With the stalled pairs included, the median pair ratio would be 1.989x and the interval 0.013 to 2.367.

## Pairs

| Pair | Baseline samples/s | Candidate samples/s | Ratio | Stall |
|---:|---:|---:|---:|---|
| 0 | 5,027.81 | 11,900.50 | 2.367 |  |
| 1 | 5,739.48 | 13,201.33 | 2.300 |  |
| 2 | 8,314.78 | 104.52 | 0.013 | candidate run stalled, set aside |
| 3 | 6,233.95 | 12,396.44 | 1.989 |  |
| 4 | 10,311.02 | 9,740.31 | 0.945 |  |

## Loss check

Status: **identical**. Tolerance 5% on both numbers below; curves smoothed over 5 points and aligned on samples (84 common points).

- Curve-level mean relative difference: 0.0% (largest single point 0.0%).
- Last 10 points: baseline 0.3497, candidate 0.3497, relative difference 0.0%.
- For reference, baseline repeat 0 vs repeat 1: identical, mean relative difference 0.0%.

## Per-repeat numbers (in run order)

| # | Phase | Arm | Repeat | samples/s | Exit | System CPU % | Load 1m | Directory |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | warmup | baseline | 0 | 5,733.53 | 0 | 46 | 13.1 | `warmup/baseline-0` |
| 2 | warmup | candidate | 0 | 5,744.65 | 0 | 60 | 11.8 | `warmup/candidate-0` |
| 3 | measure | baseline | 0 | 5,027.81 | 0 | 47 | 21.0 | `baseline/rep-0` |
| 4 | measure | candidate | 0 | 11,900.50 | 0 | 31 | 18.3 | `candidate/rep-0` |
| 5 | measure | candidate | 1 | 13,201.33 | 0 | 37 | 10.5 | `candidate/rep-1` |
| 6 | measure | baseline | 1 | 5,739.48 | 0 | 30 | 8.8 | `baseline/rep-1` |
| 7 | measure | baseline | 2 | 8,314.78 | 0 | 19 | 7.7 | `baseline/rep-2` |
| 8 | measure | candidate | 2 | 104.52 | 0 | 20 | 7.4 | `candidate/rep-2` |
| 9 | measure | candidate | 3 | 12,396.44 | 0 | 25 | 3.3 | `candidate/rep-3` |
| 10 | measure | baseline | 3 | 6,233.95 | 0 | 26 | 3.2 | `baseline/rep-3` |
| 11 | measure | baseline | 4 | 10,311.02 | 0 | 13 | 3.0 | `baseline/rep-4` |
| 12 | measure | candidate | 4 | 9,740.31 | 0 | 17 | 2.8 | `candidate/rep-4` |

System CPU % is the whole machine's CPU use during the run and Load 1m is the load average at its start, so background work shows up there. Warmup runs are discarded. Within each run, the first warmup steps are excluded and the window is bracketed by device synchronization.

## Settings

5 repeats per arm, paired: each repeat runs baseline and candidate back to back, order alternating between pairs (order seed 0), 1 discarded warmup run(s) per arm, 5 warmup steps and 4.0 s measured  per run, seed 0 (set before the script's own seeding). Interval: 95% percentile bootstrap of the median pair ratio (candidate / baseline), 10000 resamples, seed 0; stalled pairs set aside first (a run under half its arm's median and beyond 4 MADs, or under a quarter of it), at most 1 per 4 pairs.

## Machine

Apple M4, 10 logical cores, 16,384 MiB RAM, Darwin 25.2.0, GPU telemetry: apple. train-doctor 0.1.2.

## Limits

- Set aside 1 stalled pair(s) out of 5: candidate repeat 2 ran at 104.5 samples/s, 113.9 times slower than that arm's median 1.19e+04. Every run in an arm is the same command, so a gap that large is the machine (another process, swapping, sleep), not the change. With all pairs the interval would be 0.01 to 2.37.
- Pair ratios vary a lot (from 0.94 to 2.37), so the machine's speed changed between pairs; close other apps for a firmer answer.
- The machine was busy: the 1-minute load average reached 21.0 on 10 logical cores during the runs, so other processes competed for CPU. Pairing cancels slow stretches that hit both runs of a pair, not ones that hit only one. Rerun when the machine is quieter for a firmer answer.
- The candidate takes 7.8 s longer to reach its first step (median). That one-time cost is outside the throughput window; weigh it against the run length.
- Apple laptops change clock speeds with temperature and power state; pairing and alternating the order reduce but don't remove that drift.
