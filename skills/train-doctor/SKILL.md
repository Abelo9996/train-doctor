---
name: train-doctor
description: Find out why a training run is slow, fix it one change at a time, and prove each speedup with repeated, interleaved measurements and a loss check. Use when a user asks why training is slow, wants a faster training loop, or wants to verify that an optimization actually helped.
---

# train-doctor

You have a measuring instrument for training loops. It runs the user's real
training command for a short bounded window, records where step time goes,
ranks the likely causes, and benchmarks a changed command against the original
with repeats. Use it instead of guessing, and never report a speedup that
`compare` did not confirm.

Tools (MCP) or CLI equivalents:

| MCP tool | CLI |
|---|---|
| `profile(command=[...])` | `train-doctor profile -- <command>` |
| `diagnose(run_dir)` | `train-doctor diagnose [RUN_DIR]` |
| `compare(baseline=[...], candidate=[...])` | `train-doctor compare --baseline "<cmd>" --candidate "<cmd>"` |
| `report(run_dir)` | `train-doctor report [RUN_DIR]` |

## Workflow

1. **Profile the unchanged command.** Use the exact command the user runs.
   The run stops by itself after the warmup steps plus the measured window
   (default: at least 50 steps and 2 seconds), so it's safe on long jobs. Steps
   are detected from `optimizer.step()`. If the training loop has no optimizer
   step (custom loops, JAX, other frameworks), add `train_doctor.step(samples=..., loss=...)`
   at the end of each step, or make sure the script prints a line containing
   `loss` each step.
2. **Read the findings in rank order.** Each finding gives the measured
   evidence, a suggested change, an upper bound or description of the expected
   effect, and the risk to results. Read `limits` too: they say what the
   numbers can't tell you.
3. **Change one thing.** Make the smallest edit that addresses the top
   finding. Prefer a command-line flag or a copy of the script so the baseline
   stays runnable. Don't batch several fixes into one comparison: you won't
   know which one helped or which one hurt.
4. **Compare.** Run `compare` with the original command as baseline and the
   changed one as candidate. Defaults: 1 discarded warmup run per arm, 5
   measured repeats per arm in randomized pair order, same seed, 95%
   bootstrap interval on the ratio of median throughput.
5. **Keep only clear wins.** Keep the change only when `decision` is `keep`:
   verdict `faster` (the whole interval clears 1 + min_effect) and the loss
   trajectory `identical` or `within tolerance`. If the decision is
   `inconclusive`, say so and either drop the change or rerun with more
   repeats (`--repeats 9`) or a longer window (`--seconds 4`). If it is
   `reject`, revert it. Give each run a `--label` so the run directories
   read as a history of what was tried.
6. **Stack.** The kept candidate becomes the new baseline. Profile it again
   (the bottleneck moves) and repeat from step 2.
7. **Report.** Give the user the compare report path(s) and quote the numbers
   from them: medians, ratio and interval, loss check. Mention every limit the
   report lists. Use the words the tool used: "no clear difference" is not
   "slightly faster".

## Rules

- Never claim a speedup inside the noise. A ratio of 1.03x with an interval
  of 0.97 to 1.09 is "no clear difference".
- Changes flagged `risk: high` (batch size, learning rate) change training
  dynamics. The loss check will usually flag them. Don't keep them on speed
  alone; tell the user what changed and that convergence needs a longer run.
- Mixed precision and torch.compile change numerics slightly. Rely on the
  loss check and say so.
- If a run fails, read `stderr.log` in the run directory before retrying.
- Don't edit the user's training code without telling them what you changed.
- train-doctor runs the command locally and sends nothing anywhere. It does
  not edit code itself; you do.

## Reading the time split

- `data_wait`: time inside `DataLoader.__next__`. High with `num_workers=0`
  means loading and compute take turns. High with workers means per-sample
  preprocessing is too slow for the worker count.
- `host_sync`: time blocked in `.item()`, `.cpu()`, `.tolist()`, `float()`,
  `bool()` on GPU/MPS tensors, or `synchronize()`. Part of it is device work
  that has to happen anyway; removing syncs helps by letting the host queue
  ahead.
- `checkpoint_save`, `log_write`: `torch.save` and stdout/stderr writes.
- `other`: forward, backward, optimizer and Python overhead.
