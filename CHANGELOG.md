# Changelog

## 0.1.2

`compare` is now a paired comparison, so one stalled run on a busy laptop no longer decides the result.

- Each repeat runs baseline and candidate back to back, and the order alternates between pairs
  (`--order-seed` now only picks which command goes first in the first pair). Each pair gives one
  candidate/baseline throughput ratio. The verdict comes from the median pair ratio and a 95%
  percentile bootstrap interval of that median (10,000 resamples, seed 0), with the same thresholds as
  before: `faster` only when the whole interval is above 1 + min-effect. 0.1.1 compared the two arms'
  medians as independent samples, so a slow stretch of the machine could not cancel out and one
  stalled run could blow the interval open (0.11 to 22.99 in the agent session quoted in the README).
- Stalls are detected and reported instead of dominating. A run is a stall when its throughput is
  below its own arm's median by more than 2x and more than 4 scaled MADs, or by more than 4x however
  noisy the arm is (every run in an arm is the same command, so a gap like that is the machine). Pairs
  with a stalled run are set aside, at most 1 in 4 pairs (1 of 5, 2 of 9), and only when at least 3
  pairs remain. If more pairs look stalled, none are set aside and the result says the machine was too
  busy. Set-aside pairs are listed in `stalls`, `pairs` and `limits`, with the interval you would get
  with them included (`ratio_all_pairs`).
- New result fields: `pairs` (per-pair values, ratio, stall flags), `wins` (pairs the candidate won),
  `wilcoxon` (exact signed-rank test on the log pair ratios, with the smallest p the pair count allows),
  `stalls`, `ratio_all_pairs`. `mann_whitney` is gone. Reasons now say how many pairs the candidate won,
  and report.md has a per-pair table.
- With 5 or fewer pairs, the bootstrap interval of the median is the range of the pair ratios, so
  `faster` means every pair that was not set aside beat 1 + min-effect. This is stricter than 0.1.1 for
  ordinary noise. `examples/reanalyze_paired.py` re-runs the committed 0.1.1 examples through the new
  analysis: 7 of 9 verdicts are unchanged, one goes from no clear difference to faster (a stalled
  candidate run set aside; the same change was faster with 9 repeats too) and one goes from faster to
  no clear difference (one of its 5 pairs had a ratio of 1.00).
- A faster or slower verdict needs at least 4 usable pairs. With 3 the interval is the range of 3
  ratios, which covers the true ratio only 75% of the time, and a macOS CI runner called an identical
  command faster that way during this release. Results with 5 or fewer pairs state the interval's real
  coverage (1 - 2/2^n) in `limits`.
- Progress: the CLI prints one line per run to stderr with the run's throughput and an estimate of the
  time left, also with `--json` (stdout stays pure JSON; `--quiet` turns it off). The MCP `compare` tool
  sends a progress notification after every run and no longer blocks the server while it works.
- `examples/stall_once.py` freezes one chosen invocation of a training script mid-window, and
  `examples/sleep_loop/` is a loop with a known 2x speedup. With one 60 s stall in a candidate run of the
  tabular example's original vs final compare, 0.1.2 says keep (2.67x, 2.28 to 3.66, 4 of 4 pairs) where
  the unpaired analysis of the same runs gives 0.02 to 3.66. All stall runs are under `examples/*/runs/`
  and described in `examples/README.md`.

## 0.1.1

Fixes from a first-time-user audit and a real Claude Code session driving the MCP server.

- `uvx train-doctor profile -- python train.py` ran the training script with train-doctor's own
  interpreter, because uvx puts its tool environment first on PATH, so `import torch` failed even with
  the user's venv active. The training command now gets the caller's PATH without uvx's (or pipx's, or
  `uv tool`'s) environment, unless that environment has torch (`uvx --with torch ...`). The same fix
  applies to the MCP server started by `uvx train-doctor mcp`.
- A command that fails before its first step is now an error: `profile` prints the last stderr lines
  and a fix (for a missing module: which interpreter ran and how to pass the right one) and exits 1.
  The MCP result carries the same in an `error` field. Before, it printed "findings: none" and exited 0.
- `python` missing from PATH (common on macOS) now says to use `python3` or the venv's interpreter.
- `profile` and `compare` results have a `next_step` that says what to do next (which change to try
  and the compare command, keep and re-profile, revert, or rerun with more repeats). The CLI prints it
  as `next:`, and the profile output now shows each finding's expected effect.
- MCP tools: descriptions say when to use each tool and what to call next, commands are accepted as an
  argv list or a single string, run directories are resolved against `cwd` and returned as absolute
  paths, and `compare` puts `decision`, `reason` and `next_step` first.
- `report.json` and run metadata record the resolved executable of the command.
- `setup` ends with the prompt to try in a new agent session.

## 0.1.0

First version.

- `profile`: runs a training command for a bounded window (warmup plus at least N steps and T seconds) with an in-process recorder injected through `sitecustomize`. Records step times, DataLoader wait, host/device sync calls, `torch.save` and log-write time, samples per step, loss, DataLoader and precision config, process CPU and memory, and GPU telemetry (nvidia-smi, or ioreg on Apple silicon). Optional torch.profiler steps after the window and py-spy sampling.
- `diagnose`: deterministic rules for DataLoader-bound runs, expensive per-sample preprocessing, pin_memory on CUDA, host syncs in the loop, small batches, missing mixed precision, missing torch.compile, checkpoints and logging in the hot loop, gradient accumulation (including DDP without no_sync), cuDNN autotuning, and CPU training with an accelerator available.
- `compare`: discarded warmup runs, K repeats per arm in randomized pair order, same seed, median and spread, bootstrap interval on the ratio of medians, a verdict that refuses to call changes inside the noise, and a loss-trajectory check.
- `report`: Markdown and JSON bundles with machine, versions, commands, per-repeat numbers, findings, verdict and limits.
- `setup`: registers the MCP server with Claude Code, Codex and Cursor and installs the skill, with a dry run, backups and idempotent re-runs.
- `--label` names run directories; unstartable commands give a clear error instead of a traceback.
- MCP server over stdio with `profile`, `diagnose`, `compare`, `report` and `version` tools.
