# Changelog

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
