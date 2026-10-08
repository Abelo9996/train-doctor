# Changelog

## 0.1.0 (unreleased)

First version.

- `profile`: runs a training command for a bounded window (warmup plus at least N steps and T seconds) with an in-process recorder injected through `sitecustomize`. Records step times, DataLoader wait, host/device sync calls, `torch.save` and log-write time, samples per step, loss, DataLoader and precision config, process CPU and memory, and GPU telemetry (nvidia-smi, or ioreg on Apple silicon). Optional torch.profiler steps after the window and py-spy sampling.
- `diagnose`: deterministic rules for DataLoader-bound runs, expensive per-sample preprocessing, pin_memory on CUDA, host syncs in the loop, small batches, missing mixed precision, missing torch.compile, checkpoints and logging in the hot loop, gradient accumulation (including DDP without no_sync), cuDNN autotuning, and CPU training with an accelerator available.
- `compare`: discarded warmup runs, K repeats per arm in randomized pair order, same seed, median and spread, bootstrap interval on the ratio of medians, a verdict that refuses to call changes inside the noise, and a loss-trajectory check.
- `report`: Markdown and JSON bundles with machine, versions, commands, per-repeat numbers, findings, verdict and limits.
- `setup`: registers the MCP server with Claude Code, Codex and Cursor and installs the skill, with a dry run, backups and idempotent re-runs.
- `--label` names run directories; unstartable commands give a clear error instead of a traceback.
- MCP server over stdio with `profile`, `diagnose`, `compare`, `report` and `version` tools.
