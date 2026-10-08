# Notes for agents working on this repo

train-doctor is a deterministic measuring tool. It never calls an LLM API; the host agent decides what to change.

## Layout

- `src/train_doctor/_hook/`: injected into the measured process. `sitecustomize.py` installs `td_hook.Recorder` when `TRAIN_DOCTOR_RUN_DIR` is set, then chains to any other `sitecustomize`. `td_hook.py` patches torch right after it is imported (optimizer step post-hook, DataLoader iterator, sync methods, `torch.save`, autocast, GradScaler, `torch.compile`, DDP) and writes `events.<pid>.jsonl`. Standard library only.
- `runner.py`: starts the command with the hook on PYTHONPATH, samples CPU/memory/GPU, saves stdout and stderr, enforces the timeout.
- `evidence.py`: raw files to one evidence dict (step stats, throughput, time split, config, resources, profiler, py-spy, loss, limits).
- `rules.py`: evidence to ranked findings. Pure functions.
- `stats.py`, `lossdiff.py`: medians, bootstrap interval, Mann-Whitney, verdict, loss trajectory check.
- `api.py`: `profile`, `diagnose`, `compare`, `make_report`. The CLI (`cli.py`) and MCP server (`mcp_server.py`) are thin wrappers.
- `report.py`: Markdown and JSON.
- `setup_agents.py`: `train-doctor setup`.
- `skills/train-doctor/SKILL.md`: the workflow agents follow. Shipped in the wheel as `train_doctor/skill/SKILL.md`.

## Rules

- Tests are offline and use a temporary HOME (see `tests/conftest.py`). Never run `setup --yes` against a real home directory while developing.
- Keep measurement honest: warmup excluded, window bracketed by device sync, repeats interleaved, and no verdict of "faster" unless the whole interval clears the threshold.
- No em or en dashes in any text.
