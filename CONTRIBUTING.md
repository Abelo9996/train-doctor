# Contributing

```bash
uv venv --python 3.12
uv pip install -e . pytest ruff
uv run --no-sync pytest -q          # offline, under 2 minutes
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
```

Torch tests run when torch is installed (`uv pip install torch`) and are skipped otherwise.

Ground rules:

- Every number in docs and examples comes from a real run; say where it came from.
- New diagnosis rules go in `src/train_doctor/rules.py` with a test in `tests/test_rules.py` built on hand-written evidence. State the threshold and the ranking basis.
- `src/train_doctor/_hook/td_hook.py` runs inside the user's Python: standard library only, never raise into the training loop.
- No em or en dashes in text (a test checks).

See `AGENTS.md` for how the pieces fit together.
