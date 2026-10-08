"""The owner's writing rules: no em or en dashes anywhere in the repo's text."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "runs", ".train-doctor"}
EXTS = {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".cfg", ".json"}


def test_no_long_dashes():
    bad = []
    for p in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts) or not p.is_file() or p.suffix not in EXTS:
            continue
        text = p.read_text(errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if chr(0x2014) in line or chr(0x2013) in line:
                bad.append(f"{p.relative_to(ROOT)}:{n}")
    assert not bad, bad
