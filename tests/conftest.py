import os
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """No test may touch the real home directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def script(tmp_path):
    def make(name: str, body: str) -> list[str]:
        p = tmp_path / name
        p.write_text(textwrap.dedent(body))
        return [sys.executable, str(p)]

    return make


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """Put executables on PATH: fake_bin('nvidia-smi', 'echo ...')."""
    d = tmp_path / "bin"
    d.mkdir()
    monkeypatch.setenv("PATH", str(d) + os.pathsep + os.environ.get("PATH", ""))

    def make(name: str, body: str) -> Path:
        p = d / name
        p.write_text("#!/bin/sh\n" + textwrap.dedent(body))
        p.chmod(0o755)
        return p

    return make
