"""Resolving a bare `python` to the target repository's interpreter."""

import os

from train_doctor import runner


def _make_venv(cwd, rel=".venv"):
    binp = cwd / rel / ("Scripts" if os.name == "nt" else "bin")
    binp.mkdir(parents=True)
    exe = binp / ("python.exe" if os.name == "nt" else "python")
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    return str(exe)


def test_finds_project_venv(tmp_path):
    exe = _make_venv(tmp_path)
    path, how = runner.find_target_python(str(tmp_path))
    assert path == exe and "virtualenv" in how


def test_bare_python_rewritten_to_venv(tmp_path):
    exe = _make_venv(tmp_path)
    cmd, note = runner.resolve_target_interpreter(["python", "train.py", "--lr", "0.1"], str(tmp_path))
    assert cmd == [exe, "train.py", "--lr", "0.1"]
    assert note and "resolved" in note


def test_explicit_interpreter_left_alone(tmp_path):
    _make_venv(tmp_path)
    cmd, note = runner.resolve_target_interpreter(["/usr/bin/python3", "train.py"], str(tmp_path))
    assert cmd == ["/usr/bin/python3", "train.py"] and note is None
    cmd, note = runner.resolve_target_interpreter(["python3.12", "train.py"], str(tmp_path))
    assert cmd == ["python3.12", "train.py"] and note is None


def test_no_venv_reports_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    cmd, note = runner.resolve_target_interpreter(["python", "train.py"], str(tmp_path))
    assert cmd == ["python", "train.py"]
    assert note and "could not resolve" in note
