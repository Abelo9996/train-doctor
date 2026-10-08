"""First-run and agent-facing behavior: the right interpreter, failures that say the fix, limits that explain noise."""

import importlib.util
import os
import sys

from train_doctor import api
from train_doctor.runner import child_path, tool_env_bin


def _tool_env(tmp_path, monkeypatch, marker=None):
    env = tmp_path / "archive-v0" / "AbCdEf" if marker is None else tmp_path / "tool-env"
    (env / "bin").mkdir(parents=True)
    if marker:
        (env / marker).write_text("")
    monkeypatch.setattr(sys, "prefix", str(env))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base-python"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "torch" else real(name, *a))
    return env


def test_uvx_env_is_dropped_from_child_path(tmp_path, monkeypatch):
    env = _tool_env(tmp_path, monkeypatch)
    user = tmp_path / "user-venv" / "bin"
    path = os.pathsep.join([str(env / "bin"), str(user), "/usr/bin"])
    assert tool_env_bin() == str(env / "bin")
    assert child_path(path).split(os.pathsep) == [str(user), "/usr/bin"]


def test_uv_tool_and_pipx_envs_are_detected(tmp_path, monkeypatch):
    for marker in ("uv-receipt.toml", "pipx_metadata.json"):
        env = _tool_env(tmp_path / marker, monkeypatch, marker)
        assert tool_env_bin() == str(env / "bin")


def test_users_own_env_is_kept(tmp_path, monkeypatch):
    env = _tool_env(tmp_path, monkeypatch)
    monkeypatch.setenv("VIRTUAL_ENV", str(env))
    path = os.pathsep.join([str(env / "bin"), "/usr/bin"])
    assert tool_env_bin() is None
    assert child_path(path) == path


def test_project_venv_is_kept(tmp_path, monkeypatch):
    # A uv project venv also has CACHEDIR.TAG; only uvx's cache envs and installed tool envs count.
    env = _tool_env(tmp_path, monkeypatch, "CACHEDIR.TAG")
    assert env.name == "tool-env"
    assert tool_env_bin() is None


def test_tool_env_with_torch_is_kept(tmp_path, monkeypatch):
    _tool_env(tmp_path, monkeypatch)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object())
    assert tool_env_bin() is None


def test_failed_command_reports_stderr_and_fix(tmp_path, script):
    cmd = script("broken.py", "import definitely_not_a_module_td\n")
    res = api.profile(cmd, out_dir=tmp_path / "runs", profiler_steps=0, warmup_steps=1, steps=5, seconds=0)
    err = res["error"]
    assert err["exit_code"] == 1
    assert "No module named 'definitely_not_a_module_td'" in err["stderr_tail"]
    assert "can't import definitely_not_a_module_td" in err["hint"]
    assert res["next_step"].startswith("The command failed before the first step")


def test_compare_names_busy_machine_and_stalled_repeats(tmp_path, monkeypatch):
    calls = iter(range(100))

    def fake_run(cfg, run_dir, machine, kind, label=None):
        i = next(calls)
        stalled = i == 3  # one measured repeat starts 30 s late
        return {
            "source": "hook",
            "steps": {"end_reason": "window_complete"},
            "throughput": {"samples_per_s": 100.0 + i, "steps_per_s": 10.0},
            "loss": [],
            "run": {"exit_code": 0, "load_avg_1m": 9.5, "time_to_first_step_s": 30.0 if stalled else 2.0},
        }

    monkeypatch.setattr(api, "_single_run", fake_run)
    monkeypatch.setattr(api, "machine_info", lambda: {"cores_logical": 4, "gpu_backend": "none"})
    res = api.compare([sys.executable, "-c", "0"], [sys.executable, "-c", "1"], repeats=3, warmup_runs=1, out_dir=tmp_path / "runs")
    limits = " ".join(res["limits"])
    assert "load average reached 9.5 on 4 logical cores" in limits
    assert "took over 3 times the usual time to reach the first step" in limits
    assert "when the machine is quieter" in res["next_step"]
