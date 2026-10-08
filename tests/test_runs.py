"""Real child processes, no torch: markers, stdout fallback, chaining, timeout, compare, py-spy."""

import json
import sys

from train_doctor import api
from train_doctor.cli import main
from train_doctor.runner import RunConfig, run

MARKER_LOOP = """
import sys, time
import train_doctor
delay = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
for i in range(1000):
    time.sleep(delay)
    train_doctor.step(samples=8, loss=1.0 / (i + 1))
print("finished all steps")
"""

FAST = {"warmup_steps": 2, "steps": 10, "seconds": 0.0}


def test_profile_with_markers_stops_after_window(tmp_path, script):
    cmd = script("loop.py", MARKER_LOOP)
    res = api.profile(cmd, out_dir=tmp_path / "runs", profiler_steps=0, **FAST)
    ev = res["evidence"]
    assert ev["source"] == "hook"
    assert ev["steps"]["measured"] == 10 and ev["steps"]["warmup"] == 2
    assert ev["steps"]["end_reason"] == "window_complete"
    assert ev["run"]["exit_code"] == 0
    assert 300 < ev["throughput"]["samples_per_s"] < 850  # 8 samples per ~10 ms sleep
    assert ev["loss"][0] == [0, 1.0]
    run_dir = tmp_path / "runs"
    out = next(run_dir.iterdir())
    assert "finished all steps" not in (out / "stdout.log").read_text()
    assert (out / "report.md").exists() and (out / "report.json").exists()
    bundle = json.loads((out / "report.json").read_text())
    assert bundle["kind"] == "profile" and "evidence" in bundle


def test_stdout_loss_lines_fallback(tmp_path, script):
    cmd = script(
        "logs.py",
        """
        import time
        for i in range(25):
            time.sleep(0.02)
            print(f"step {i} loss: {2.0 - i * 0.05:.3f}")
    """,
    )
    ev = api.profile(cmd, out_dir=tmp_path / "runs", profiler_steps=0, **FAST)["evidence"]
    assert ev["source"] == "log_lines"
    assert len(ev["loss"]) == 25
    assert 20 < ev["throughput"]["steps_per_s"] < 60


def test_existing_sitecustomize_still_runs(tmp_path, script):
    other = tmp_path / "other"
    other.mkdir()
    flag = tmp_path / "flag.txt"
    (other / "sitecustomize.py").write_text(f"open({str(flag)!r}, 'w').write('ran')\n")
    cmd = script("loop.py", MARKER_LOOP)
    cfg = RunConfig(cmd=cmd, warmup_steps=1, steps=3, seconds=0.0, env={"PYTHONPATH": str(other)})
    meta = run(cfg, tmp_path / "r")
    assert meta["exit_code"] == 0
    assert flag.read_text() == "ran"
    assert list((tmp_path / "r").glob("events.*.jsonl"))


def test_hook_is_inert_without_run_dir(tmp_path, script):
    cmd = script("plain.py", "import train_doctor\ntrain_doctor.step(samples=1)\nprint('ok')\n")
    import subprocess

    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"


def test_timeout_is_enforced(tmp_path, script):
    cmd = script("hang.py", "import time\ntime.sleep(60)\n")
    cfg = RunConfig(cmd=cmd, timeout=1.0)
    meta = run(cfg, tmp_path / "r")
    assert meta["timed_out"] is True
    assert meta["wall_s"] < 15


def test_compare_detects_clear_speedup(tmp_path, script):
    base = script("loop.py", MARKER_LOOP)
    res = api.compare([*base, "0.04"], [*base, "0.01"], repeats=3, warmup_runs=0, out_dir=tmp_path / "runs", **FAST)
    assert res["metric"] == "samples_per_s"
    assert res["verdict"] == "faster"
    assert res["ratio"]["low"] > 1.5  # nominally 4x; loose so a busy CI machine doesn't flake
    assert res["loss_check"]["status"] == "identical"
    assert res["decision"] == "keep"
    measured = [r for r in res["runs"] if r["phase"] == "measure"]
    assert len(measured) == 6
    root = tmp_path / "runs" / next((tmp_path / "runs").iterdir()).name
    assert (root / "compare.json").exists() and (root / "report.md").exists()
    assert "Decision: **keep**" in (root / "report.md").read_text()


def test_compare_same_command_is_not_called_faster(tmp_path, script):
    cmd = [*script("loop.py", MARKER_LOOP), "0.01"]
    res = api.compare(cmd, cmd, repeats=3, warmup_runs=0, min_effect=0.1, out_dir=tmp_path / "runs", **FAST)
    assert res["verdict"] == "no clear difference"
    assert res["decision"] == "inconclusive"
    assert res["loss_check"]["status"] == "identical"


def test_compare_flags_changed_loss(tmp_path, script):
    a = script("a.py", MARKER_LOOP)
    b = script("b.py", MARKER_LOOP.replace("loss=1.0 / (i + 1)", "loss=2.0 / (i + 1)").replace("delay = float", "delay = 0.5 * float"))
    res = api.compare([*a, "0.02"], [*b, "0.02"], repeats=2, warmup_runs=0, out_dir=tmp_path / "runs", **FAST)
    assert res["loss_check"]["status"] == "outside tolerance"
    assert res["decision"] != "keep"


def test_fake_py_spy(tmp_path, script, fake_bin):
    fake_bin(
        "py-spy",
        """
        out=""
        while [ $# -gt 0 ]; do
          if [ "$1" = "--output" ]; then out="$2"; fi
          shift
        done
        printf 'main (t.py:1);work (t.py:5) 9\\nmain (t.py:1);idle (t.py:9) 1\\n' > "$out"
    """,
    )
    cmd = script("loop.py", MARKER_LOOP)
    ev = api.profile(cmd, out_dir=tmp_path / "runs", profiler_steps=0, py_spy=True, **FAST)["evidence"]
    assert ev["py_spy"]["total_samples"] == 10
    assert ev["py_spy"]["own"][0]["frame"] == "work (t.py:5)"


def test_cli_end_to_end(tmp_path, script, capsys):
    cmd = script("loop.py", MARKER_LOOP)
    out = str(tmp_path / "runs")
    assert (
        main(["profile", "--out", out, "--warmup-steps", "2", "--steps", "5", "--seconds", "0", "--profiler-steps", "0", "--", *cmd]) == 0
    )
    text = capsys.readouterr().out
    assert "samples/s" in text and "report:" in text
    assert main(["diagnose", "--out", out, "--json"]) == 0
    assert "findings" in json.loads(capsys.readouterr().out)
    assert main(["report", "--out", out]) == 0
    assert "# train-doctor profile" in capsys.readouterr().out
    assert main(["profile"]) == 2


def test_home_is_redacted_in_shared_files(tmp_path, _isolated_home):
    from pathlib import Path

    proj = _isolated_home / "proj"
    proj.mkdir()
    (proj / "loop.py").write_text(MARKER_LOOP)
    res = api.profile([sys.executable, str(proj / "loop.py")], cwd=str(proj), out_dir=proj / "runs", profiler_steps=0, **FAST)
    run_dir = Path(res["run_dir"])
    home = str(_isolated_home)
    assert home in str(run_dir)
    for name in ("run.json", "evidence.json", "report.md", "report.json"):
        text = (run_dir / name).read_text()
        assert home not in text, name
    assert "~/proj/loop.py" in (run_dir / "run.json").read_text()


def test_script_that_ends_early_is_flagged(tmp_path, script):
    cmd = script("short.py", MARKER_LOOP.replace("range(1000)", "range(6)"))
    ev = api.profile(cmd, out_dir=tmp_path / "runs", profiler_steps=0, **FAST)["evidence"]
    assert ev["steps"]["end_reason"] == "process_exit"
    assert any("did not complete" in lim for lim in ev["limits"])


def test_missing_command_is_a_clear_error(tmp_path, capsys):
    out = str(tmp_path / "runs")
    assert main(["profile", "--out", out, "--", "python examples/nope.py"]) == 2
    err = capsys.readouterr().err
    assert "could not start" in err and "one argument" in err
    assert not list((tmp_path / "runs").iterdir())
    assert main(["compare", "--out", out, "--baseline", "no-such-binary-xyz", "--candidate", "no-such-binary-xyz"]) == 2
    assert "command not found" in capsys.readouterr().err
