"""Run a training command for a bounded window and collect raw evidence.

The command runs as a child process with the train-doctor hook directory
prepended to PYTHONPATH (see ``_hook/``). While it runs, a sampler thread
records CPU and memory of the whole process tree and GPU telemetry. Child
stdout and stderr are saved verbatim.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil

from train_doctor.accel import gpu_backend, sample_gpu
from train_doctor.redact import dump_json

HOOK_DIR = Path(__file__).parent / "_hook"

# Matches "loss: 0.123", "loss=1.2e-3", "train_loss 0.5", "'loss': 0.5" and similar.
LOSS_RE = re.compile(r"(?i)\bloss['\"]?\s*[:=]?\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)")


@dataclass
class RunConfig:
    cmd: list[str]
    cwd: str = "."
    warmup_steps: int = 5
    steps: int = 50
    seconds: float = 2.0
    timeout: float = 600.0
    profiler_steps: int = 0
    seed: int | None = None
    hook: bool = True
    stop_at_end: bool = True
    py_spy: bool = False
    sample_interval: float = 0.25
    gpu_interval: float = 1.0
    echo: bool = False
    env: dict = field(default_factory=dict)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


class _Sampler(threading.Thread):
    def __init__(self, pid: int, out: Path, interval: float, gpu_interval: float) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.out = out
        self.interval = interval
        self.gpu_interval = gpu_interval
        self.stop_evt = threading.Event()
        self.procs: dict[int, psutil.Process] = {}
        self.backend = gpu_backend()

    def _tree(self) -> list[psutil.Process]:
        try:
            root = psutil.Process(self.pid)
            procs = [root, *root.children(recursive=True)]
        except psutil.Error:
            return []
        live = []
        for p in procs:
            known = self.procs.get(p.pid)
            if known is None:
                try:
                    p.cpu_percent(None)  # prime
                except psutil.Error:
                    continue
                self.procs[p.pid] = p
                known = p
            live.append(known)
        return live

    def run(self) -> None:
        psutil.cpu_percent(None)
        last_gpu = 0.0
        with open(self.out, "w") as f:
            while not self.stop_evt.is_set():
                t = time.time()
                cpu = 0.0
                rss = 0
                rss_max = 0
                n = 0
                for p in self._tree():
                    try:
                        cpu += p.cpu_percent(None)
                        r = p.memory_info().rss
                    except psutil.Error:
                        continue
                    rss += r
                    rss_max = max(rss_max, r)
                    n += 1
                rec: dict = {
                    "wall": t,
                    "proc_cpu_pct": cpu,
                    "procs": n,
                    "tree_rss": rss,
                    "max_proc_rss": rss_max,
                    "sys_cpu_pct": psutil.cpu_percent(None),
                }
                if self.backend != "none" and t - last_gpu >= self.gpu_interval:
                    g = sample_gpu(self.backend)
                    if g:
                        rec["gpu"] = g
                    last_gpu = t
                f.write(json.dumps(rec) + "\n")
                f.flush()
                self.stop_evt.wait(self.interval)


def _reader(stream, path: Path, loss_path: Path | None, echo: bool, tag: str) -> None:
    with open(path, "w", encoding="utf-8", errors="replace") as f, open(loss_path, "w") if loss_path else open(os.devnull, "w") as lf:
        for raw in iter(stream.readline, b""):
            wall = time.time()
            line = raw.decode("utf-8", errors="replace")
            f.write(line)
            f.flush()
            if echo:
                os.write(2, f"[{tag}] {line}".encode())
            if loss_path:
                m = LOSS_RE.search(line)
                if m:
                    try:
                        lf.write(json.dumps({"wall": wall, "loss": float(m.group(1))}) + "\n")
                    except ValueError:
                        pass
    stream.close()


def build_env(cfg: RunConfig, run_dir: Path) -> dict:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg.env.items()})
    env["PYTHONUNBUFFERED"] = "1"
    if cfg.hook:
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(HOOK_DIR) + (os.pathsep + prev if prev else "")
        env["TRAIN_DOCTOR_RUN_DIR"] = str(run_dir)
        env["TRAIN_DOCTOR_WARMUP"] = str(cfg.warmup_steps)
        env["TRAIN_DOCTOR_STEPS"] = str(cfg.steps)
        env["TRAIN_DOCTOR_SECONDS"] = str(cfg.seconds)
        env["TRAIN_DOCTOR_PROFILER_STEPS"] = str(cfg.profiler_steps)
        env["TRAIN_DOCTOR_STOP"] = "1" if cfg.stop_at_end else "0"
        if cfg.seed is not None:
            env["TRAIN_DOCTOR_SEED"] = str(cfg.seed)
    return env


def _kill_tree(proc: subprocess.Popen, grace: float = 10.0) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()


def run(cfg: RunConfig, run_dir: Path) -> dict:
    """Run ``cfg.cmd`` once, writing raw evidence into ``run_dir``. Returns run metadata."""
    run_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(cfg, run_dir)
    started = _now_iso()
    try:
        load_1m = round(os.getloadavg()[0], 2)
    except OSError:
        load_1m = None
    t0 = time.time()
    proc = subprocess.Popen(
        cfg.cmd,
        cwd=cfg.cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    readers = [
        threading.Thread(
            target=_reader, args=(proc.stdout, run_dir / "stdout.log", run_dir / "log_losses.jsonl", cfg.echo, "out"), daemon=True
        ),
        threading.Thread(target=_reader, args=(proc.stderr, run_dir / "stderr.log", None, cfg.echo, "err"), daemon=True),
    ]
    for r in readers:
        r.start()
    sampler = _Sampler(proc.pid, run_dir / "samples.jsonl", cfg.sample_interval, cfg.gpu_interval)
    sampler.start()
    spy = _start_py_spy(proc.pid, run_dir) if cfg.py_spy else None

    timed_out = False
    try:
        proc.wait(timeout=cfg.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
    except KeyboardInterrupt:
        _kill_tree(proc, grace=3.0)
        raise
    finally:
        sampler.stop_evt.set()
        sampler.join(timeout=5)
        for r in readers:
            r.join(timeout=5)
    wall = time.time() - t0
    spy_info = _finish_py_spy(spy, run_dir) if spy else None

    meta = {
        "cmd": list(cfg.cmd),
        "cwd": os.path.abspath(cfg.cwd),
        "started_at": started,
        "start_wall": t0,
        "load_avg_1m": load_1m,
        "wall_s": round(wall, 3),
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "config": {k: v for k, v in asdict(cfg).items() if k not in ("cmd", "env")},
        "env_set": {k: v for k, v in env.items() if k.startswith("TRAIN_DOCTOR_")},
        "hook_dir": str(HOOK_DIR) if cfg.hook else None,
        "py_spy": spy_info,
    }
    dump_json(run_dir / "run.json", meta)
    return meta


# ----------------------------------------------------------------- py-spy
def _start_py_spy(pid: int, run_dir: Path) -> subprocess.Popen | None:
    exe = shutil.which("py-spy")
    if not exe:
        (run_dir / "pyspy.log").write_text("py-spy not found on PATH\n")
        return None
    log = open(run_dir / "pyspy.log", "w")  # noqa: SIM115 - closed in _finish_py_spy
    cmd = [
        exe,
        "record",
        "--pid",
        str(pid),
        "--subprocesses",
        "--format",
        "raw",
        "--rate",
        "100",
        "--output",
        str(run_dir / "pyspy.txt"),
        "--nonblocking",
    ]
    log.write(" ".join(cmd) + "\n")
    log.flush()
    try:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    except OSError as e:
        log.write(f"failed to start: {e}\n")
        log.close()
        return None
    p._td_log = log  # type: ignore[attr-defined]
    return p


def _finish_py_spy(p: subprocess.Popen, run_dir: Path) -> dict:
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        p.terminate()
        p.wait()
    p._td_log.close()  # type: ignore[attr-defined]
    out = run_dir / "pyspy.txt"
    return {"exit_code": p.returncode, "output": out.name if out.exists() else None}
