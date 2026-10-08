"""Machine and provenance facts recorded with every run."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path

import psutil

from train_doctor import __version__
from train_doctor.accel import gpu_backend, query_apple_gpu, query_nvidia


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=3, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def cpu_name() -> str:
    if platform.system() == "Darwin":
        name = _sysctl("machdep.cpu.brand_string")
        if name:
            return name
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def machine_info() -> dict:
    vm = psutil.virtual_memory()
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "arch": platform.machine(),
        "cpu": cpu_name(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_bytes": int(vm.total),
        "tool_python": platform.python_version(),
        "train_doctor": __version__,
        "gpu_backend": gpu_backend(),
    }
    backend = info["gpu_backend"]
    if backend == "nvidia":
        gpus = query_nvidia() or []
        info["gpus"] = [{"name": g["name"], "mem_total_mib": g["mem_total_mib"]} for g in gpus]
    elif backend == "apple":
        g = query_apple_gpu()
        info["gpus"] = [{"name": g.get("name", "Apple GPU")}] if g else []
    return info


def file_sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def command_provenance(cmd: list[str], cwd: str) -> dict:
    """Hash every argument that names an existing file, and the git state of cwd."""
    files = {}
    for arg in cmd:
        p = Path(cwd, arg) if not os.path.isabs(arg) else Path(arg)
        if p.is_file() and p.stat().st_size < 50 * 1024 * 1024:
            files[arg] = file_sha256(p)
    prov: dict = {"cwd": cwd, "file_sha256": files}
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=False)
        if rev.returncode == 0:
            prov["git_commit"] = rev.stdout.strip()
            st = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5, check=False)
            prov["git_dirty"] = bool(st.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return prov
