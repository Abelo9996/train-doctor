"""Accelerator and machine telemetry from the tools the OS already ships.

* NVIDIA: ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``
* Apple GPU: ``ioreg -r -d 1 -w 0 -c IOAccelerator`` (``Device Utilization %``,
  system-wide, no sudo needed)
* CPU and memory of the process tree: psutil
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess

NVIDIA_FIELDS = ["index", "name", "utilization.gpu", "memory.used", "memory.total", "power.draw"]


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _num(s: str) -> float | None:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_nvidia_smi(text: str) -> list[dict]:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(NVIDIA_FIELDS):
            continue
        gpus.append(
            {
                "index": int(_num(parts[0]) or 0),
                "name": parts[1],
                "util_pct": _num(parts[2]),
                "mem_used_mib": _num(parts[3]),
                "mem_total_mib": _num(parts[4]),
                "power_w": _num(parts[5]),
            }
        )
    return gpus


def query_nvidia() -> list[dict] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    text = _run([exe, "--query-gpu=" + ",".join(NVIDIA_FIELDS), "--format=csv,noheader,nounits"])
    if text is None:
        return None
    return parse_nvidia_smi(text)


_IOREG_UTIL = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
_IOREG_MODEL = re.compile(r'"model"\s*=\s*"([^"]+)"')


def parse_ioreg(text: str) -> dict | None:
    m = _IOREG_UTIL.search(text)
    if not m:
        return None
    out: dict = {"util_pct": float(m.group(1))}
    mm = _IOREG_MODEL.search(text)
    if mm:
        out["name"] = mm.group(1)
    return out


def query_apple_gpu() -> dict | None:
    if platform.system() != "Darwin":
        return None
    exe = shutil.which("ioreg") or "/usr/sbin/ioreg"
    text = _run([exe, "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
    if text is None:
        return None
    return parse_ioreg(text)


def gpu_backend() -> str:
    """Which telemetry source applies on this machine: 'nvidia', 'apple' or 'none'."""
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple"
    return "none"


def sample_gpu(backend: str) -> dict | None:
    if backend == "nvidia":
        gpus = query_nvidia()
        return {"backend": "nvidia", "gpus": gpus} if gpus else None
    if backend == "apple":
        g = query_apple_gpu()
        return {"backend": "apple", "gpus": [g]} if g else None
    return None
