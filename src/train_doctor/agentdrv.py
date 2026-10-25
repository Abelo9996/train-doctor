"""Drive a coding-agent CLI headlessly to make one source edit, for ``fix --agent``.

Some of the biggest speedups the rules find (mixed precision, torch.compile, a
device move, removing per-step syncs) need a change to the training source that
``fix`` will not make blindly. ``fix --agent claude`` (or ``codex``) asks the
user's own agent CLI to make exactly one such change in a scratch copy of the
script, then proves it with the same paired ``compare`` and loss check. The agent
writes the code; ``compare`` is the guardrail that keeps only the changes that are
actually faster without moving the loss.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

AGENTS = ("claude", "codex")


def available(agent: str) -> bool:
    return shutil.which(agent) is not None


def _prompt(file_name: str, instruction: str) -> str:
    return (
        f"Edit the file `{file_name}` in the current directory to make exactly ONE change: {instruction}\n\n"
        "Rules: change nothing else; keep the training loop and its outputs intact; do not add explanatory comments, "
        "do not create or delete files, and do not run anything. Make the edit and stop."
    )


def _cmd(agent: str, prompt: str) -> list[str]:
    if agent == "claude":
        return ["claude", "-p", prompt, "--permission-mode", "acceptEdits", "--allowedTools", "Edit,Read,Write", "--max-turns", "8"]
    if agent == "codex":
        return ["codex", "exec", "--full-auto", prompt]
    raise ValueError(f"unknown agent {agent!r}")


def edit_file(agent: str, scratch: Path, instruction: str, cwd: str, timeout: float = 300.0) -> dict:
    """Ask ``agent`` to make one edit to ``scratch`` (a file inside ``cwd``). Returns what happened."""
    before = scratch.read_text()
    try:
        out = subprocess.run(
            _cmd(agent, _prompt(scratch.name, instruction)), cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "changed": False, "error": f"the agent did not finish within {timeout:.0f}s"}
    except OSError as e:
        return {"ok": False, "changed": False, "error": str(e)}
    after = scratch.read_text()
    changed = after != before
    valid = True
    if changed:
        try:
            compile(after, str(scratch), "exec")
        except SyntaxError as e:
            valid = False
            scratch.write_text(before)  # roll back an edit that does not parse
            return {"ok": False, "changed": False, "error": f"the agent left invalid Python: {e}", "reverted": True}
    return {
        "ok": out.returncode == 0,
        "changed": changed,
        "valid": valid,
        "returncode": out.returncode,
        "stdout_tail": (out.stdout or "")[-800:],
        "stderr_tail": (out.stderr or "")[-400:],
    }
