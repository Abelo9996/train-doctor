"""`train-doctor setup`: register the MCP server and install the skill for coding agents.

Shows every change first. Applies only with ``--yes`` (or an interactive yes),
backs up any file it edits, and does nothing for entries that are already in
place, so running it twice is safe.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

NAME = "train-doctor"
DEFAULT_SERVER = ["uvx", "train-doctor", "mcp"]


@dataclass
class Action:
    target: str
    description: str
    apply: object  # callable or None when nothing to do

    @property
    def needed(self) -> bool:
        return self.apply is not None


def skill_text() -> str:
    try:
        from importlib.resources import files

        p = files("train_doctor").joinpath("skill/SKILL.md")
        if p.is_file():
            return p.read_text()
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    repo = Path(__file__).resolve().parents[2] / "skills" / NAME / "SKILL.md"
    return repo.read_text()


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d%H%M%S")


def _backup(path: Path) -> Path | None:
    if path.exists():
        b = path.with_name(f"{path.name}.train-doctor-backup-{_stamp()}")
        shutil.copy2(path, b)
        return b
    return None


def _write(path: Path, text: str) -> str:
    b = _backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return f"wrote {path}" + (f" (backup: {b})" if b else "")


def _json_server_action(target: str, path: Path, server: list[str]) -> Action:
    entry = {"command": server[0], "args": server[1:]}
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text() or "{}")
        except json.JSONDecodeError:
            return Action(target, f"{path} is not valid JSON; add this under mcpServers by hand: {json.dumps({NAME: entry})}", None)
    if (data.get("mcpServers") or {}).get(NAME) == entry:
        return Action(target, f"{path} already has the {NAME} server", None)
    new = dict(data)
    new["mcpServers"] = {**(data.get("mcpServers") or {}), NAME: entry}
    verb = "update" if path.exists() else "create"
    return Action(target, f"{verb} {path}: mcpServers.{NAME} = {json.dumps(entry)}", lambda: _write(path, json.dumps(new, indent=2) + "\n"))


def _toml_quote(s: str) -> str:
    return json.dumps(s)  # JSON string escaping is valid TOML basic-string escaping for these values


def _codex_action(home: Path, server: list[str]) -> Action:
    path = home / ".codex" / "config.toml"
    text = path.read_text() if path.exists() else ""
    header = f"[mcp_servers.{NAME}]"
    if header in text or f'[mcp_servers."{NAME}"]' in text:
        return Action("Codex", f"{path} already has {header}", None)
    block = f"\n{header}\ncommand = {_toml_quote(server[0])}\nargs = [{', '.join(_toml_quote(a) for a in server[1:])}]\n"
    verb = "append to" if path.exists() else "create"
    return Action("Codex", f"{verb} {path}:{block.rstrip()}", lambda: _write(path, text + block))


def _claude_action(server: list[str]) -> Action:
    exe = shutil.which("claude")
    if not exe:
        return Action("Claude Code", "the `claude` CLI is not on PATH; use --project DIR to write a project .mcp.json instead", None)
    try:
        got = subprocess.run([exe, "mcp", "get", NAME], capture_output=True, text=True, timeout=30, check=False)
        if got.returncode == 0:
            return Action("Claude Code", f"`claude mcp get {NAME}` shows it is already registered", None)
    except (OSError, subprocess.TimeoutExpired):
        pass
    cmd = [exe, "mcp", "add", "--scope", "user", NAME, "--", *server]

    def run() -> str:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        if out.returncode != 0:
            raise RuntimeError(f"`{' '.join(cmd)}` failed: {out.stderr.strip() or out.stdout.strip()}")
        return "ran " + " ".join(cmd)

    return Action("Claude Code", "run: claude mcp add --scope user " + NAME + " -- " + " ".join(server), run)


def _skill_action(target: str, dest: Path, text: str) -> Action:
    if dest.exists() and dest.read_text() == text:
        return Action(target, f"{dest} is up to date", None)
    verb = "update" if dest.exists() else "create"
    return Action(target, f"{verb} {dest}", lambda: _write(dest, text))


def plan(home: Path | None = None, project: str | None = None, server: list[str] | None = None) -> list[Action]:
    home = home or Path.home()
    server = server or DEFAULT_SERVER
    actions: list[Action] = []
    if shutil.which("claude") or (home / ".claude").exists():
        actions.append(_claude_action(server))
    if shutil.which("codex") or (home / ".codex").exists():
        actions.append(_codex_action(home, server))
    if (home / ".cursor").exists():
        actions.append(_json_server_action("Cursor", home / ".cursor" / "mcp.json", server))
    if project:
        actions.append(_json_server_action("Project", Path(project) / ".mcp.json", server))
    text = skill_text()
    if (home / ".claude").exists():
        actions.append(_skill_action("Claude Code skill", home / ".claude" / "skills" / NAME / "SKILL.md", text))
    if (home / ".codex").exists():
        actions.append(_skill_action("Codex skill", home / ".codex" / "skills" / NAME / "SKILL.md", text))
    return actions


def run_setup(apply: bool = False, project: str | None = None, server_command: str | None = None, home: Path | None = None) -> int:
    import shlex

    server = shlex.split(server_command) if server_command else DEFAULT_SERVER
    actions = plan(home=home, project=project, server=server)
    if not actions:
        print("No Claude Code, Codex or Cursor installation found. Use --project DIR to write a project .mcp.json.")
        return 0
    print("train-doctor setup plan:")
    for a in actions:
        print(f"  [{'change' if a.needed else 'ok'}] {a.target}: {a.description}")
    todo = [a for a in actions if a.needed]
    if not todo:
        print("Nothing to change.")
        return 0
    if not apply:
        if sys.stdin.isatty():
            ans = input("Apply these changes? [y/N] ").strip().lower()
            apply = ans in ("y", "yes")
        if not apply:
            print("Dry run: nothing changed. Re-run with --yes to apply.")
            return 0
    rc = 0
    for a in todo:
        try:
            print(f"  {a.target}: {a.apply()}")  # type: ignore[operator]
        except Exception as e:
            print(f"  {a.target}: FAILED: {e}")
            rc = 1
    if rc == 0:
        print(
            "Done. Start a new agent session in your training project and ask, for example:\n"
            '  "train.py is slow. Use train-doctor to find out why, fix it, and prove the speedup."'
        )
    return rc
