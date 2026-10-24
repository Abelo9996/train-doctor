"""Turn a proven change into a reversible source patch.

``fix`` proves a change by injecting settings at run time, which touches no
source. To hand the user something durable it also translates that change into a
unified diff against their script: DataLoader keyword arguments, or a line after
``import torch``. The diff applies and reverses with ``git apply`` / ``patch``.
Patching is best effort and conservative: if the target can't be found, parsed,
or edited unambiguously, no patch is written and the exact edit is described in
words instead.
"""

from __future__ import annotations

import ast
import datetime as _dt
import difflib
from pathlib import Path


def _target_file(argv: list[str], cwd: str) -> Path | None:
    for a in argv:
        if a.endswith(".py"):
            p = Path(cwd, a)
            if p.is_file():
                return p
    return None


def _is_dataloader(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "DataLoader"
    if isinstance(func, ast.Attribute):
        return func.attr == "DataLoader"
    return False


def _offset(lines: list[str], lineno: int, col: int) -> int:
    return sum(len(x) for x in lines[: lineno - 1]) + col


def _lit(val) -> str:
    return repr(val)


def _apply_dataloader_kwargs(text: str, kwargs: dict) -> tuple[str, str | None]:
    """Add or override keyword arguments on the first DataLoader(...) call. Returns (new_text, error)."""
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return text, f"could not parse the script ({e})"
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _is_dataloader(n.func)]
    if not calls:
        return text, "no DataLoader(...) call found in the script"
    call = calls[0]
    lines = text.splitlines(keepends=True)
    existing = {k.arg: k for k in call.keywords if k.arg}
    edits: list[tuple[int, int, str]] = []
    append: list[str] = []
    for key, val in kwargs.items():
        if key in existing:
            v = existing[key].value
            edits.append((_offset(lines, v.lineno, v.col_offset), _offset(lines, v.end_lineno, v.end_col_offset), _lit(val)))
        else:
            append.append(f", {key}={_lit(val)}")
    if append:
        nodes = list(call.args) + [k.value for k in call.keywords]
        if not nodes:
            return text, "DataLoader(...) has no arguments to extend"
        last = max(_offset(lines, n.end_lineno, n.end_col_offset) for n in nodes)
        edits.append((last, last, "".join(append)))
    for s, e, r in sorted(edits, key=lambda x: x[0], reverse=True):
        text = text[:s] + r + text[e:]
    return text, None


def _apply_import_line(text: str, line: str) -> tuple[str, str | None]:
    """Insert ``line`` right after the module-level ``import torch`` (once)."""
    if line in text:
        return text, "already present"
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return text, f"could not parse the script ({e})"
    target_lineno = None
    for n in tree.body:
        if isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names):
            target_lineno = n.end_lineno
            break
    if target_lineno is None:
        return text, "no top-level `import torch` found"
    lines = text.splitlines(keepends=True)
    indent = ""
    lines.insert(target_lineno, indent + line + "\n")
    return "".join(lines), None


def apply_edits_to_text(text: str, edits: list[dict]) -> tuple[str, list[str]]:
    """Apply the structured ``edits`` to source ``text`` in order. Returns ``(text, notes)``."""
    notes: list[str] = []
    for e in edits:
        if e["kind"] == "dataloader_kwargs":
            text, err = _apply_dataloader_kwargs(text, e["kwargs"])
        elif e["kind"] == "line_after_import_torch":
            text, err = _apply_import_line(text, e["line"])
            if err == "already present":
                err = None
        else:
            err = f"unknown edit kind {e['kind']}"
        if err:
            notes.append(err)
    return text, notes


def diff_text(target: Path, original_text: str, new_text: str, out_dir: Path) -> dict:
    """Write a reversible unified diff from ``original_text`` to ``new_text`` for ``target``."""
    target = Path(target)
    if new_text == original_text:
        return {"patch_path": None, "reason": "nothing to change", "target": str(target)}
    rel = target.name
    diff = "".join(
        difflib.unified_diff(original_text.splitlines(keepends=True), new_text.splitlines(keepends=True), fromfile=f"a/{rel}", tofile=f"b/{rel}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_path = out_dir / f"{target.stem}.fix.patch"
    patch_path.write_text(diff)
    return {"patch_path": str(patch_path), "target": str(target), "_new_text": new_text}


def make_patch(argv: list[str], cwd: str, edits: list[dict], out_dir: Path) -> dict:
    """Produce a reversible unified diff for ``edits`` against the command's script.

    Returns a dict with ``patch_path`` and ``target`` on success, or ``reason`` when no
    patch could be generated. ``_new_text`` (the patched file contents) is included for
    ``apply_patch`` and is not meant to be serialized.
    """
    path = _target_file(argv, cwd)
    if path is None:
        return {"patch_path": None, "reason": "could not find a .py script in the command to patch"}
    original = path.read_text()
    text = original
    notes: list[str] = []
    for e in edits:
        if e["kind"] == "dataloader_kwargs":
            text, err = _apply_dataloader_kwargs(text, e["kwargs"])
        elif e["kind"] == "line_after_import_torch":
            text, err = _apply_import_line(text, e["line"])
            if err == "already present":
                err = None
        else:
            err = f"unknown edit kind {e['kind']}"
        if err:
            notes.append(err)
    if text == original:
        return {"patch_path": None, "reason": "; ".join(notes) or "nothing to change", "target": str(path)}
    rel = path.name
    diff = "".join(
        difflib.unified_diff(original.splitlines(keepends=True), text.splitlines(keepends=True), fromfile=f"a/{rel}", tofile=f"b/{rel}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_path = out_dir / f"{path.stem}.fix.patch"
    patch_path.write_text(diff)
    return {"patch_path": str(patch_path), "target": str(path), "notes": notes, "_new_text": text}


def apply_patch(patch_info: dict) -> dict:
    """Write the patched contents to the target, backing up the original first. Reversible via the backup or `git apply -R`."""
    target = patch_info.get("target")
    new_text = patch_info.get("_new_text")
    if not target or new_text is None:
        return {"applied": False, "reason": "no target or patched text"}
    p = Path(target)
    stamp = _dt.datetime.now().strftime("%Y%m%d%H%M%S")
    backup = p.with_name(f"{p.name}.train-doctor-backup-{stamp}")
    backup.write_text(p.read_text())
    p.write_text(new_text)
    return {"applied": True, "target": target, "backup": str(backup)}
