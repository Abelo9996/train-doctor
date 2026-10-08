"""Replace the home directory with ~ in files meant to be shared."""

from __future__ import annotations

import json
import os
from pathlib import Path


def home_prefix() -> str:
    return os.path.expanduser("~")


def redact_text(text: str) -> str:
    home = home_prefix()
    if not home or home == "/":
        return text
    return text.replace(home + os.sep, "~" + os.sep).replace(home, "~")


def dump_json(path: Path, data, indent: int = 2) -> None:
    Path(path).write_text(redact_text(json.dumps(data, indent=indent, default=str)))
