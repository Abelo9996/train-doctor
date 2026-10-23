"""Patch generation and reversible apply for proven changes."""

import ast

from train_doctor import patcher

SCRIPT = """\
import torch
from torch.utils.data import DataLoader, Dataset


class DS(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return torch.tensor([float(i)])


loader = DataLoader(DS(), batch_size=2, shuffle=True, num_workers=0)
"""


def _write(tmp_path, text=SCRIPT):
    p = tmp_path / "train.py"
    p.write_text(text)
    return p


def test_dataloader_kwargs_override_and_append(tmp_path):
    p = _write(tmp_path)
    info = patcher.make_patch(
        ["python", str(p)], ".", [{"kind": "dataloader_kwargs", "kwargs": {"num_workers": 5, "persistent_workers": True}}], tmp_path / "out"
    )
    assert info["patch_path"]
    new = info["_new_text"]
    assert "num_workers=5" in new and "persistent_workers=True" in new
    assert "num_workers=0" not in new  # the existing value was overridden, not duplicated
    ast.parse(new)  # still valid python


def test_cudnn_line_after_import(tmp_path):
    p = _write(tmp_path)
    info = patcher.make_patch(["python", str(p)], ".", [{"kind": "line_after_import_torch", "line": "torch.backends.cudnn.benchmark = True"}], tmp_path / "out")
    assert info["patch_path"]
    assert "torch.backends.cudnn.benchmark = True" in info["_new_text"]
    ast.parse(info["_new_text"])


def test_no_dataloader_gives_reason(tmp_path):
    p = tmp_path / "train.py"
    p.write_text("import torch\nx = torch.zeros(3)\n")
    info = patcher.make_patch(["python", str(p)], ".", [{"kind": "dataloader_kwargs", "kwargs": {"num_workers": 4}}], tmp_path / "out")
    assert info["patch_path"] is None
    assert "DataLoader" in info["reason"]


def test_apply_is_reversible(tmp_path):
    p = _write(tmp_path)
    original = p.read_text()
    info = patcher.make_patch(["python", str(p)], ".", [{"kind": "dataloader_kwargs", "kwargs": {"num_workers": 5, "persistent_workers": True}}], tmp_path / "out")
    applied = patcher.apply_patch(info)
    assert applied["applied"] is True
    assert "num_workers=5" in p.read_text()
    from pathlib import Path

    backup = Path(applied["backup"])
    assert backup.read_text() == original  # the original is preserved for reversal


def test_missing_script_reason(tmp_path):
    info = patcher.make_patch(["python", "does_not_exist.py"], str(tmp_path), [{"kind": "dataloader_kwargs", "kwargs": {"num_workers": 4}}], tmp_path / "out")
    assert info["patch_path"] is None and "find" in info["reason"]
