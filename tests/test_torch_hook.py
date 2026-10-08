"""Real PyTorch training loops under the hook (CPU only, skipped without torch)."""

import pytest

from train_doctor import api

pytest.importorskip("torch")
pytestmark = pytest.mark.torch

LOOP = """
import os, sys, torch
from torch.utils.data import DataLoader, Dataset

class DS(Dataset):
    def __len__(self):
        return 4096
    def __getitem__(self, i):
        x = torch.full((16,), float(i % 7))
        return x, x.sum() / 16

def main():
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    loader = DataLoader(DS(), batch_size=32, shuffle=False, num_workers=0)
    accum = int(os.environ.get("ACCUM", "1"))
    use_amp = os.environ.get("AMP") == "1"
    for epoch in range(100):
        for i, (x, y) in enumerate(loader):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_amp):
                loss = torch.nn.functional.mse_loss(model(x).squeeze(1).float(), y)
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
                print(f"loss {loss.item():.4f}")
                if os.environ.get("SAVE") == "1":
                    torch.save(model.state_dict(), os.devnull)

if __name__ == "__main__":
    main()
"""

FAST = {"warmup_steps": 2, "steps": 8, "seconds": 0.0}


def test_auto_hook_detects_steps_data_and_config(tmp_path, script):
    cmd = script("train.py", LOOP)
    res = api.profile(cmd, out_dir=tmp_path / "runs", profiler_steps=3, **FAST)
    ev = res["evidence"]
    assert ev["source"] == "hook", ev.get("limits")
    assert ev["steps"]["boundary"] == "optimizer"
    assert ev["steps"]["measured"] == 8 and ev["steps"]["profiled"] == 3
    assert ev["per_step"]["samples"] == 32
    assert ev["throughput"]["metric"] == "samples_per_s"
    cfg = ev["config"]
    assert cfg["device"] == "cpu"
    assert cfg["dataloaders"][0]["num_workers"] == 0 and cfg["dataloaders"][0]["batch_size"] == 32
    assert cfg["params"] == 17 and cfg["optimizer"] == "SGD"
    assert ev["data"]["getitem_measured"] is True
    assert ev["per_step"]["cpu_scalar_reads"] >= 1  # loss.item() on a CPU tensor
    assert ev["per_step"]["sync_calls"] == 0  # ...which is not a device sync
    assert len(ev["loss"]) == 13 and all(v > 0 for _, v in ev["loss"])
    assert ev["profiler"]["profiled_steps"] == 3
    assert ev["profiler"]["top_ops"]


def test_accumulation_amp_and_save_are_seen(tmp_path, script):
    cmd = script("train.py", LOOP)
    from train_doctor.evidence import write_evidence
    from train_doctor.runner import RunConfig, run

    cfg = RunConfig(cmd=cmd, warmup_steps=1, steps=4, seconds=0.0, env={"ACCUM": "2", "AMP": "1", "SAVE": "1"})
    run(cfg, tmp_path / "r")
    ev = write_evidence(tmp_path / "r")
    assert ev["per_step"]["backward_calls"] == 2
    assert ev["per_step"]["samples"] == 64
    assert ev["per_step"]["save_calls"] == 1
    assert ["cpu", "bfloat16", True] in ev["config"]["autocast"]
