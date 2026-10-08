"""The recorder's CUDA paths, driven by a fake torch module (no GPU, no torch needed)."""

import json
import sys
import types

import pytest

from train_doctor._hook import td_hook


class FakeCuda:
    def __init__(self):
        self.syncs = 0

    def is_available(self):
        return True

    def synchronize(self):
        self.syncs += 1

    def memory_allocated(self):
        return 2 << 30

    def max_memory_allocated(self):
        return 3 << 30

    def memory_reserved(self):
        return 4 << 30

    def get_device_properties(self, idx):
        return types.SimpleNamespace(total_memory=16 << 30, name="Fake A100", major=8, minor=0)

    def current_device(self):
        return 0


@pytest.fixture
def fake_torch(monkeypatch):
    t = types.ModuleType("torch")
    t.cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", t)
    return t


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_DOCTOR_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("TRAIN_DOCTOR_WARMUP", "2")
    monkeypatch.setenv("TRAIN_DOCTOR_STEPS", "3")
    monkeypatch.setenv("TRAIN_DOCTOR_SECONDS", "0")
    monkeypatch.setenv("TRAIN_DOCTOR_STOP", "1")
    r = td_hook.Recorder()
    monkeypatch.setattr(td_hook, "R", r)
    return r


def events(tmp_path):
    (f,) = tmp_path.glob("events.*.jsonl")
    return [json.loads(x) for x in f.read_text().splitlines()]


def test_cuda_window_syncs_and_memory(tmp_path, fake_torch, recorder):
    recorder.device = "cuda"
    with pytest.raises(SystemExit):
        for _ in range(10):
            recorder.boundary(samples=4, source="optimizer")
    evs = events(tmp_path)
    kinds = [e["type"] for e in evs]
    assert kinds.count("step") == 5  # 2 warmup + 3 measured, then the run stops
    assert "window_start" in kinds and "window_end" in kinds and kinds[-1] == "end"
    # device synced exactly at window start and window end
    assert fake_torch.cuda.syncs == 2
    wend = next(e for e in evs if e["type"] == "window_end")
    assert wend["mem"]["peak_allocated"] == 3 << 30 and wend["mem"]["total"] == 16 << 30
    first = next(e for e in evs if e["type"] == "step")
    assert first["mem"]["allocated"] == 2 << 30


def test_explicit_sync_calls_are_counted(fake_torch, recorder):
    td_hook._wrap_module_sync(fake_torch.cuda)
    fake_torch.cuda.synchronize()
    fake_torch.cuda.synchronize()
    assert recorder.c_sync_calls == 2
    # train-doctor's own syncs are not counted
    recorder.device = "cuda"
    recorder.sync_device()
    assert recorder.c_sync_calls == 2


def test_batch_len_shapes():
    T = types.SimpleNamespace
    assert td_hook._batch_len(T(shape=(32, 3))) == 32
    assert td_hook._batch_len((T(shape=(16,)), T(shape=(16,)))) == 16
    assert td_hook._batch_len({"input_ids": T(shape=(8, 128))}) == 8
    assert td_hook._batch_len([1, 2, 3]) is None
