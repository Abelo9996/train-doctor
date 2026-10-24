"""Candidate generation and the fix orchestration, with profile/compare mocked.

The real profile and compare run subprocesses; here they are replaced with canned
results so the candidate mapping, winner selection, combination and patch/card
wiring can be checked offline and fast.
"""

import json
from pathlib import Path

from train_doctor import autofix


def _finding(fid, risk="low", numbers=None, title="t", suggestion="s"):
    return {"id": fid, "risk": risk, "numbers": numbers or {}, "title": title, "suggestion": suggestion}


def test_candidate_from_dataloader_bound_mps():
    ev = {"config": {"device": "mps"}}
    cands, sugg = autofix.candidates_from_findings(ev, [_finding("dataloader_bound", numbers={"suggested_workers": 6})], 4)
    assert len(cands) == 1
    c = cands[0]
    assert c.id == "dataloader_workers"
    assert c.env == {"TRAIN_DOCTOR_FIX_NUM_WORKERS": "6"}  # num_workers only: loss-neutral
    assert c.edits[0]["kwargs"] == {"num_workers": 6}
    assert [s["id"] for s in sugg] == ["persistent_workers"]  # offered separately, not auto-applied


def test_candidate_dataloader_cuda_adds_pin_memory():
    ev = {"config": {"device": "cuda"}}
    cands, _ = autofix.candidates_from_findings(ev, [_finding("dataloader_bound", numbers={"suggested_workers": 4})], 4)
    assert cands[0].env["TRAIN_DOCTOR_FIX_PIN_MEMORY"] == "1"
    assert cands[0].edits[0]["kwargs"]["pin_memory"] is True
    assert "TRAIN_DOCTOR_FIX_PERSISTENT_WORKERS" not in cands[0].env


def test_cudnn_and_unmapped_findings():
    ev = {"config": {"device": "cuda"}}
    findings = [_finding("cudnn_benchmark_off"), _finding("no_mixed_precision", risk="medium")]
    cands, sugg = autofix.candidates_from_findings(ev, findings, 4)
    assert [c.id for c in cands] == ["cudnn_benchmark"]
    assert [s["id"] for s in sugg] == ["no_mixed_precision"]


def test_budget_caps_candidates():
    ev = {"config": {"device": "cuda"}}
    findings = [_finding("dataloader_bound", numbers={"suggested_workers": 4}), _finding("cudnn_benchmark_off")]
    cands, _ = autofix.candidates_from_findings(ev, findings, 1)
    assert len(cands) == 1 and cands[0].id == "dataloader_workers"


def _patch_api(monkeypatch, tmp_path, findings, compare_fn):
    def fake_profile(argv, **kw):
        return {
            "run_dir": str(tmp_path / "prof"),
            "evidence": {"config": {"device": "mps"}, "machine": {"gpu_backend": "apple", "cpu": "Apple M4"}},
            "findings": findings,
            "next_step": "n/a",
        }

    monkeypatch.setattr(autofix.api, "profile", fake_profile)
    monkeypatch.setattr(autofix.api, "compare", compare_fn)


def _compare_result(candidate_env, ratio, decision="keep", status="identical"):
    return {
        "decision": decision,
        "verdict": "faster" if decision == "keep" else "no clear difference",
        "reason": f"{decision} {ratio}",
        "ratio": {"point": ratio, "low": ratio - 0.1, "high": ratio + 0.1, "n_pairs": 5},
        "loss_check": {"status": status, "mean_rel_diff": 0.0, "final_rel_diff": 0.0},
        "throughput": {"baseline": {"median": 1000.0}, "candidate": {"median": 1000.0 * ratio}},
        "metric": "samples_per_s",
        "root": "/tmp/cmp",
    }


def _script(tmp_path):
    p = tmp_path / "mymodel.py"  # non-generic stem, so the label is the file name, not the parent dir
    p.write_text("import torch\nfrom torch.utils.data import DataLoader\nloader = DataLoader([1, 2], batch_size=1, num_workers=0)\n")
    return p


def test_fix_picks_best_and_combines(monkeypatch, tmp_path):
    script = _script(tmp_path)
    findings = [_finding("dataloader_bound", numbers={"suggested_workers": 5}), _finding("cudnn_benchmark_off")]

    def compare(baseline, candidate, **kw):
        env = kw.get("candidate_env") or {}
        if "TRAIN_DOCTOR_FIX_NUM_WORKERS" in env and "TRAIN_DOCTOR_FIX_CUDNN_BENCHMARK" in env:
            return _compare_result(env, 1.8)  # combined
        if "TRAIN_DOCTOR_FIX_NUM_WORKERS" in env:
            return _compare_result(env, 1.5)
        return _compare_result(env, 1.1)

    _patch_api(monkeypatch, tmp_path, findings, compare)
    out = autofix.fix(["python", str(script)], out_dir=str(tmp_path / "out"), make_card=True, cwd=str(tmp_path))
    assert out["proven"]["candidate"] == "combined"
    assert "1.80x faster" in out["headline"]
    assert out["patch"]["patch_path"] and "num_workers=5" in Path(out["patch"]["patch_path"]).read_text()
    assert out["card"]["svg"]
    # fix.json was written and is valid
    assert json.loads(Path(out["fix_dir"] + "/fix.json").read_text())["label"] == "mymodel"


def test_fix_no_winner_is_honest(monkeypatch, tmp_path):
    script = _script(tmp_path)
    findings = [_finding("dataloader_bound", numbers={"suggested_workers": 5})]
    _patch_api(monkeypatch, tmp_path, findings, lambda b, c, **kw: _compare_result(kw.get("candidate_env"), 1.02, decision="inconclusive"))
    out = autofix.fix(["python", str(script)], out_dir=str(tmp_path / "out"), make_card=True, cwd=str(tmp_path))
    assert out["proven"] is None
    assert "no proven win" in out["headline"]
    assert "card" not in out


def test_agent_candidates_from_findings():
    ev = {"config": {"device": "mps"}}
    findings = [
        _finding("no_mixed_precision", risk="medium"),
        _finding("no_torch_compile"),
        _finding("host_sync_in_loop", risk="none"),
        _finding("cpu_with_accelerator_available", numbers={"available": "mps"}),
    ]
    cands = autofix.agent_candidates_from_findings(ev, findings, limit=5)
    ids = [c.id for c in cands]
    # same-math changes first, numerics-changing ones last
    assert ids == ["remove_sync", "torch_compile", "amp", "device_move"]
    byid = {c.id: c for c in cands}
    assert "autocast(device_type='mps', dtype=torch.float16)" in byid["amp"].instruction
    assert "torch.compile(model)" in byid["torch_compile"].instruction
    assert "mps" in byid["device_move"].instruction


def test_fix_plan_mode_spends_nothing(monkeypatch, tmp_path):
    script = _script(tmp_path)
    findings = [_finding("no_mixed_precision", risk="medium"), _finding("no_torch_compile")]

    def boom(*a, **k):
        raise AssertionError("must not run compare or the agent in plan mode")

    monkeypatch.setattr(autofix.agentdrv, "edit_file", boom)
    _patch_api(monkeypatch, tmp_path, findings, boom)
    out = autofix.fix(["python", str(script)], out_dir=str(tmp_path / "out"), agent="claude", yes=False, make_card=False, cwd=str(tmp_path))
    assert out["mode"] == "plan"
    assert out["proven"] is None
    assert [a["id"] for a in out["agent_plan"]["agent_candidates"]] == ["torch_compile", "amp"]
    assert "plan only" in out["headline"] and "Cost" in out["cost_note"]


def test_fix_agent_stack_keeps_proven_source_edit(monkeypatch, tmp_path):
    script = _script(tmp_path)
    original = script.read_text()
    findings = [_finding("no_torch_compile")]

    def fake_edit(agent, scratch, instruction, cwd, timeout=300.0):
        from pathlib import Path as _P

        p = _P(scratch)
        p.write_text(p.read_text() + "\n# compiled\nmodel = None\n")
        return {"ok": True, "changed": True, "valid": True, "returncode": 0}

    monkeypatch.setattr(autofix.agentdrv, "edit_file", fake_edit)
    # every compare (agent candidate and the final confirm) comes back as a keep
    _patch_api(monkeypatch, tmp_path, findings, lambda b, c, **kw: _compare_result(kw.get("candidate_env"), 1.6))
    out = autofix.fix(["python", str(script)], out_dir=str(tmp_path / "out"), agent="claude", yes=True, make_card=True, cwd=str(tmp_path))
    assert out["agent"] == "claude"
    assert out["proven"] and out["proven"]["candidate"] == "combined"
    assert "1.60x faster" in out["headline"]
    assert out["patch"]["patch_path"] and "# compiled" in Path(out["patch"]["patch_path"]).read_text()
    # scratch files are cleaned up; the original script is untouched without --keep
    assert script.read_text() == original
    assert not list(tmp_path.glob("*.td-*.py"))


def test_fix_keep_applies_patch(monkeypatch, tmp_path):
    script = _script(tmp_path)
    original = script.read_text()
    findings = [_finding("dataloader_bound", numbers={"suggested_workers": 5})]
    _patch_api(monkeypatch, tmp_path, findings, lambda b, c, **kw: _compare_result(kw.get("candidate_env"), 1.5))
    out = autofix.fix(["python", str(script)], out_dir=str(tmp_path / "out"), keep=True, make_card=False, cwd=str(tmp_path))
    assert out["applied"]["applied"] is True
    assert "num_workers=5" in script.read_text()
    assert original != script.read_text()
