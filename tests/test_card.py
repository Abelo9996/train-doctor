"""The result card: SVG always, PNG when resvg-py is installed."""

from pathlib import Path

from train_doctor import card

RESULT = {
    "metric": "samples_per_s",
    "ratio": {"point": 2.31, "low": 1.9, "high": 2.7, "n_pairs": 5},
    "throughput": {"baseline": {"median": 2100.0}, "candidate": {"median": 4851.0}},
    "loss_check": {"status": "within tolerance"},
}


def test_svg_always_written(tmp_path):
    out = card.render_card("mnist_cnn", RESULT, {"gpu_backend": "apple", "cpu": "Apple M4"}, ["python", "train.py"], tmp_path)
    svg = Path(out["svg"]).read_text()
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "2.31x faster" in svg
    assert "loss unchanged" in svg
    assert "check yours: uvx train-doctor fix" in svg


def test_escaping_in_title(tmp_path):
    out = card.render_card("a<b>&c", RESULT, {"gpu_backend": "none", "cpu": "CPU"}, ["python", "train.py"], tmp_path)
    svg = Path(out["svg"]).read_text()
    assert "<b>" not in svg and "&amp;c" in svg


def test_png_rendered_when_available(tmp_path):
    import importlib.util

    out = card.render_card("t", RESULT, {"gpu_backend": "apple"}, ["python", "train.py"], tmp_path)
    if importlib.util.find_spec("resvg_py") is not None:
        assert out["png"] and Path(out["png"]).read_bytes()[:4] == b"\x89PNG"
    else:
        assert out["png"] is None and "png_note" in out
