from train_doctor.lossdiff import diff


def ev(losses, samples=32):
    return {"loss": [[i, v] for i, v in enumerate(losses)], "samples_by_step": [samples] * len(losses)}


def test_identical_prefix_counts_as_identical():
    a = ev([2.0 - 0.01 * i for i in range(40)])
    b = ev([2.0 - 0.01 * i for i in range(55)])  # ran longer in the same time window
    assert diff(a, b, 0.05)["status"] == "identical"


def test_small_noise_is_within_tolerance():
    base = [2.0 - 0.02 * i for i in range(50)]
    cand = [v * (1.02 if i % 2 else 0.98) for i, v in enumerate(base)]
    out = diff(ev(base), ev(cand), 0.05)
    assert out["status"] == "within tolerance"
    assert out["mean_rel_diff"] < 0.02


def test_shifted_curve_is_outside_tolerance():
    base = [2.0 - 0.02 * i for i in range(50)]
    cand = [v * 1.3 for v in base]
    assert diff(ev(base), ev(cand), 0.05)["status"] == "outside tolerance"


def test_alignment_on_samples_for_batch_size_change():
    # candidate uses twice the batch: its step i covers the samples of baseline steps 2i and 2i+1
    base = [3.0 - 0.01 * i for i in range(80)]
    cand = [3.0 - 0.01 * (2 * i + 1) for i in range(40)]
    out = diff(ev(base, 32), ev(cand, 64), 0.05)
    assert out["axis"] == "samples"
    assert out["mean_rel_diff"] < 0.01


def test_missing_loss_is_unavailable():
    assert diff({"loss": []}, ev([1.0, 0.9]), 0.05)["status"] == "unavailable"
