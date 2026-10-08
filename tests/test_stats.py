import math

from train_doctor.stats import bootstrap_ratio_ci, describe, mann_whitney_u, median, moving_average, quantile, verdict


def test_quantiles_match_numpy_linear():
    xs = [1, 2, 3, 4, 10]
    assert median(xs) == 3
    assert quantile(xs, 0.25) == 2
    assert math.isclose(quantile(xs, 0.9), 4 + 0.6 * 6)


def test_describe():
    d = describe([10.0, 12.0, 11.0, 13.0])
    assert d["n"] == 4 and d["median"] == 11.5 and d["min"] == 10 and d["max"] == 13
    assert math.isclose(d["iqr"], d["p75"] - d["p25"])


def test_bootstrap_clear_speedup():
    base = [100, 101, 99, 100, 102]
    cand = [200, 199, 205, 201, 198]
    point, lo, hi = bootstrap_ratio_ci(base, cand)
    assert math.isclose(point, 2.0)
    assert 1.9 < lo <= point <= hi < 2.1
    assert verdict(lo, hi, 0.02) == "faster"


def test_bootstrap_is_deterministic():
    a = bootstrap_ratio_ci([1.0, 1.1, 0.9], [1.2, 1.0, 1.1])
    b = bootstrap_ratio_ci([1.0, 1.1, 0.9], [1.2, 1.0, 1.1])
    assert a == b


def test_overlapping_runs_are_no_clear_difference():
    base = [100, 104, 97, 102, 99]
    cand = [101, 98, 105, 100, 103]
    _, lo, hi = bootstrap_ratio_ci(base, cand)
    assert lo < 1 < hi
    assert verdict(lo, hi, 0.02) == "no clear difference"


def test_verdict_requires_clearing_threshold():
    # excludes 1.0 but not 1.02: still not called faster
    assert verdict(1.01, 1.04, 0.02) == "no clear difference"
    assert verdict(1.03, 1.10, 0.02) == "faster"
    assert verdict(0.80, 0.97, 0.02) == "slower"


def test_mann_whitney_exact():
    u, p = mann_whitney_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert u == 25
    assert math.isclose(p, 2 / 252)
    _, p2 = mann_whitney_u([1, 2, 3], [1.5, 2.5, 3.5])
    assert p2 > 0.5


def test_moving_average():
    assert moving_average([1, 2, 3, 4], 2) == [1, 1.5, 2.5, 3.5]
