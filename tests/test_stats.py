import math

from train_doctor.stats import (
    bootstrap_median_ci,
    bootstrap_ratio_ci,
    describe,
    find_stalls,
    median,
    moving_average,
    paired_analysis,
    quantile,
    verdict,
    wilcoxon_signed_rank,
)


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


def test_wilcoxon_exact():
    # all 5 differences positive: the most extreme of 2**5 sign patterns, on both sides
    w = wilcoxon_signed_rank([0.1, 0.2, 0.3, 0.4, 0.5])
    assert w["w_plus"] == 15 and w["n"] == 5
    assert math.isclose(w["p_two_sided"], 2 / 32) and math.isclose(w["p_min"], 2 / 32)
    # one small negative: W+ = 13; |W+ - 7.5| >= 5.5 happens for W+ in {0, 1, 2, 13, 14, 15}
    assert math.isclose(wilcoxon_signed_rank([0.1, -0.2, 0.3, 0.4, 0.5])["p_two_sided"], 6 / 32)
    # tied magnitudes get average ranks (2, 2, 2, 4); zeros are dropped
    w = wilcoxon_signed_rank([1, 1, -1, 2, 0])
    assert w["n"] == 4 and w["w_plus"] == 8 and math.isclose(w["p_two_sided"], 0.5)
    assert wilcoxon_signed_rank([0, 0])["p_two_sided"] == 1.0


def test_bootstrap_median_is_the_range_for_five_or_fewer():
    for n in (2, 3, 4, 5):
        xs = [float(i) for i in range(1, n + 1)]
        _, lo, hi = bootstrap_median_ci(xs)
        assert (lo, hi) == (1.0, n)
    _, lo, hi = bootstrap_median_ci([float(i) for i in range(1, 8)])
    assert 1 < lo < 4 < hi < 7
    assert bootstrap_median_ci([0.1, 0.3, 0.2]) == bootstrap_median_ci([0.1, 0.3, 0.2])


def test_find_stalls_flags_only_far_outliers():
    assert find_stalls([100, 98, 103, 2.5, 101]) == [False, False, False, True, False]
    # 30% run-to-run noise on a busy laptop is not a stall
    assert not any(find_stalls([100, 70, 130, 95, 110]))
    # a run under half the median, but the arm itself is that noisy: not flagged
    assert not any(find_stalls([100, 45, 200, 60, 150]))
    assert not any(find_stalls([100, 1]))  # too few runs to tell
    # a very noisy arm (load average 42 on 10 cores, real run): the 15x slower run is still a stall
    assert find_stalls([651.6, 1322.2, 49.5, 725.8, 1289.8]) == [False, False, True, False, False]


# Throughput (samples/s) per pair. The candidate is about 1.3x faster in every pair, except
# pair 2, where a machine hiccup made one run about 40x slower.
BASE = [100.0, 102.0, 99.0, 101.0, 98.0]
CAND = [130.0, 133.0, 128.0, 131.0, 127.0]


def test_candidate_stall_is_set_aside_and_verdict_survives():
    cand = CAND.copy()
    cand[2] = 3.2
    pa = paired_analysis(BASE, cand)
    assert pa["stalled_pairs"] == [2] and pa["excluded_pairs"] == [2]
    assert pa["pairs"][2]["stalled"] == ["candidate"]
    ra = pa["ratio"]
    assert ra["n_pairs"] == 4 and 1.29 < ra["low"] <= ra["point"] <= ra["high"] < 1.31
    assert verdict(ra["low"], ra["high"], 0.02) == "faster"
    assert pa["wins"] == 4 and pa["n_pairs"] == 4
    # the same numbers analysed the 0.1.1 way (unpaired ratio of medians) are not called faster
    _, lo, _ = bootstrap_ratio_ci(BASE, cand)
    assert lo < 0.1 and verdict(lo, 2.0, 0.02) == "no clear difference"
    # and with every pair kept, the paired interval would be blown open too
    assert pa["ratio_all_pairs"]["low"] < 0.05


def test_baseline_stall_and_whole_pair_stall_are_set_aside():
    base = BASE.copy()
    base[1] = 2.4
    pa = paired_analysis(base, CAND)
    assert pa["excluded_pairs"] == [1] and pa["pairs"][1]["stalled"] == ["baseline"]
    assert pa["ratio"]["high"] < 1.31
    # a hiccup that slows both runs of one pair
    base, cand = BASE.copy(), CAND.copy()
    base[3], cand[3] = 3.0, 2.0
    pa = paired_analysis(base, cand)
    assert pa["excluded_pairs"] == [3] and pa["pairs"][3]["stalled"] == ["baseline", "candidate"]
    assert verdict(pa["ratio"]["low"], pa["ratio"]["high"], 0.02) == "faster"


def test_too_many_stalls_are_kept_and_the_answer_stays_open():
    cand = CAND.copy()
    cand[1], cand[3] = 3.0, 2.5
    pa = paired_analysis(BASE, cand)
    assert pa["max_excluded"] == 1
    assert pa["stalled_pairs"] == [1, 3] and pa["excluded_pairs"] == [] and pa["stalls_kept"] == [1, 3]
    assert verdict(pa["ratio"]["low"], pa["ratio"]["high"], 0.02) == "no clear difference"
    assert "ratio_all_pairs" not in pa


def test_paired_no_difference_is_not_called_faster():
    base = [100.0, 104.0, 97.0, 102.0, 99.0]
    cand = [101.0, 102.0, 99.0, 100.0, 103.0]
    pa = paired_analysis(base, cand)
    assert pa["excluded_pairs"] == []
    assert verdict(pa["ratio"]["low"], pa["ratio"]["high"], 0.02) == "no clear difference"


def test_pairing_cancels_drift_that_hits_both_runs():
    # The machine slows down steadily; within each pair the candidate is 1.10x faster.
    base = [100.0, 80.0, 60.0, 45.0, 35.0]
    cand = [x * 1.10 for x in base]
    pa = paired_analysis(base, cand)
    assert math.isclose(pa["ratio"]["low"], 1.10) and math.isclose(pa["ratio"]["high"], 1.10)
    _, lo, hi = bootstrap_ratio_ci(base, cand)
    assert lo < 1 < hi  # unpaired: the drift swamps the effect


def test_moving_average():
    assert moving_average([1, 2, 3, 4], 2) == [1, 1.5, 2.5, 3.5]
