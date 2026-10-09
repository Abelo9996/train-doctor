"""Small, dependency-free statistics used for timing and comparison.

Everything here is deterministic: the bootstrap uses a fixed seed so the same
inputs always produce the same interval.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (same as numpy's default)."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def median(xs: Sequence[float]) -> float:
    return quantile(xs, 0.5)


def stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def describe(xs: Sequence[float]) -> dict:
    """Summary used for step times and per-run throughput."""
    if not xs:
        return {"n": 0}
    m = mean(xs)
    sd = stdev(xs)
    return {
        "n": len(xs),
        "median": median(xs),
        "mean": m,
        "std": sd,
        "cv": sd / m if m else float("nan"),
        "min": min(xs),
        "p10": quantile(xs, 0.10),
        "p25": quantile(xs, 0.25),
        "p75": quantile(xs, 0.75),
        "p90": quantile(xs, 0.90),
        "max": max(xs),
        "iqr": quantile(xs, 0.75) - quantile(xs, 0.25),
    }


def bootstrap_ratio_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Ratio of medians (candidate / baseline) with a percentile bootstrap interval.

    Each arm is resampled independently with replacement. Returns
    ``(point, low, high)``.
    """
    if not baseline or not candidate:
        raise ValueError("both arms need at least one value")
    point = median(candidate) / median(baseline)
    rng = random.Random(seed)
    nb, nc = len(baseline), len(candidate)
    ratios = []
    for _ in range(n_boot):
        b = median([baseline[rng.randrange(nb)] for _ in range(nb)])
        c = median([candidate[rng.randrange(nc)] for _ in range(nc)])
        if b > 0:
            ratios.append(c / b)
    alpha = (1 - level) / 2
    return point, quantile(ratios, alpha), quantile(ratios, 1 - alpha)


def bootstrap_median_ci(
    xs: Sequence[float],
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Median of ``xs`` with a percentile bootstrap interval. Returns ``(point, low, high)``.

    At the 95% level with 5 or fewer values, the interval is the range of the values:
    a resample has the smallest value as its median in more than 2.5% of draws.
    """
    if not xs:
        raise ValueError("need at least one value")
    rng = random.Random(seed)
    n = len(xs)
    meds = [median([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)]
    alpha = (1 - level) / 2
    return median(xs), quantile(meds, alpha), quantile(meds, 1 - alpha)


def wilcoxon_signed_rank(d: Sequence[float]) -> dict:
    """Exact two-sided Wilcoxon signed-rank test that the differences ``d`` are centered on 0.

    Zero differences are dropped and tied magnitudes get average ranks; the null
    distribution is counted exactly (dynamic programming over doubled ranks), so
    ties don't need a correction. ``p_min`` is the smallest p-value the number of
    nonzero differences allows (2 / 2**n): with 5 pairs it is 0.0625.
    """
    nz = [x for x in d if x != 0]
    n = len(nz)
    if n == 0:
        return {"w_plus": 0.0, "n": 0, "p_two_sided": 1.0, "p_min": 1.0}
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks2 = [0] * n  # doubled ranks, integers even with ties
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks2[order[k]] = i + j + 2  # 2 * average of ranks i+1 .. j+1
        i = j + 1
    w2 = sum(r for r, x in zip(ranks2, nz, strict=True) if x > 0)
    counts = {0: 1}
    for r in ranks2:
        nxt = dict(counts)
        for s, c in counts.items():
            nxt[s + r] = nxt.get(s + r, 0) + c
        counts = nxt
    total = 2**n
    mu2 = sum(ranks2) / 2
    dev = abs(w2 - mu2)
    extreme = sum(c for s, c in counts.items() if abs(s - mu2) >= dev - 1e-9)
    return {"w_plus": w2 / 2, "n": n, "p_two_sided": min(1.0, extreme / total), "p_min": min(1.0, 2 / total)}


def find_stalls(xs: Sequence[float], factor: float = 2.0, mads: float = 4.0, hard_factor: float = 4.0) -> list[bool]:
    """Flag runs that are far slower than the other runs of the same command.

    On a log scale, a run is a stall when its throughput is below the arm's median
    by more than ``factor`` (default: under half the median) and by more than
    ``mads`` scaled median absolute deviations, or when it is below the median by
    more than ``hard_factor`` however noisy the arm is (default: under a quarter).
    Every run in an arm executes the same command, so a gap that large is the
    machine, not the change.
    """
    logs = [math.log(x) if x and x > 0 else float("-inf") for x in xs]
    finite = [v for v in logs if v != float("-inf")]
    if len(finite) < 3:
        return [v == float("-inf") for v in logs]
    m = median(finite)
    mad = 1.4826 * median([abs(v - m) for v in finite])
    limit = min(max(math.log(factor), mads * mad), math.log(hard_factor))
    return [m - v > limit for v in logs]


def paired_analysis(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired comparison of throughput: ``baseline[i]`` and ``candidate[i]`` ran back to back.

    Each pair gives a log ratio ``log(candidate / baseline)``. Pairs where either
    run is a stall (see ``find_stalls``) are set aside, at most ``n // 4`` of them
    (1 of 5, 2 of 9) and only if at least 3 pairs remain; if more pairs look
    stalled, none are set aside and ``stalls_kept`` says so. The estimate is the
    median pair ratio, with a percentile bootstrap interval of that median.
    """
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate need one value per pair")
    n = len(baseline)
    if n < 2:
        raise ValueError("need at least 2 pairs")
    sb, sc = find_stalls(baseline), find_stalls(candidate)
    pairs = []
    for i, (b, c) in enumerate(zip(baseline, candidate, strict=True)):
        stalled = [arm for arm, flag in (("baseline", sb[i]), ("candidate", sc[i])) if flag]
        pairs.append({"pair": i, "baseline": b, "candidate": c, "ratio": c / b, "log_ratio": math.log(c / b), "stalled": stalled})
    flagged = [p["pair"] for p in pairs if p["stalled"]]
    cap = n // 4 if n >= 4 else 0
    excluded = flagged if flagged and len(flagged) <= cap and n - len(flagged) >= 3 else []
    for p in pairs:
        p["excluded"] = p["pair"] in excluded

    def estimate(sel: list[dict]) -> dict:
        logs = [p["log_ratio"] for p in sel]
        point, lo, hi = bootstrap_median_ci(logs, level, n_boot, seed)
        return {"point": math.exp(point), "low": math.exp(lo), "high": math.exp(hi), "n_pairs": len(sel)}

    kept = [p for p in pairs if not p["excluded"]]
    out = {
        "pairs": pairs,
        "ratio": estimate(kept),
        "wins": sum(1 for p in kept if p["ratio"] > 1),
        "n_pairs": len(kept),
        "wilcoxon": wilcoxon_signed_rank([p["log_ratio"] for p in kept]),
        "stalled_pairs": flagged,
        "excluded_pairs": excluded,
        "stalls_kept": [i for i in flagged if i not in excluded],
        "max_excluded": cap,
    }
    if excluded:
        out["ratio_all_pairs"] = estimate(pairs)
    return out


def verdict(ratio_low: float, ratio_high: float, min_effect: float) -> str:
    """Speed verdict from the interval of candidate/baseline throughput.

    * ``faster``: the whole interval is above ``1 + min_effect``
    * ``slower``: the whole interval is below ``1 - min_effect``
    * ``no clear difference``: anything else, including intervals that exclude
      1.0 but don't clear the practical threshold
    """
    if ratio_low > 1 + min_effect:
        return "faster"
    if ratio_high < 1 - min_effect:
        return "slower"
    return "no clear difference"


def moving_average(xs: Sequence[float], window: int) -> list[float]:
    if window <= 1:
        return list(xs)
    out = []
    acc = 0.0
    for i, x in enumerate(xs):
        acc += x
        if i >= window:
            acc -= xs[i - window]
        out.append(acc / min(i + 1, window))
    return out
