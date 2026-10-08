"""Small, dependency-free statistics used for timing and comparison.

Everything here is deterministic: the bootstrap uses a fixed seed so the same
inputs always produce the same interval.
"""

from __future__ import annotations

import functools
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


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """U statistic for ``b > a`` and its exact two-sided p-value (small samples, no ties correction).

    Exact enumeration is fine for the repeat counts used here (n <= 12 per arm).
    """
    na, nb = len(a), len(b)
    u = 0.0
    for x in a:
        for y in b:
            if y > x:
                u += 1
            elif y == x:
                u += 0.5
    if na == 0 or nb == 0:
        return u, 1.0
    if na > 12 or nb > 12:
        # normal approximation
        mu = na * nb / 2
        sigma = math.sqrt(na * nb * (na + nb + 1) / 12)
        z = (u - mu) / sigma if sigma else 0.0
        p = math.erfc(abs(z) / math.sqrt(2))
        return u, min(1.0, p)
    # exact null distribution of U via counting rank-sum arrangements
    counts = _u_distribution(na, nb)
    total = sum(counts.values())
    mu = na * nb / 2
    dev = abs(u - mu)
    extreme = sum(c for k, c in counts.items() if abs(k - mu) >= dev - 1e-9)
    return u, min(1.0, extreme / total)


def _u_distribution(na: int, nb: int) -> dict[float, int]:
    # number of orderings giving each U value, by dynamic programming over (i, j)

    @functools.cache
    def f(i: int, j: int) -> tuple:
        if i == 0 or j == 0:
            return ((0, 1),)
        out: dict[int, int] = {}
        for k, c in f(i, j - 1):  # largest element is from b: it beats all i elements of a
            out[k + i] = out.get(k + i, 0) + c
        for k, c in f(i - 1, j):  # largest element is from a: adds nothing
            out[k] = out.get(k, 0) + c
        return tuple(sorted(out.items()))

    return {float(k): c for k, c in f(na, nb)}


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
