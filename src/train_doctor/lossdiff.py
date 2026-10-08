"""Check that a candidate's loss trajectory stays close to the baseline's.

Trajectories are aligned on cumulative samples seen when both runs report
samples (so a batch-size change is compared at equal data), otherwise on step
index. Both curves are smoothed with a short moving average before the
relative difference is taken.
"""

from __future__ import annotations

from train_doctor.stats import mean, moving_average

SMOOTH = 5


def trajectory(ev: dict) -> tuple[list[float], list[float], str]:
    """(x, loss, axis) for one run's evidence."""
    loss = ev.get("loss") or []
    if not loss:
        return [], [], "none"
    by_step = ev.get("samples_by_step") or []
    if by_step and all(s is not None for s in by_step):
        cum = []
        acc = 0
        for s in by_step:
            acc += s
            cum.append(acc)
        xs, ys = [], []
        for step, val in loss:
            if 0 <= step < len(cum):
                xs.append(float(cum[step]))
                ys.append(float(val))
        if xs:
            return xs, ys, "samples"
    return [float(s + 1) for s, _ in loss], [float(v) for _, v in loss], "step"


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[hi]
    if x1 == x0:
        return ys[lo]
    return ys[lo] + (ys[hi] - ys[lo]) * (x - x0) / (x1 - x0)


def diff(base_ev: dict, cand_ev: dict, tol: float, max_steps: int | None = None) -> dict:
    bx, by, bax = trajectory(base_ev)
    cx, cy, cax = trajectory(cand_ev)
    if not by or not cy:
        return {"status": "unavailable", "reason": "a run recorded no loss values", "tol": tol}
    axis = "samples" if bax == cax == "samples" else "step"
    if axis == "step" and (bax != cax):
        bx, cx = [float(i + 1) for i in range(len(by))], [float(i + 1) for i in range(len(cy))]
    if max_steps:
        bx, by = bx[:max_steps], by[:max_steps]
        cx, cy = cx[:max_steps], cy[:max_steps]
    bs = moving_average(by, SMOOTH)
    cs = moving_average(cy, SMOOTH)
    lo = max(bx[0], cx[0])
    hi = min(bx[-1], cx[-1])
    pts = [(x, b) for x, b in zip(bx, bs, strict=False) if lo <= x <= hi]
    if len(pts) < 2:
        return {"status": "unavailable", "reason": "the two runs share fewer than 2 loss points", "tol": tol, "axis": axis}
    rel = []
    for x, b in pts:
        c = _interp(cx, cs, x)
        rel.append(abs(c - b) / max(abs(b), 1e-8))
    raw_identical = len(by) == len(cy) and all(a == b for a, b in zip(by, cy, strict=False))
    out = {
        "axis": axis,
        "points": len(pts),
        "smoothing": SMOOTH,
        "tol": tol,
        "max_rel_diff": max(rel),
        "mean_rel_diff": mean(rel),
        "final_rel_diff": rel[-1],
        "baseline_final": pts[-1][1],
        "candidate_final": _interp(cx, cs, pts[-1][0]),
    }
    if raw_identical:
        out["status"] = "identical"
    elif out["mean_rel_diff"] <= tol and out["final_rel_diff"] <= tol:
        out["status"] = "within tolerance"
    else:
        out["status"] = "outside tolerance"
    return out
