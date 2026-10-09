"""Re-analyze committed compare runs with the paired method used since 0.1.2.

Runs made with 0.1.1 already executed baseline and candidate back to back for
each repeat index, so their per-run numbers can be paired after the fact.
Prints the 0.1.1 result (ratio of medians) next to the paired one.

    python examples/reanalyze_paired.py examples/*/runs/*compare*/compare.json
"""

import json
import sys
from pathlib import Path

from train_doctor.stats import bootstrap_ratio_ci, paired_analysis, verdict


def main(paths: list[str]) -> None:
    for path in paths:
        r = json.loads(Path(path).read_text())
        runs = [x for x in r["runs"] if x["phase"] == "measure" and x.get("value")]
        idx = sorted({x["index"] for x in runs if sum(1 for y in runs if y["index"] == x["index"]) == 2})
        b = [next(x["value"] for x in runs if x["arm"] == "baseline" and x["index"] == i) for i in idx]
        c = [next(x["value"] for x in runs if x["arm"] == "candidate" and x["index"] == i) for i in idx]
        min_effect = r["settings"]["min_effect"]
        p0, l0, h0 = bootstrap_ratio_ci(b, c)
        pa = paired_analysis(b, c)
        ra = pa["ratio"]
        print(Path(path).parent.name)
        print(f"  unpaired ratio of medians: {p0:.2f}x ({l0:.2f} to {h0:.2f}), {verdict(l0, h0, min_effect)}")
        print(
            f"  paired median ratio:       {ra['point']:.2f}x ({ra['low']:.2f} to {ra['high']:.2f}), "
            f"{verdict(ra['low'], ra['high'], min_effect)}, candidate won {pa['wins']} of {pa['n_pairs']} pairs"
        )
        ratios = ", ".join(f"{p['ratio']:.2f}" for p in pa["pairs"])
        print(f"  pair ratios: {ratios}")
        for p in pa["pairs"]:
            if p["stalled"]:
                print(f"  pair {p['pair']}: {' and '.join(p['stalled'])} run stalled, {'set aside' if p['excluded'] else 'kept'}")


if __name__ == "__main__":
    main(sys.argv[1:])
