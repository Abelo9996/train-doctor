"""Run a training script, and on one chosen invocation freeze it mid-training.

Used to show what one stalled run does to `train-doctor compare`. Each call
bumps a counter in --counter; the call whose number equals --on-run sleeps for
--sleep seconds right after optimizer step --at-step, which is inside the
measured window. Every other call runs the script unchanged.

    python examples/stall_once.py --counter /tmp/cand.count --on-run 4 \
        -- examples/cnn_images/train.py --batch-augment
"""

import argparse
import runpy
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counter", required=True, help="file that counts invocations")
    ap.add_argument("--on-run", type=int, default=0, help="which invocation stalls (1 = first; 0 = never)")
    ap.add_argument("--at-step", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=60.0)
    ap.add_argument("script", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    script = args.script[1:] if args.script[:1] == ["--"] else args.script

    counter = Path(args.counter)
    n = int(counter.read_text() or 0) + 1 if counter.exists() else 1
    counter.write_text(str(n))

    if n == args.on_run:
        from torch.optim.optimizer import register_optimizer_step_post_hook

        steps = [0]

        def stall(optimizer, a, kw):
            steps[0] += 1
            if steps[0] == args.at_step:
                print(f"stall_once: sleeping {args.sleep:.0f} s after step {args.at_step} (invocation {n})", file=sys.stderr)
                time.sleep(args.sleep)

        register_optimizer_step_post_hook(stall)

    sys.argv = script
    runpy.run_path(script[0], run_name="__main__")


if __name__ == "__main__":
    main()
