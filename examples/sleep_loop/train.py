"""A training loop whose step time is a fixed sleep, so the true speedup is known.

Each step sleeps --step-ms (a stand-in for device work that other processes
don't slow down much), then runs one optimizer step on a tiny model. Comparing
--step-ms 20 against --step-ms 10 is a real 2x change, which makes this script
useful for checking what `compare` does with a stall (see ../stall_once.py).
"""

import argparse
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-ms", type=float, default=20.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = torch.nn.Linear(8, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(args.batch_size, 8)
    y = x.sum(dim=1, keepdim=True)
    for step in range(args.max_steps):
        time.sleep(args.step_ms / 1000)
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
