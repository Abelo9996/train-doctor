"""A tabular MLP on in-memory numeric features (compute-side, not data-bound).

The features are precomputed once into a tensor, so __getitem__ is a cheap
tensor index and the DataLoader is not the bottleneck. The model is a wider MLP,
so the step is dominated by the forward and backward pass. Included as a target
where the usual DataLoader fix does not apply: train-doctor fix should say so
rather than invent a win.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

IN_DIM = 64


def make_data(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN_DIM, generator=g)
    w = torch.randn(IN_DIM, generator=g)
    logits = x @ w + 0.3 * torch.randn(n, generator=g)
    y = (logits > logits.median()).long()
    return x, y


class MLP(nn.Module):
    def __init__(self, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rows", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=4000)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    x, y = make_data(args.rows)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        generator=torch.Generator().manual_seed(0),
    )
    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    step, t0 = 0, time.time()
    while step < args.max_steps:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 20 == 0:
                print(f"step {step} loss {loss.item():.4f}")
            if step >= args.max_steps:
                break
    print(f"done: {step} steps in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
