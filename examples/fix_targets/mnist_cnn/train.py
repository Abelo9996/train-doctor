"""A small MNIST-style CNN, written the way pytorch/examples/mnist is.

The dataset generates each 28x28 image on demand in __getitem__ and runs a
deterministic but expensive preprocessing pipeline over it (a repeated fixed
blur, then standardize), with num_workers=0. The dataset object itself holds only
the labels, so it is cheap to hand to worker processes, like an ImageFolder that
decodes files lazily. The model is small, so step time is dominated by preparing
the next batch: a data-loading bound loop, the most common reason small training
runs are slow. The preprocessing is deterministic and keyed by index, so moving
it to worker processes produces identical batches.

The images are synthetic (no download): each class is a bright patch over
per-index noise, with label noise, so the loss actually moves. Nothing here is
specific to train-doctor; it is a plain training script.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SyntheticDigits(Dataset):
    def __init__(self, n: int = 40000, seed: int = 0, passes: int = 14):
        self.seed = seed
        self.passes = passes
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, 10, (n,), generator=g)
        flip = torch.rand(n, generator=g) < 0.3  # 30% label noise keeps the loss meaningful
        self.labels[flip] = torch.randint(0, 10, (int(flip.sum()),), generator=g)
        k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
        self.kernel = k.view(1, 1, 3, 3)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        c = int(self.labels[i])
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)  # deterministic per index
        img = torch.rand(1, 1, 28, 28, generator=g) * 0.6
        r, col = (c // 3) * 8 + 2, (c % 3) * 8 + 2
        img[:, :, r : r + 8, col : col + 8] += 0.4
        for _ in range(self.passes):  # fixed, deterministic blur: the per-sample cost
            img = F.conv2d(img, self.kernel, padding=1)
        img = img.view(1, 28, 28)
        img = (img - img.mean()) / (img.std() + 1e-6)
        return img, self.labels[i]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=4000)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    ds = SyntheticDigits(args.rows)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, generator=torch.Generator().manual_seed(0)
    )
    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

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
