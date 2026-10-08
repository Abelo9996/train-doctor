"""Small CNN on a synthetic 10-class image dataset.

Written the way a lot of first training scripts are: per-sample augmentation
in __getitem__, num_workers=0, and the loss and accuracy read back with
.item() and printed on every step.

Each command-line flag stands in for one code edit, so train-doctor can
compare a single change at a time:

  --num-workers N   DataLoader workers (default 0)
  --log-every N     read metrics and print every N steps (default 1);
                    between logs, metrics stay on the device
  --amp             autocast to float16 on mps/cuda
  --batch-augment   do the crop, flip and normalize once per batch on the
                    device instead of per sample in __getitem__
  --batch-size N    default 64
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
STD = torch.tensor([0.25, 0.25, 0.25]).view(3, 1, 1)


class SyntheticImages(Dataset):
    """10 classes told apart by a faint colored square in noise. Stored as uint8 like decoded JPEGs."""

    def __init__(self, n: int = 20000, seed: int = 0, augment: bool = True):
        self.augment = augment
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, 10, (n,), generator=g)
        base = torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.uint8)
        # a faint class-colored square blended into the noise, plus 10% label noise
        colors = torch.linspace(0, 255, 10)
        for c in range(10):
            idx = (self.labels == c).nonzero().squeeze(1)
            patch = base[idx, c % 3, 8:24, 8:24].float()
            base[idx, c % 3, 8:24, 8:24] = (0.75 * patch + 0.25 * colors[c]).to(torch.uint8)
        flip = torch.rand(n, generator=g) < 0.1
        self.labels[flip] = torch.randint(0, 10, (int(flip.sum()),), generator=g)
        self.images = base

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if not self.augment:
            return self.images[i], self.labels[i]
        img = self.images[i].float() / 255.0
        # random crop with 4 px padding
        img = F.pad(img, (4, 4, 4, 4), mode="reflect")
        x, y = torch.randint(0, 9, (2,)).tolist()
        img = img[:, y : y + 32, x : x + 32]
        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])
        img = (img - MEAN) / STD
        return img, self.labels[i]


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Same augmentation as SyntheticImages.__getitem__, vectorized over a uint8 batch on the device."""
    b = x.shape[0]
    x = F.pad(x.float() / 255.0, (4, 4, 4, 4), mode="reflect")
    oy = torch.randint(0, 9, (b,), device=x.device)
    ox = torch.randint(0, 9, (b,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (oy[:, None] + ar)[:, None, :, None].expand(b, 3, 32, 40)
    x = torch.gather(x, 2, rows)
    cols = (ox[:, None] + ar)[:, None, None, :].expand(b, 3, 32, 32)
    x = torch.gather(x, 3, cols)
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.c2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.c3(x))), 2)
        return self.fc(x.flatten(1))


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
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch-augment", action="store_true")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.002)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    ds = SyntheticImages(augment=not args.batch_augment)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    model = Net().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    step = 0
    run_loss = torch.zeros((), device=device)
    run_correct = torch.zeros((), device=device)
    t0 = time.time()
    while step < args.max_steps:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if args.batch_augment:
                xb = augment_batch(xb)
            with torch.autocast(device.type, dtype=torch.float16, enabled=args.amp):
                out = model(xb)
                loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            run_loss += loss.detach()
            run_correct += (out.argmax(1) == yb).sum()
            if step % args.log_every == 0:
                n = args.log_every * args.batch_size
                print(f"step {step} loss {run_loss.item() / args.log_every:.4f} acc {run_correct.item() / n:.3f}")
                run_loss.zero_()
                run_correct.zero_()
            if step >= args.max_steps:
                break
    print(f"done: {step} steps in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
