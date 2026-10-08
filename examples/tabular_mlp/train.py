"""MLP on synthetic tabular records with Python feature engineering per row.

The dataset keeps raw CSV-like text rows and turns each one into a feature
vector inside __getitem__: parse the line, normalize numeric columns,
one-hot encode categoricals, and hash free-text tokens into buckets, all in
plain Python. The loop also writes a checkpoint every few steps.

Each flag stands in for one code edit:

  --num-workers N     DataLoader workers (default 0)
  --precompute        featurize every row once at startup and train from a
                      tensor (same features, computed once)
  --ckpt-every N      torch.save every N steps (default 5, 0 disables)
  --device            auto picks cuda, then mps, then cpu
"""

import argparse
import os
import random
import shutil
import tempfile
import time
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

CITIES = [f"city_{i}" for i in range(40)]
JOBS = [f"job_{i}" for i in range(30)]
WORDS = [
    "fast",
    "slow",
    "cheap",
    "premium",
    "broken",
    "new",
    "used",
    "repaired",
    "refurbished",
    "blue",
    "red",
    "green",
    "large",
    "small",
    "heavy",
    "light",
]
TEXT_BUCKETS = 128
NUMERIC = ["age", "income", "tenure", "visits", "spend", "score"]


def make_rows(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        age = rng.randint(18, 80)
        income = rng.lognormvariate(10.5, 0.6)
        tenure = rng.randint(0, 30)
        visits = rng.randint(0, 200)
        spend = rng.lognormvariate(5, 1)
        score = rng.random()
        city = rng.choice(CITIES)
        job = rng.choice(JOBS)
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 12)))
        signal = (age > 45) + (income > 40000) + ("premium" in text) + (CITIES.index(city) % 3 == 0)
        label = int(signal + rng.random() * 1.5 > 2)
        rows.append(f"{age},{income:.2f},{tenure},{visits},{spend:.2f},{score:.4f},{city},{job},{text},{label}")
    return rows


def column_stats(rows: list[str]) -> list[tuple[float, float]]:
    cols = [[float(r.split(",")[i]) for r in rows[:2000]] for i in range(len(NUMERIC))]
    out = []
    for c in cols:
        m = sum(c) / len(c)
        sd = (sum((x - m) ** 2 for x in c) / len(c)) ** 0.5
        out.append((m, sd or 1.0))
    return out


def featurize(line: str, stats) -> tuple[list[float], int]:
    parts = line.split(",")
    feats = []
    for i, (m, sd) in enumerate(stats):
        feats.append((float(parts[i]) - m) / sd)
    city = [0.0] * len(CITIES)
    city[CITIES.index(parts[6])] = 1.0
    job = [0.0] * len(JOBS)
    job[JOBS.index(parts[7])] = 1.0
    text = [0.0] * TEXT_BUCKETS
    for tok in parts[8].split():
        for gram in (tok, tok[:3], tok[-3:]):
            text[zlib.crc32(gram.encode()) % TEXT_BUCKETS] += 1.0
    return feats + city + job + text, int(parts[9])


class RawRows(Dataset):
    def __init__(self, rows, stats):
        self.rows = rows
        self.stats = stats

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y = featurize(self.rows[i], self.stats)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y)


IN_DIM = len(NUMERIC) + len(CITIES) + len(JOBS) + TEXT_BUCKETS


class MLP(nn.Module):
    def __init__(self, hidden: int = 1024):
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
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    rows = make_rows(args.rows)
    stats = column_stats(rows)
    if args.precompute:
        feats, labels = zip(*(featurize(r, stats) for r in rows), strict=True)
        ds = TensorDataset(torch.tensor(feats, dtype=torch.float32), torch.tensor(labels))
    else:
        ds = RawRows(rows, stats)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=args.num_workers > 0, drop_last=True
    )
    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_dir = tempfile.mkdtemp(prefix="tabular_mlp_ckpt_")
    try:
        train(model, opt, loader, device, args, ckpt_dir)
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def train(model, opt, loader, device, args, ckpt_dir):
    step = 0
    t0 = time.time()
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
            if args.ckpt_every and step % args.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, os.path.join(ckpt_dir, "last.pt"))
            if step >= args.max_steps:
                break
    print(f"done: {step} steps in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
