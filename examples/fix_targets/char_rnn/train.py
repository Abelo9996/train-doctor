"""A tiny char-level LSTM language model on CPU (nanoGPT-small in spirit).

The corpus is a short synthetic text repeated to a few hundred KB. The dataset
encodes a fresh window per sample inside __getitem__ in plain Python (character
to index through a dict, then build input/target tensors), with num_workers=0,
so the training step waits on that Python work. The model is small and runs on
CPU. A common shape for small language-model experiments.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump! "
    "the five boxing wizards jump quickly. "
)


def make_corpus(size: int = 200_000) -> str:
    reps = size // len(SEED_TEXT) + 1
    return (SEED_TEXT * reps)[:size]


class CharWindows(Dataset):
    def __init__(self, text: str, block: int = 64, n: int = 8000, seed: int = 0):
        self.text = text
        self.block = block
        self.n = n
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab = len(chars)
        g = torch.Generator().manual_seed(seed)
        self.starts = torch.randint(0, len(text) - block - 1, (n,), generator=g).tolist()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = self.starts[i]
        window = self.text[s : s + self.block + 1]
        ids = [self.stoi[c] for c in window]  # per-character Python encode
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


class CharLSTM(nn.Module):
    def __init__(self, vocab: int, emb: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.head(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=4000)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    ds = CharWindows(make_corpus(), block=args.block, n=args.samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        generator=torch.Generator().manual_seed(0),
    )
    model = CharLSTM(ds.vocab).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    step, t0 = 0, time.time()
    while step < args.max_steps:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
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
