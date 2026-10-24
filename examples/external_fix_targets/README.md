# external fix targets

`train-doctor fix` run on two **unmodified third-party scripts** from
[pytorch/examples](https://github.com/pytorch/examples): `mnist/main.py` (a CNN) and `mnist_rnn/main.py`
(a GRU). These are the honest test, because the scripts were written with no knowledge of this tool. The
scripts are not vendored here; fetch them and reproduce with:

```bash
mkdir -p /tmp/ext/mnist /tmp/ext/mnist_rnn
curl -so /tmp/ext/mnist/main.py     https://raw.githubusercontent.com/pytorch/examples/main/mnist/main.py
curl -so /tmp/ext/mnist_rnn/main.py https://raw.githubusercontent.com/pytorch/examples/main/mnist_rnn/main.py
PY=.venv/bin/python                                   # an interpreter with torch + torchvision
"$PY" -m pip install torchvision                      # these scripts need torchvision + MNIST (~11 MB)
"$PY" -m train_doctor.cli fix              --cwd /tmp/ext/mnist     -- "$PY" main.py
"$PY" -m train_doctor.cli fix --agent claude --yes --cwd /tmp/ext/mnist     -- "$PY" main.py
"$PY" -m train_doctor.cli fix              --cwd /tmp/ext/mnist_rnn -- "$PY" main.py
"$PY" -m train_doctor.cli fix --agent claude       --cwd /tmp/ext/mnist_rnn -- "$PY" main.py
```

Machine: Apple M4 MacBook (10 logical cores, 16 GB), torch 2.14.1, torchvision 0.29.1. `mnist/main.py`
runs on MPS; `mnist_rnn/main.py` runs on CPU (its default; it uses the GPU only with `--accel`). Numbers
differ on another machine or under load. The `fix.md` reports are copied here verbatim.

## mnist/main.py (CNN, MPS)

This script already sets `num_workers=1, persistent_workers=True, pin_memory=True`, so the data loader is
not the bottleneck and there is no change `fix` can inject.

- **Plain `fix`: no proven win.** The top findings (`no_mixed_precision`, `no_torch_compile`) need a source
  edit, so it stops at suggestions. ([mnist_main_plain_fix.md](mnist_main_plain_fix.md))
- **`fix --agent claude --yes`:** the agent made each source change in a scratch copy and `compare` judged it
  (across two runs; the second was bounded to `torch.compile` with `--budget 1` to fit the environment's time
  limit). Both changes were faster and both were **reverted by the loss check**, so there is no proven
  loss-preserving win:

  | Source change (by claude) | Speed | Loss | Decision |
  |---|---|---|---|
  | `torch.compile(model)` | 1.22x (1.12 to 1.23) | outside 5% tolerance (9.0% mean) | reject, reverted |
  | autocast float16 | 1.32x (4,280 -> 6,618 samples/s) | outside 5% tolerance | reject, reverted |

  This is the guardrail doing its job: on this script the obvious speedups change the training numerics past
  the tolerance in the measured window, so `fix` does not keep them. A user who accepts different numerics can
  re-run with a higher `--loss-tol`. The committed [mnist_main_agent_fix.md](mnist_main_agent_fix.md) is the
  bounded `torch.compile` run.

## mnist_rnn/main.py (GRU, CPU)

Run with its defaults, this script trains on CPU with `num_workers=0`, so per-sample work serializes with
the step.

- **Plain `fix`: a proven win, no agent needed.** `num_workers=5` -> **1.30x faster** (95% interval 1.18 to
  1.90, candidate won 5 of 5 pairs), **loss identical**. `fix` wrote the reversible patch
  ([mnist_rnn.fix.patch](mnist_rnn.fix.patch): adds `num_workers=5` to the DataLoader) and the card
  ([PNG](mnist_rnn.card.png), [SVG](mnist_rnn.card.svg)). ([mnist_rnn_plain_fix.md](mnist_rnn_plain_fix.md))
- **`fix --agent claude` (plan only):** reports that it would inject `num_workers=5` (no agent) and then ask
  the agent to try `torch.compile(model)` and a device move to MPS, one at a time, each proven with a
  compare. Printed as a plan with a cost estimate; nothing was spent. Not run to `--yes` here, since the
  injected win already stands.

## What this validates

The orchestrator's point was right: on a well-written third-party script (`mnist/main.py`) the data loader is
already tuned, so plain `fix` has only suggestions. `fix --agent` closes that gap by making the suggested
source changes and proving them, and its loss check is a real guardrail: here it reverted two genuinely
faster changes because they moved the loss. On the second script (`mnist_rnn/main.py`) plain `fix` found a
loss-identical 1.30x with no agent at all. One proven win, one honest "both faster changes reverted"; neither
number was invented.
