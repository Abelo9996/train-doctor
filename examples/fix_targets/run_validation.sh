#!/bin/sh
# Validation runs for `train-doctor fix` on three small real training scripts.
# Run from the repo root with an interpreter that has torch, for example:
#   PATH="$PWD/.venv/bin:$PATH" sh examples/fix_targets/run_validation.sh
# Each run writes a fix report, the compares it made, and (on a proven win) a card
# under .train-doctor/fix_targets/.
set -e
PY="${PY:-.venv/bin/python}"
OUT=.train-doctor/fix_targets

"$PY" -m train_doctor.cli fix --out "$OUT" -- "$PY" examples/fix_targets/mnist_cnn/train.py
"$PY" -m train_doctor.cli fix --out "$OUT" -- "$PY" examples/fix_targets/char_rnn/train.py
"$PY" -m train_doctor.cli fix --out "$OUT" -- "$PY" examples/fix_targets/tabular_mlp/train.py
