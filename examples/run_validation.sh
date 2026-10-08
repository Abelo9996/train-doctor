#!/bin/sh
# The exact sequence that produced examples/*/runs, in order.
# Run from the repo root with a venv that has torch and train-doctor installed:
#   uv venv --python 3.12 && uv pip install -e . torch numpy
#   PATH="$PWD/.venv/bin:$PATH" sh examples/run_validation.sh
# Each flag on the training scripts stands in for one code edit. Which edit
# came next was decided from the previous profile's top finding and the
# previous compare's decision (see examples/README.md).
set -e

CNN="python examples/cnn_images/train.py"
OUT=examples/cnn_images/runs
train-doctor profile --out $OUT --label baseline -- $CNN
train-doctor compare --out $OUT --label workers-5 --baseline "$CNN" --candidate "$CNN --num-workers 5"
train-doctor compare --seconds 4 --out $OUT --label batch-augment --baseline "$CNN" --candidate "$CNN --batch-augment"
train-doctor profile --out $OUT --label batch-augment -- $CNN --batch-augment
train-doctor compare --seconds 4 --out $OUT --label log-every-50 --baseline "$CNN --batch-augment" --candidate "$CNN --batch-augment --log-every 50"
train-doctor profile --out $OUT --label log-every-50 -- $CNN --batch-augment --log-every 50
train-doctor compare --seconds 4 --out $OUT --label amp --baseline "$CNN --batch-augment --log-every 50" --candidate "$CNN --batch-augment --log-every 50 --amp"
train-doctor compare --seconds 4 --out $OUT --label original-vs-final --baseline "$CNN" --candidate "$CNN --batch-augment --log-every 50"

MLP="python examples/tabular_mlp/train.py"
OUT=examples/tabular_mlp/runs
train-doctor profile --out $OUT --label baseline -- $MLP
train-doctor compare --seconds 4 --out $OUT --label ckpt-every-200 --baseline "$MLP" --candidate "$MLP --ckpt-every 200"
train-doctor compare --seconds 4 --repeats 9 --out $OUT --label ckpt-every-200-9-repeats --baseline "$MLP" --candidate "$MLP --ckpt-every 200"
train-doctor profile --out $OUT --label ckpt-every-200 -- $MLP --ckpt-every 200
train-doctor compare --seconds 4 --out $OUT --label workers-5 --baseline "$MLP --ckpt-every 200" --candidate "$MLP --ckpt-every 200 --num-workers 5"
train-doctor profile --out $OUT --label workers-5 -- $MLP --ckpt-every 200 --num-workers 5
train-doctor compare --seconds 4 --out $OUT --label original-vs-final --baseline "$MLP" --candidate "$MLP --ckpt-every 200 --num-workers 5"
