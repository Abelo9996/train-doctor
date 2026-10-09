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

# 0.1.2: one injected stall (stall_once.py freezes the 4th invocation of one command, which is
# measured pair 2 counting from 0, for 60 s inside its measured window). Load averages at run
# starts are in examples/README.md.
W="python examples/stall_once.py"
S=examples/sleep_loop/train.py
OUT=examples/sleep_loop/runs
rm -f /tmp/td-stall-*.count
train-doctor compare --out $OUT --label stall-in-candidate --baseline "$W --counter /tmp/td-stall-f-base.count -- $S --step-ms 20" --candidate "$W --counter /tmp/td-stall-f-cand.count --on-run 4 --sleep 60 -- $S --step-ms 10"
train-doctor compare --out $OUT --label stall-in-baseline --baseline "$W --counter /tmp/td-stall-g-base.count --on-run 4 --sleep 60 -- $S --step-ms 20" --candidate "$W --counter /tmp/td-stall-g-cand.count -- $S --step-ms 10"
CNN=examples/cnn_images/train.py
MLP=examples/tabular_mlp/train.py
train-doctor compare --out examples/cnn_images/runs --label stall-in-candidate-quiet --baseline "$W --counter /tmp/td-stall-h-base.count -- $CNN" --candidate "$W --counter /tmp/td-stall-h-cand.count --on-run 4 --sleep 60 -- $CNN --batch-augment"
train-doctor compare --out examples/tabular_mlp/runs --label original-vs-final-stall-quiet --baseline "$W --counter /tmp/td-stall-i-base.count -- $MLP" --candidate "$W --counter /tmp/td-stall-i-cand.count --on-run 4 --sleep 60 -- $MLP --ckpt-every 200 --num-workers 5"
# Re-analysis of every compare above (0.1.1 unpaired vs 0.1.2 paired):
python examples/reanalyze_paired.py examples/*/runs/*compare*/compare.json
