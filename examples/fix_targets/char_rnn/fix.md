# train-doctor fix: char_rnn

Command: `.venv/bin/python examples/fix_targets/char_rnn/train.py`

**char_rnn: no proven win. No change fix can apply on its own addressed the top findings.**

## Candidates tried

None: no change fix can apply on its own addressed the top findings. See the suggestions below.

## Suggested, not applied

These findings need a source change fix does not make on its own:

- [cpu_with_accelerator_available] Training runs on CPU while mps is available (risk low): Move the model and each batch to torch.device('mps').
- [no_torch_compile] torch.compile is not used (risk low): Try `model = torch.compile(model)`. Compare with more warmup steps (for example --warmup-steps 20): compilation happens in the first steps.
