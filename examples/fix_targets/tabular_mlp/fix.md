# train-doctor fix: tabular_mlp

Command: `.venv/bin/python examples/fix_targets/tabular_mlp/train.py`

**tabular_mlp: no proven win. No change fix can apply on its own addressed the top findings.**

## Candidates tried

None: no change fix can apply on its own addressed the top findings. See the suggestions below.

## Suggested, not applied

These findings need a source change fix does not make on its own:

- [no_mixed_precision] No mixed precision on hardware that supports it (risk medium): Wrap the forward pass and loss in `with torch.autocast('mps', dtype=torch.float16):`. Gains on Apple GPUs vary by op, so measure.
- [no_torch_compile] torch.compile is not used (risk low): Try `model = torch.compile(model)`. Support for MPS is newer than for CUDA and CPU; expect some models to fall back or run slower. Compare with more warmup steps (for example --warmup-steps 20): compilation happens in the first steps.
