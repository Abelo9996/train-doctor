# train-doctor fix: mnist

Command: `/Users/abelyagubyan/Downloads/train-doctor/.venv/bin/python main.py`

**mnist: no proven win. (torch_compile: reject)**

## Injected candidates tried

None applicable: the top findings need source changes.

## Source changes by claude (one at a time)

| Candidate | Decision | Ratio (95% interval) | Loss | Why |
|---|---|---|---|---|
| torch.compile(model) | reject | 1.22x (1.12 to 1.23) | outside tolerance | candidate is faster (1.22x) but the loss trajectory is outside the 5% tolerance (mean relative difference 9.0%) |

## Suggested, not applied

These findings need a source change fix does not make on its own:

- [no_mixed_precision] No mixed precision on hardware that supports it (risk medium): Wrap the forward pass and loss in `with torch.autocast('mps', dtype=torch.float16):`. Gains on Apple GPUs vary by op, so measure.
- [no_torch_compile] torch.compile is not used (risk low): Try `model = torch.compile(model)`. Support for MPS is newer than for CUDA and CPU; expect some models to fall back or run slower. Compare with more warmup steps (for example --warmup-steps 20): compilation happens in the first steps.
