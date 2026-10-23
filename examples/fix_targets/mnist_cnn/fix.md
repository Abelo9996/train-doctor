# train-doctor fix: mnist_cnn

Command: `.venv/bin/python examples/fix_targets/mnist_cnn/train.py`

**mnist_cnn: 7,375.6 -> 16,000.3 samples/s, 2.16x faster (95% interval 1.46 to 2.23), loss unchanged.**

## Proven change

- num_workers=5 (from candidate `dataloader_workers`)
- Loss check: identical
- Speed: 2.16x (95% interval 1.46 to 2.23), measured over 5 pairs
- Evidence: `.train-doctor/fix_targets/20261023-135252-compare-mnist-cnn-fix-dataloader-workers/report.md`
- Patch: `.train-doctor/fix_targets/train.fix.patch` (apply with `git apply` or `patch -p0`; reverse with `-R`)

## Candidates tried

| Candidate | Decision | Ratio (95% interval) | Loss | Why |
|---|---|---|---|---|
| DataLoader workers (num_workers=5) | keep | 2.16x (1.46 to 2.23) | identical | candidate is faster (2.16x, 95% interval 1.46 to 2.23, won 5 of 5 pairs) and the loss trajectory is identical |

## Suggested, not applied

These findings need a source change fix does not make on its own:

- [persistent_workers] persistent_workers=True (risk low): If num_workers wins, persistent_workers=True also avoids re-spawning the workers each epoch. It is not applied automatically because it changes the per-epoch shuffle order (a different, statistically equivalent trajectory), which the loss check would flag.
- [cpu_bound_preprocessing] Per-sample preprocessing in SyntheticDigits is expensive (risk low): Make preprocessing cheaper instead of (or as well as) parallelizing it: precompute or cache deterministic transforms once, move per-element Python loops to vectorized tensor ops on the whole batch (or on the device), and keep only cheap random augmentation per sample.
- [no_mixed_precision] No mixed precision on hardware that supports it (risk medium): Wrap the forward pass and loss in `with torch.autocast('mps', dtype=torch.float16):`. Gains on Apple GPUs vary by op, so measure.
- [no_torch_compile] torch.compile is not used (risk low): Try `model = torch.compile(model)`. Support for MPS is newer than for CUDA and CPU; expect some models to fall back or run slower. Compare with more warmup steps (for example --warmup-steps 20): compilation happens in the first steps.
