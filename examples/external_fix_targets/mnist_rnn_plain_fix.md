# train-doctor fix: mnist_rnn

Command: `/Users/abelyagubyan/Downloads/train-doctor/.venv/bin/python main.py`

**mnist_rnn: 10,365.3 -> 13,489.4 samples/s, 1.30x faster (95% interval 1.18 to 1.90), loss unchanged.**

## Proven change

- num_workers=5 (from `dataloader_workers`)
- Loss check: identical
- Speed: 1.30x (95% interval 1.18 to 1.90), measured over 5 pairs
- Evidence: `/tmp/tdext_out/mnist_rnn/20261024-050202-compare-mnist-rnn-fix-dataloader-workers/report.md`
- Patch: `/tmp/tdext_out/mnist_rnn/main.fix.patch` (apply with `git apply` or `patch -p0`; reverse with `-R`)

## Injected candidates tried

| Candidate | Decision | Ratio (95% interval) | Loss | Why |
|---|---|---|---|---|
| DataLoader workers (num_workers=5) | keep | 1.30x (1.18 to 1.90) | identical | candidate is faster (1.30x, 95% interval 1.18 to 1.90, won 5 of 5 pairs) and the loss trajectory is identical |

## Suggested, not applied

These findings need a source change fix does not make on its own:

- [persistent_workers] persistent_workers=True (risk low): If num_workers wins, persistent_workers=True also avoids re-spawning the workers each epoch. It is not applied automatically because it changes the per-epoch shuffle order (a different, statistically equivalent trajectory), which the loss check would flag.
- [cpu_bound_preprocessing] Per-sample preprocessing in MNIST is expensive (risk low): Make preprocessing cheaper instead of (or as well as) parallelizing it: precompute or cache deterministic transforms once, move per-element Python loops to vectorized tensor ops on the whole batch (or on the device), and keep only cheap random augmentation per sample.
- [cpu_with_accelerator_available] Training runs on CPU while mps is available (risk low): Move the model and each batch to torch.device('mps').
- [no_torch_compile] torch.compile is not used (risk low): Try `model = torch.compile(model)`. Compare with more warmup steps (for example --warmup-steps 20): compilation happens in the first steps.
