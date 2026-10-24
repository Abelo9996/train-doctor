# fix targets

Two sets of scripts `train-doctor fix` was validated on. The three here are **synthetic**: small plain
PyTorch scripts written for this project, each with a realistic slow spot (or none `fix` can address on
its own). The **external** validation on unmodified third-party scripts (pytorch/examples) is in
[../external_fix_targets](../external_fix_targets) and is the better test, because those scripts were
written with no knowledge of this tool. Reproduce the synthetic set with [run_validation.sh](run_validation.sh):

```bash
PATH="$PWD/.venv/bin:$PATH" sh examples/fix_targets/run_validation.sh
```

Machine: Apple M4 MacBook (10 logical cores, 16 GB), macOS 26 (Darwin 25.2), Python 3.12.13, torch 2.14.1.
`mnist_cnn` and `tabular_mlp` run on MPS; `char_rnn` runs on CPU. Numbers differ on a different machine or
under different load; `fix` measures each run and decides from the measurements, so the decision can change
too. NVIDIA paths are untested on this machine.

| Target | Device | Top finding | `fix` candidate | Decision | Result |
|---|---|---|---|---|---|
| [mnist_cnn](mnist_cnn) | MPS | `dataloader_bound` (63% of step time waiting on the DataLoader) | `num_workers=5` | **keep** | 7,376 -> 16,000 samples/s, **2.16x faster** (95% interval 1.46 to 2.23, won 5 of 5 pairs), loss identical |
| [char_rnn](char_rnn) | CPU | `cpu_with_accelerator_available` (on CPU while MPS is free) | none applicable | no proven win | the top fix is a device move, which `fix` will not make on its own; reported as a suggestion |
| [tabular_mlp](tabular_mlp) | MPS | `no_mixed_precision` (compute-bound, not data-bound) | none applicable | no proven win | the loop is compute-bound; the suggested fixes (autocast, torch.compile) need a source change |

Per target, `fix` wrote a `fix.md` (the report above, verbatim). For `mnist_cnn` it also wrote the
reversible patch it proved ([mnist_cnn.fix.patch](mnist_cnn/mnist_cnn.fix.patch): `num_workers=args.num_workers`
-> `num_workers=5` on the DataLoader) and the result card ([PNG](mnist_cnn/mnist_cnn.card.png),
[SVG](mnist_cnn/mnist_cnn.card.svg)). The raw profiles, compares, per-pair numbers and traces land under
`.train-doctor/fix_targets/` (git-ignored); the report's `Evidence:` line points at the compare's
`report.md` there.

## What each result shows

- **mnist_cnn** is the headline path end to end: a data-loading-bound loop (the dataset generates and
  blurs each image in `__getitem__`), one applicable change, a clear 2.16x with the loss trajectory byte
  for byte identical (the preprocessing is deterministic and the DataLoader shuffle is seeded, so moving
  the work to 5 worker processes produces the same batches), a patch, and a card. `persistent_workers=True`
  is reported as a suggestion, not applied: it avoids re-spawning workers each epoch but changes the
  per-epoch shuffle order, which the loss check would flag.
- **char_rnn** and **tabular_mlp** are the honest other half. Neither is data-loading bound, so there is no
  change `fix` applies on its own, and it says "no proven win" rather than running a comparison it would
  have to report as inconclusive. It still lists the real findings (move to MPS; mixed precision;
  torch.compile) as suggestions, each needing a source edit a person should make and measure with
  `compare`.
