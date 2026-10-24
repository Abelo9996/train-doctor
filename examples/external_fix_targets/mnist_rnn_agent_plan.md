# train-doctor fix: mnist_rnn

Command: `/Users/abelyagubyan/Downloads/train-doctor/.venv/bin/python main.py`

**mnist_rnn: plan only (no spend). Would try 1 injected change(s) and ask claude for 2 source change(s), one at a time, each proven with a paired compare. Re-run with --yes to spend.**

Cost: 2 headless claude session(s) (roughly a few cents to about $0.30 each, model dependent) plus 4 paired compare(s), which have no agent cost. Nothing was run.

## Would inject (no agent)

- num_workers=5

## Would ask claude to make, one at a time

- [torch_compile] torch.compile(model) (from `no_torch_compile`, risk low): after the model is created and moved to its device, compile it with `model = torch.compile(model)` and train with the compiled model.
- [device_move] move training to mps (from `cpu_with_accelerator_available`, risk low): change the device selection so the model and every input batch are moved to torch.device('mps') instead of cpu.

Re-run with `--yes` to spend.