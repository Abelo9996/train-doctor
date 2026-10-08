# train-doctor

Give your agent a way to find out why your training run is slow, fix it, and prove the speedup.

```bash
uvx train-doctor profile -- python train.py        # where does step time go?
uvx train-doctor compare --baseline "python train.py" --candidate "python train.py --num-workers 4"
uvx train-doctor setup                             # give Claude Code, Codex and Cursor the MCP tools
```

> Not on PyPI yet. Until the first release, run it straight from GitHub by replacing `uvx train-doctor` with
> `uvx --from git+https://github.com/Abelo9996/train-doctor train-doctor`. `setup` registers `uvx train-doctor mcp`, so it works once the
> package is on PyPI.

## Example output

From a real run in this repo: [examples/cnn_images](examples/cnn_images), a small CNN written the way many
first scripts are (per-sample augmentation in `__getitem__`, `num_workers=0`, `.item()` and a print on every
step). Apple M4 (10 cores, 16 GB), torch 2.14.1 on MPS, with other jobs running on the machine (1-minute
load average between 4.6 and 10.4 across all runs).

Profile output (trimmed where marked `...`):

```text
$ train-doctor profile -- python examples/cnn_images/train.py
source: hook, device: mps, steps measured: 87
step time: median 20.88 ms (p10 14.32, p90 32.84)
throughput: 2766.8 samples/s
split: data_wait 39.1%, host_sync 26.4%, checkpoint_save 0.0%, log_write 0.1%, other 34.4%
findings:
  1. [dataloader_bound] Training waits on the DataLoader (num_workers=0) (score 0.391, risk low)
  2. [cpu_bound_preprocessing] Per-sample preprocessing in SyntheticImages is expensive (score 0.319, risk low)
     evidence: 39% of step time is data wait and 91% of that is inside SyntheticImages.__getitem__ (0.13 ms per sample).
  3. [host_sync_in_loop] Host waits on the device inside every step (score 0.132, risk none)
     evidence: 2.0 blocking reads of mps tensors per step (...), 6.1 ms per step blocked in them (26% of step time).
  4. [no_mixed_precision] ...  5. [logging_in_hot_loop] ...  6. [no_torch_compile] ...
```

Then one change at a time, each checked with `train-doctor compare` (5 measured repeats per side after one
discarded warmup run, randomized order, 95% bootstrap interval on the ratio of median samples/s):

| Change | Median samples/s, before to after | Ratio (95% interval) | Loss check | Decision |
|---|---|---|---|---|
| `num_workers=5` | 2,249 to 2,750 | 1.22x (0.66 to 1.68) | within tolerance | inconclusive, dropped |
| crop/flip/normalize per batch on the device | 2,232 to 2,855 | 1.28x (1.10 to 1.35) | within tolerance | keep |
| read metrics every 50 steps instead of every step | 2,420 to 3,029 | 1.25x (1.11 to 1.35) | identical | keep |
| autocast float16 | 2,648 to 2,842 | 1.07x (0.81 to 1.28) | within tolerance | inconclusive, dropped |
| original vs the two kept changes | 2,091 to 3,381 | 1.62x (1.37 to 3.48) | within tolerance | keep |

Two of the four candidates looked faster by their medians and were still not kept, because their intervals
included 1.0. In the second example ([examples/tabular_mlp](examples/tabular_mlp)), checkpointing less often
was inconclusive with 5 repeats (1.54x, interval 0.62 to 2.67) and clearly faster with 9 (2.01x, 1.54 to 2.52);
original vs final there is 3.05x (2.15 to 3.40) with an identical loss trajectory. Every report, per-repeat
number and raw trace is committed under `examples/*/runs/`, and
[examples/run_validation.sh](examples/run_validation.sh) is the exact command sequence.

## How it works

A "2x faster" claim is only useful if you can trust it. train-doctor excludes warmup, brackets the
measured window with a device synchronize, runs baseline and candidate several times in randomized
interleaved order with the same seed, reports medians, spread and a bootstrap interval, and calls a
change "faster" only when the whole interval clears a threshold. It also checks that the loss
trajectory didn't move, so a speedup that broke training is rejected.

train-doctor runs your real training command as a child process and stops it after a bounded window
(warmup steps, then at least N steps and T seconds; defaults 5, 50 and 2 s). No code changes are needed
for PyTorch:

- **Step recorder.** A `sitecustomize.py` on the child's PYTHONPATH loads a small standard-library
  recorder that patches torch right after it is imported. It marks a step at every `optimizer.step()`
  (a global post-hook), times `DataLoader.__next__`, times blocking reads of GPU/MPS tensors
  (`.item()`, `.cpu()`, `.tolist()`, `float()`, `bool()`, `synchronize()`), times `torch.save` and
  stdout/stderr writes, counts `backward()` calls, and keeps the loss tensor passed to `backward()`
  (converted to a number only at the end, so recording it adds no sync). It also notes DataLoader
  settings, autocast, GradScaler, `torch.compile`, DDP, parameter count and dtype. Any other
  `sitecustomize` you have still runs. Loops without an optimizer (or other frameworks) can call
  `train_doctor.step(samples=..., loss=...)`, or just print a line containing `loss` each step.
- **torch.profiler** traces a few extra steps after the measured window (so it doesn't distort the
  timings) and saves a Chrome trace you can open in Perfetto.
- **py-spy** (optional, `--py-spy`) samples Python stacks for any framework when it is installed.
- **System telemetry**: psutil for CPU and memory of the whole process tree (including DataLoader
  workers), `nvidia-smi` for NVIDIA GPUs, and `ioreg` for Apple GPU utilization (no sudo needed).
- **Diagnosis** is a fixed set of rules over that evidence: DataLoader-bound, expensive per-sample
  preprocessing, pin_memory off on CUDA, host syncs in the loop, small batch on an idle device, no
  mixed precision on capable hardware, no torch.compile, checkpoints or logging in the hot loop,
  gradient accumulation (including DDP without `no_sync()`), cuDNN autotuning off, and training on CPU
  while an accelerator is available. Each finding carries the numbers it fired on, a suggested change,
  the expected effect (an upper bound where one can be computed) and the risk to results.
- **Compare** runs one discarded warmup run per command, then K repeats per command (default 5) in
  randomized pair order. Throughput is samples/s over the synchronized window. The verdict uses a 95%
  percentile bootstrap interval on the ratio of medians: "faster" only if the whole interval is above
  1 + min-effect (default 2%), "slower" if it's below 1 - min-effect, otherwise "no clear difference".
  The loss check aligns both loss curves on samples seen, smooths them, and requires the mean and final
  relative difference to stay within a tolerance (default 5%). The decision is `keep` only when both pass.
- **Reports**: every run directory gets `report.md` and `report.json` with the machine, versions,
  commands, file hashes, git commit, per-repeat numbers, findings, verdict and limits, plus the raw
  files (per-step events, resource samples, stdout and stderr, profiler output).

The tool is deterministic and never calls an LLM. Your coding agent reads the findings, makes one
change, and uses `compare` to decide whether to keep it.

## Setup for agents

```bash
uvx train-doctor setup          # shows what it would change
uvx train-doctor setup --yes    # applies it
```

`setup` detects Claude Code, Codex and Cursor. It runs `claude mcp add --scope user train-doctor -- uvx train-doctor mcp`,
adds `[mcp_servers.train-doctor]` to `~/.codex/config.toml`, adds the server to `~/.cursor/mcp.json`, and copies the
skill to `~/.claude/skills/train-doctor/` and `~/.codex/skills/train-doctor/`. It backs up every file it edits and
does nothing when an entry is already there. `--project DIR` also writes a project `.mcp.json`.

MCP tools: `profile`, `diagnose`, `compare`, `report`, `version`. The skill
([skills/train-doctor/SKILL.md](skills/train-doctor/SKILL.md)) tells the agent the workflow: profile, read
the findings, change one thing, compare, keep only clear wins that don't move the loss, report.

CLI: `train-doctor profile|diagnose|compare|report|setup|mcp`, each with `--help` and `--json`.

## What it can't do

- It doesn't edit your code. The agent (or you) makes the change; train-doctor measures it.
- Timings come from the host side. On CUDA and MPS, time inside a sync call includes waiting for device
  work that has to happen anyway, so the "host sync" share is an upper bound on what removing syncs saves.
- Apple GPU utilization from `ioreg` is system-wide, not per process. There is no per-kernel timing on MPS.
- `__getitem__` time is measured only with `num_workers=0`; with workers it runs in other processes.
- py-spy needs root on macOS, so `--py-spy` usually fails there. It was only tested with a fake py-spy.
- CUDA support (nvidia-smi parsing, CUDA sync and memory paths, CUDA-specific rules) is implemented and
  tested with fakes. It has not been run on an NVIDIA GPU yet. The real runs in this repo are on Apple MPS.
- Multi-GPU: with DDP it reads the process that recorded the most steps (usually rank 0); it doesn't merge ranks.
- The loss check covers the first steps only. It catches changes that break or shift training early; it
  can't tell you about final accuracy. Batch size and learning-rate changes are flagged as high risk for that reason.
- Short windows on a busy machine are noisy. Reports record the load average and system CPU per run; when
  the spread is high, compare says so and is more likely to answer "no clear difference".

## Privacy and safety

Everything runs locally. train-doctor makes no network requests and sends nothing anywhere. It runs the
command you give it with your permissions, and writes to `.train-doctor/runs/` (or `--out`). Reports
replace your home directory with `~`. `setup` changes only the files it lists, and only after `--yes`.

## License

MIT, see [LICENSE](LICENSE).
