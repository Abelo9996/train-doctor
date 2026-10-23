# train-doctor

Find out why your training run is slow, fix it, and prove the speedup, in one command.

```bash
uvx train-doctor fix -- python train.py            # profile, try the fixes, prove the winner
uvx train-doctor profile -- python train.py        # or just: where does step time go?
uvx train-doctor compare --baseline "python train.py" --candidate "python train_fixed.py"  # did a change help?
uvx train-doctor setup --agent claude              # give your coding agent the MCP tools
```

`fix` profiles the command, turns the top findings into the changes it can apply on its own, measures
each one against the original with a paired, noise-aware benchmark, keeps only the clear winners, checks
the loss trajectory is unchanged, and ends with "before -> after, Nx faster, loss unchanged" plus a
reversible patch and a result card. If nothing clears the bar it says "no proven win" and lists what it
tried. It never reports a speedup the benchmark did not confirm.

Run it from the environment that has your training dependencies (an activated venv or conda env), or
give the interpreter's full path: `uvx train-doctor fix -- .venv/bin/python train.py`. A bare `python`
is resolved to the target repo's `.venv`/`venv` or poetry interpreter automatically. train-doctor
doesn't install torch; it runs your command with your interpreter. It needs [uv](https://docs.astral.sh/uv/)
for `uvx` (or `pip install train-doctor`; add `[card]` for PNG cards). A profile takes a few seconds past
your script's startup; a full `fix` takes a few minutes.

## What `fix` can prove, and what it only suggests

`fix` applies a change by injecting it into the candidate run, so the baseline and candidate are the same
unchanged command and only the injected setting differs. That keeps the set of automatic changes small
and safe:

| Change | Applied by `fix` | Notes |
|---|---|---|
| DataLoader `num_workers` | yes | Loss-neutral: with `persistent_workers=False` the shuffle RNG advances as single-process. |
| `pin_memory`, cuDNN autotuning | yes, on CUDA | NVIDIA paths are untested on the author's hardware (Apple M4 / MPS and CPU). |
| `persistent_workers` | suggested | Avoids re-spawning workers per epoch but changes the per-epoch shuffle order. |
| device move, mixed precision, `torch.compile`, batch size, removing per-step `.item()`/sync points | suggested | Need a source change `fix` will not make blindly; it reports them with evidence so you can decide. |

Everything `fix` applies is also written as a reversible unified diff against your script (applied with
`--keep`, otherwise left as a patch file).

## `fix` on a real script

A real run from this repo: [examples/fix_targets/mnist_cnn](examples/fix_targets/mnist_cnn), a small
MNIST-style CNN whose dataset generates and blurs each image in `__getitem__` with `num_workers=0`. Apple
M4 (10 cores, 16 GB), torch 2.14.1 on MPS.

![train-doctor fix on examples/fix_targets/mnist_cnn: 2.16x faster, loss unchanged](https://raw.githubusercontent.com/Abelo9996/train-doctor/main/examples/fix_targets/mnist_cnn/mnist_cnn.card.png)

```text
$ train-doctor fix -- .venv/bin/python examples/fix_targets/mnist_cnn/train.py
Profiling .venv/bin/python examples/fix_targets/mnist_cnn/train.py
Trying DataLoader workers (num_workers=5)
  [3/12] pair 1 of 5, baseline: 7337.8 samples/s in 4.5 s, about 52 s left
  [4/12] pair 1 of 5, candidate: 16009.1 samples/s in 8.5 s, about 51 s left
  ...
  DataLoader workers (num_workers=5): keep (candidate is faster (2.16x, 95% interval 1.46 to 2.23,
  won 5 of 5 pairs) and the loss trajectory is identical)

mnist_cnn: 7,375.6 -> 16,000.3 samples/s, 2.16x faster (95% interval 1.46 to 2.23), loss unchanged.

proven change: num_workers=5
patch: .train-doctor/fix_targets/train.fix.patch (apply: git apply <patch>; reverse: git apply -R <patch>)
card: mnist_cnn.card.png and mnist_cnn.card.svg
```

The proof is a paired comparison: five times it ran the unchanged command and the same command with the
workers injected, back to back, and the candidate won every pair with the loss byte-for-byte identical.
The patch it wrote changes one line (`num_workers=args.num_workers` -> `num_workers=5`). On two other
scripts in [examples/fix_targets](examples/fix_targets) (a char-level LSTM on CPU, a compute-bound tabular
MLP) `fix` found no change it could apply on its own and said "no proven win", with the real findings
listed as suggestions, instead of reporting a comparison it would have to call inconclusive.

## Example output

![train-doctor profile on examples/cnn_images](https://raw.githubusercontent.com/Abelo9996/train-doctor/main/docs/profile.gif)

The recording above is a real 0.1.1 run of `train-doctor profile` on [examples/cnn_images](examples/cnn_images)
(made with [vhs](https://github.com/charmbracelet/vhs) from [docs/profile.tape](docs/profile.tape)), on a
shared machine, so its numbers differ from the run below.

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

Then one change at a time, each checked with `train-doctor compare` (one discarded warmup run per side,
then 5 pairs of baseline and candidate run back to back). These runs were made with 0.1.1, which compared
the two sides' medians as independent samples; the last column is the same runs analyzed pair by pair the
way 0.1.2 does it ([examples/reanalyze_paired.py](examples/reanalyze_paired.py)):

| Change | Median samples/s, before to after | Loss check | 0.1.1: ratio of medians (95% interval), decision | 0.1.2: median pair ratio (95% interval), pairs won |
|---|---|---|---|---|
| `num_workers=5` | 2,249 to 2,750 | within tolerance | 1.22x (0.66 to 1.68), inconclusive | 1.09x (0.76 to 1.41), 3 of 5, inconclusive |
| crop/flip/normalize per batch on the device | 2,232 to 2,855 | within tolerance | 1.28x (1.10 to 1.35), keep | 1.23x (1.07 to 1.37), 5 of 5, keep |
| read metrics every 50 steps instead of every step | 2,420 to 3,029 | identical | 1.25x (1.11 to 1.35), keep | 1.25x (1.00 to 1.37), 5 of 5, inconclusive |
| autocast float16 | 2,648 to 2,842 | within tolerance | 1.07x (0.81 to 1.28), inconclusive | 1.07x (0.90 to 1.15), 4 of 5, inconclusive |
| original vs the two kept changes | 2,091 to 3,381 | within tolerance | 1.62x (1.37 to 3.48), keep | 1.69x (1.26 to 1.81), 4 of 4 after setting aside 1 stalled pair, keep |

Two of the four candidates looked faster by their medians and were still not kept, because their intervals
included 1.0. The paired analysis is stricter about ordinary noise: "read metrics every 50 steps" won all 5
pairs, but one pair by only 1.00x, and with 5 pairs `faster` needs every pair to clear the 2% threshold. In
the second example ([examples/tabular_mlp](examples/tabular_mlp)), checkpointing less often was inconclusive
under 0.1.1 with 5 repeats (1.54x, interval 0.62 to 2.67) because one candidate run was 2.5 times slower than
that side's median; the paired analysis sets that pair aside and says faster (1.96x, 1.36 to 2.51), which matches
the 9-repeat rerun (0.1.1: 2.01x, 1.54 to 2.52; paired: 2.18x, 1.54 to 2.44, 9 of 9 pairs). Original vs
final there is 3.05x (2.15 to 3.40) under 0.1.1 and 2.80x (2.15 to 3.33) paired, with an identical loss
trajectory. Every report, per-repeat number and raw trace is committed under `examples/*/runs/`, and
[examples/run_validation.sh](examples/run_validation.sh) is the exact command sequence.

### One stalled run

A laptop that hiccups (another job, swapping, a sleep) can make one run 30 to 100 times slower. To test
that, [examples/stall_once.py](examples/stall_once.py) froze one candidate run for 60 s inside its measured
window, in a compare of the tabular example's original vs final commands (5 pairs, Apple M4, load average
4.0 to 7.2 at run starts; numbers are rough on a shared machine):

```text
$ train-doctor compare --baseline "$W ... -- $MLP" --candidate "$W ... --on-run 4 --sleep 60 -- $MLP --ckpt-every 200 --num-workers 5"
[8/12] pair 3 of 5, candidate: 211.3 samples/s in 94.5 s, about 111 s left
...
decision: keep: candidate is faster (2.67x, 95% interval 2.28 to 3.66, won 4 of 4 pairs) and the loss trajectory is identical
stalled pairs set aside: 2 (with them the interval would be 0.022 to 3.656)
limit: Set aside 1 stalled pair(s) out of 5: candidate repeat 2 ran at 211.3 samples/s, 158.5 times slower than that arm's median 3.35e+04. ...
```

0.1.1's unpaired analysis of the same runs gives 0.02 to 3.66 and "no clear difference". With a stall in
a baseline run of a loop whose speedup is known (2x by construction), 0.1.1 still said faster but with an
interval of 1.31 to 97.46; paired it is 1.79x (1.30 to 1.94). All stall runs, including five on a heavily
loaded machine where other slow pairs kept the answer inconclusive, are in
[examples/README.md](examples/README.md#one-stalled-run-012).

## Use it with your agent

Run `uvx train-doctor setup --agent claude --yes` (or `--agent codex` / `--agent cursor`; without `--agent`
it only prints what it would do and touches no agent config), open your training project in Claude Code (or
Codex or Cursor) and type what you'd say to a colleague:

```text
train.py is way slower than it should be. Can you figure out why, make it faster, and show me the
speedup is real and didn't change the training?
```

That exact prompt was given to a headless Claude Code session (`claude -p`, train-doctor 0.1.1 as its
only MCP server, the skill as project instructions) on a copy of
[examples/cnn_images/train.py](examples/cnn_images/train.py). Other heavy jobs were running on the same
Apple M4 at the time (the 1-minute load average reached 45 on 10 cores). Excerpts, with tool calls
shortened and text verbatim:

```text
profile(["python", "train.py"])
  next_step: Make one change for [dataloader_bound] ...
agent: Top finding: data loading (35% of step time, num_workers=0). Comparing `--num-workers 5`.
compare(... "--num-workers", "5")
  decision: inconclusive
  reason: no clear difference: the 95% interval 0.11 to 22.99 does not clear the 2% threshold on either side
agent: Other processes I don't own (`ugrep`, another Python) are using CPU, so I'll leave them alone
       and rerun with more repeats and a longer window, as the tool suggests.
...
compare(..., repeats=17, seconds=4)
  reason: candidate is faster (1.16x, 95% interval 1.05 to 1.55) and the loss trajectory is within tolerance
agent: The 1.16x is the confirmed figure, and 1.05x is the most I'd promise.
```

The 0.11 to 22.99 interval in the first compare is what one stalled run did to 0.1.1's unpaired
analysis; 0.1.2 pairs the runs and sets stalls aside (see "One stalled run" below).
It also tried logging every 50 steps and float16 autocast on top, got "no clear difference" for both, and
dropped them. Two honest notes from reading the transcript. The workers change was only kept on the third
try (5, then 9, then 17 repeats), and rerunning until a comparison says `keep` inflates false wins, so
0.1.1 no longer suggests more than 9 repeats and tells the agent to rerun later when the machine was busy.
And an earlier session with 0.1.0 found a real bug: under `uvx`, a bare `python` in the command started
train-doctor's own interpreter, which has no torch. The agent worked around it ("The MCP server's
`python` resolved to an interpreter without torch; I'll use the venv interpreter explicitly"); 0.1.1 fixes it.

## How it works

A "2x faster" claim is only useful if you can trust it. train-doctor excludes warmup, brackets the
measured window with a device synchronize, runs baseline and candidate as back-to-back pairs with the
same seed, compares them pair by pair, sets aside pairs where the machine stalled (and says so), and
calls a change "faster" only when the whole interval clears a threshold. It also checks that the loss
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
- **Compare** runs one discarded warmup run per command, then K pairs (default 5): baseline and
  candidate back to back, with the order alternating between pairs so slow drift cancels out.
  Throughput is samples/s over the synchronized window, and each pair gives a candidate/baseline
  ratio. A run is a stall when it is under half its own arm's median and beyond 4 scaled MADs, or under
  a quarter of the median however noisy the arm is; pairs with a stalled run are set aside, at most 1
  in 4 pairs, and listed in the result with the interval you'd get with them included. The verdict
  uses a 95% percentile bootstrap interval of the median pair ratio: "faster" only if the whole
  interval is above 1 + min-effect (default 2%), "slower" if it's below 1 - min-effect, otherwise "no
  clear difference". The result also has the number of pairs the candidate won and an exact Wilcoxon
  signed-rank test. A progress line per run goes to stderr (MCP: a progress notification).
  The loss check aligns both loss curves on samples seen, smooths them, and requires the mean and final
  relative difference to stay within a tolerance (default 5%). The decision is `keep` only when both pass.
- **Reports**: every run directory gets `report.md` and `report.json` with the machine, versions,
  commands, file hashes, git commit, per-repeat numbers, findings, verdict and limits, plus the raw
  files (per-step events, resource samples, stdout and stderr, profiler output).

The tool is deterministic and never calls an LLM. Your coding agent reads the findings, makes one
change, and uses `compare` to decide whether to keep it.

## Setup for agents

```bash
uvx train-doctor setup --agent claude          # shows what it would change for Claude Code
uvx train-doctor setup --agent claude --yes    # applies it
```

`setup` registers the MCP server and skill with the agent(s) you name with `--agent`
(`claude`, `codex`, `cursor`; repeatable) and nothing else: it does not touch `~/.claude`, `~/.codex` or
`~/.cursor` unless that agent is named, so running it never changes an agent's config by surprise. For
Claude Code it runs `claude mcp add --scope user train-doctor -- uvx train-doctor mcp` and copies the skill
to `~/.claude/skills/train-doctor/`; for Codex it adds `[mcp_servers.train-doctor]` to `~/.codex/config.toml`
and the skill to `~/.codex/skills/`; for Cursor it adds the server to `~/.cursor/mcp.json`. It backs up
every file it edits and does nothing when an entry is already there. `--project DIR` writes a project
`.mcp.json` and needs no agent.

MCP tools: `profile`, `diagnose`, `compare`, `report`, `version`. Every `profile` and `compare` result
has a `next_step` that says what to do next, and a command that fails before its first step returns an
`error` with the last stderr lines and the fix. The skill
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
- With 5 or fewer pairs the "95%" bootstrap interval is the range of the pair ratios, which covers the
  true median ratio with probability 1 - 2/2^n (93.75% for 5 pairs, 87.5% for 4, the case after one stall
  is set aside), and the result says so. Fewer than 4 usable pairs never gets a faster or slower verdict
  (a macOS CI runner called an identical command faster from 3 pairs). One ordinary slow pair, not slow
  enough to count as a stall, makes the answer "no clear difference". Use 9 repeats for small effects.
- Stall handling is a rule with fixed thresholds, not a model of your machine. A slowdown that hits both
  runs of a pair equally cancels out; one that hits a single run by less than 2x stays in and widens
  the interval. Two stalls in 5 pairs is more than it will set aside. A stall in a candidate run is set
  aside like any other, so if the change itself causes occasional stalls (recompiling, starting
  workers), only a rerun will show it; the result says so when that happens.

## Privacy and safety

Everything runs locally. train-doctor makes no network requests and sends nothing anywhere. It runs the
command you give it with your permissions, and writes to `.train-doctor/runs/` (or `--out`). Reports
replace your home directory with `~`. `setup` changes only the files it lists, and only after `--yes`.

## License

MIT, see [LICENSE](LICENSE).
