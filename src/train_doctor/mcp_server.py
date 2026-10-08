"""MCP server over stdio exposing profile, diagnose, compare and report.

Works with the official ``mcp`` Python SDK, both 1.x (FastMCP) and 2.x
(MCPServer).
"""

from __future__ import annotations

from pathlib import Path

from train_doctor import __version__, api

INSTRUCTIONS = (
    "train-doctor measures why a PyTorch (or any) training command is slow and proves speedups. "
    "Workflow: profile the command, read the ranked findings, change one thing, compare baseline and "
    "candidate, keep the change only if the decision is 'keep', then report. Every result has a "
    "'next_step' field that says what to do next. Never claim a speedup that compare did not verdict "
    "as 'faster'. Runs execute the given command locally. Use the interpreter that has the user's "
    "training dependencies (for example .venv/bin/python) when a bare 'python' is not the right one."
)

_COMMAND_DOC = (
    'The training command as an argv list, for example ["python", "train.py", "--lr", "0.1"], or as one '
    "shell-style string. It runs in `cwd` with the caller's PATH; if `python` is not the interpreter that "
    "has torch, pass its full path."
)


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP

        return FastMCP


def _abs(path: str, cwd: str) -> str:
    return str(Path(cwd, path).resolve())


def _brief_profile(res: dict) -> dict:
    ev = res["evidence"]
    out: dict = {"next_step": res["next_step"]}
    if res.get("error"):
        out["error"] = res["error"]
    return out | {
        "run_dir": res["run_dir"],
        "source": ev.get("source"),
        "device": (ev.get("config") or {}).get("device"),
        "steps": ev.get("steps"),
        "step_time_ms": ev.get("step_time_ms"),
        "throughput": ev.get("throughput"),
        "split": ev.get("split"),
        "per_step": ev.get("per_step"),
        "config": {
            k: (ev.get("config") or {}).get(k)
            for k in ("dataloaders", "autocast", "compile_calls", "params", "param_dtypes", "grad_scaler", "ddp")
        },
        "resources": ev.get("resources"),
        "findings": res["findings"],
        "limits": ev.get("limits"),
        "report": res["run_dir"] + "/report.md",
    }


def build_server():
    Server = _server_class()
    try:
        server = Server("train-doctor", instructions=INSTRUCTIONS, version=__version__)
    except TypeError:  # mcp 1.x FastMCP has no version argument
        server = Server("train-doctor", instructions=INSTRUCTIONS)

    @server.tool()
    def profile(
        command: list[str] | str,
        cwd: str = ".",
        warmup_steps: int = 5,
        steps: int = 50,
        seconds: float = 2.0,
        timeout: float = 600.0,
        profiler_steps: int = 5,
        py_spy: bool = False,
        out_dir: str = api.DEFAULT_OUT,
        label: str = "",
    ) -> dict:
        """Find out where a training command's step time goes. Call this first, on the unchanged command.

        Runs the command for a bounded window (warmup_steps, then at least `steps` steps and `seconds`
        seconds; defaults stop most scripts within a few seconds of the first step) and stops it, so it
        is safe on long jobs. Steps are detected from optimizer.step(), train_doctor.step() markers, or
        printed loss lines. Returns step-time stats, throughput, the time split (data_wait, host_sync,
        checkpoint_save, log_write, other), ranked findings (each with evidence, a suggested change,
        expected effect and risk), limits, and `next_step`. If the command fails before its first step,
        `error` holds the exit code, the last stderr lines and the fix. Next: make one change for the top
        finding, then call `compare`. Run directories are relative to `cwd` unless out_dir is absolute.
        """
        return _brief_profile(
            api.profile(
                command,
                cwd=cwd,
                warmup_steps=warmup_steps,
                steps=steps,
                seconds=seconds,
                timeout=timeout,
                profiler_steps=profiler_steps,
                py_spy=py_spy,
                out_dir=_abs(out_dir, cwd),
                label=label or None,
            )
        )

    @server.tool()
    def diagnose(run_dir: str = "", cwd: str = ".", out_dir: str = api.DEFAULT_OUT) -> dict:
        """Re-run the ranking rules on an existing profile run directory (default: the latest profile under cwd).

        `profile` already returns findings, so this is only needed to re-read an older run.
        """
        return api.diagnose(run_dir or None, _abs(out_dir, cwd))

    @server.tool()
    def compare(
        baseline: list[str] | str,
        candidate: list[str] | str,
        cwd: str = ".",
        repeats: int = 5,
        warmup_runs: int = 1,
        warmup_steps: int = 5,
        steps: int = 50,
        seconds: float = 2.0,
        seed: int = 0,
        loss_tol: float = 0.05,
        min_effect: float = 0.02,
        timeout: float = 600.0,
        out_dir: str = api.DEFAULT_OUT,
        label: str = "",
    ) -> dict:
        """Prove whether one change made training faster without changing the loss. Call after `profile`.

        baseline is the original command and candidate the changed one (argv lists or strings, same
        format as profile's `command`). Change exactly one thing between them. Runs warmup_runs discarded
        runs per command, then `repeats` measured runs each in randomized pair order with the same seed,
        and returns median throughput per arm, the candidate/baseline ratio with a 95% bootstrap interval,
        a verdict (faster, slower, no clear difference), a loss-trajectory check, a decision (keep, reject,
        inconclusive), the reason, and `next_step`. Keep a change only when decision is 'keep'. Takes
        about (repeats + warmup_runs) x 2 runs of a few seconds each.
        """
        res = api.compare(
            baseline,
            candidate,
            cwd=cwd,
            repeats=repeats,
            warmup_runs=warmup_runs,
            warmup_steps=warmup_steps,
            steps=steps,
            seconds=seconds,
            seed=seed,
            loss_tol=loss_tol,
            min_effect=min_effect,
            timeout=timeout,
            out_dir=_abs(out_dir, cwd),
            label=label or None,
        )
        head = {k: res[k] for k in ("decision", "reason", "next_step") if k in res}
        res = head | {k: v for k, v in res.items() if k != "machine" and k not in head}
        res["report"] = res["root"] + "/report.md"
        return res

    @server.tool()
    def report(run_dir: str = "", cwd: str = ".", out_dir: str = api.DEFAULT_OUT) -> dict:
        """Return the Markdown report of a profile or compare run (default: the latest run under cwd).

        Use it at the end to quote exact numbers to the user. profile and compare already write report.md.
        """
        res = api.make_report(run_dir or None, _abs(out_dir, cwd))
        with open(res["markdown"]) as f:
            res["text"] = f.read()
        return res

    @server.tool()
    def version() -> str:
        """train-doctor version."""
        return __version__

    return server


def serve() -> None:
    build_server().run()
