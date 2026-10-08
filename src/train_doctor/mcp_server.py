"""MCP server over stdio exposing profile, diagnose, compare and report.

Works with the official ``mcp`` Python SDK, both 1.x (FastMCP) and 2.x
(MCPServer).
"""

from __future__ import annotations

from train_doctor import __version__, api

INSTRUCTIONS = (
    "train-doctor measures why a PyTorch (or any) training command is slow and proves speedups. "
    "Workflow: profile the command, read the ranked findings, change one thing, compare baseline and "
    "candidate, keep the change only if the decision is 'keep', then report. Never claim a speedup "
    "that compare did not verdict as 'faster'. Runs execute the given command locally."
)


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP

        return FastMCP


def _brief_profile(res: dict) -> dict:
    ev = res["evidence"]
    return {
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
        command: list[str],
        cwd: str = ".",
        warmup_steps: int = 5,
        steps: int = 50,
        seconds: float = 2.0,
        timeout: float = 600.0,
        profiler_steps: int = 5,
        py_spy: bool = False,
        out_dir: str = api.DEFAULT_OUT,
    ) -> dict:
        """Run a training command for a bounded window and collect evidence, then diagnose it.

        command is the argv list, for example ["python", "train.py", "--lr", "0.1"]. The run stops
        after warmup_steps plus at least `steps` steps and `seconds` seconds. Steps are detected from
        optimizer.step() (or train_doctor.step() markers). Returns step-time stats, throughput, the
        time split (data wait, host sync, checkpoint, logging, other), ranked findings and limits.
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
                out_dir=out_dir,
            )
        )

    @server.tool()
    def diagnose(run_dir: str = "", out_dir: str = api.DEFAULT_OUT) -> dict:
        """Re-run the deterministic rules on a profile run directory (default: the latest profile)."""
        return api.diagnose(run_dir or None, out_dir)

    @server.tool()
    def compare(
        baseline: list[str],
        candidate: list[str],
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
    ) -> dict:
        """Benchmark baseline vs candidate commands (argv lists) with repeats in randomized pair order.

        Returns median throughput and spread per arm, the candidate/baseline ratio with a 95% bootstrap
        interval, a verdict (faster, slower, no clear difference), a loss-trajectory check, and a
        decision: keep, reject or inconclusive. Keep a change only when the decision is 'keep'.
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
            out_dir=out_dir,
        )
        res = {k: v for k, v in res.items() if k != "machine"}
        res["report"] = res["root"] + "/report.md"
        return res

    @server.tool()
    def report(run_dir: str = "", out_dir: str = api.DEFAULT_OUT) -> dict:
        """Write report.md and report.json for a profile or compare run (default: latest) and return the Markdown."""
        res = api.make_report(run_dir or None, out_dir)
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
