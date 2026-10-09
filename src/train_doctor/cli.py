"""Command line interface: profile, diagnose, compare, report, setup, mcp."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from train_doctor import __version__, api


def _add_window(p: argparse.ArgumentParser) -> None:
    p.add_argument("--warmup-steps", type=int, default=5, help="steps excluded from timing at the start of each run (default 5, minimum 1)")
    p.add_argument("--steps", type=int, default=50, help="measure at least this many steps per run (default 50)")
    p.add_argument("--seconds", type=float, default=2.0, help="and at least this many seconds (default 2; 0 to go by steps only)")
    p.add_argument("--timeout", type=float, default=600.0, help="hard limit per run in seconds (default 600)")
    p.add_argument("--cwd", default=".", help="working directory for the command")
    p.add_argument("--out", default=api.DEFAULT_OUT, help=f"where run directories go (default {api.DEFAULT_OUT})")
    p.add_argument("--echo", action="store_true", help="stream the command's output to stderr")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p.add_argument("--label", default=None, help="short name added to the run directory and report title")


def _strip_dashdash(cmd: list[str]) -> list[str]:
    return cmd[1:] if cmd and cmd[0] == "--" else cmd


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="train-doctor",
        description="Find out why a training run is slow, fix it, and prove the speedup.",
    )
    ap.add_argument("--version", action="version", version=f"train-doctor {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "profile",
        help="run a training command for a bounded window and collect evidence",
        description="Run a training command for a bounded window and collect evidence. Example: train-doctor profile -- python train.py",
    )
    _add_window(p)
    p.add_argument(
        "--profiler-steps",
        type=int,
        default=5,
        help="extra steps traced with torch.profiler after the measured window (0 disables, default 5)",
    )
    p.add_argument("--py-spy", action="store_true", help="also sample Python stacks with py-spy if it's installed (needs root on macOS)")
    p.add_argument("--seed", type=int, default=None, help="seed torch, random and numpy at import time")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="the training command, after --")

    p = sub.add_parser("diagnose", help="turn a profile's evidence into ranked findings")
    p.add_argument("run_dir", nargs="?", help="profile run directory (default: latest)")
    p.add_argument("--out", default=api.DEFAULT_OUT)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser(
        "compare",
        help="benchmark a baseline and a candidate command in back-to-back pairs and give a verdict",
        description=(
            "Run the baseline and candidate commands as back-to-back pairs (order alternating), compare throughput pair by pair, "
            "set aside pairs where the machine stalled, and decide keep, reject or inconclusive. Progress goes to stderr."
        ),
    )
    p.add_argument("--baseline", required=True, help="baseline command, quoted")
    p.add_argument("--candidate", required=True, help="candidate command, quoted")
    p.add_argument("--repeats", type=int, default=5, help="measured runs per command (default 5)")
    p.add_argument("--warmup-runs", type=int, default=1, help="discarded runs per command before measuring (default 1)")
    p.add_argument("--seed", type=int, default=0, help="seed set in both commands before their own seeding (default 0)")
    p.add_argument("--loss-tol", type=float, default=0.05, help="allowed relative loss difference, mean and final (default 0.05)")
    p.add_argument("--min-effect", type=float, default=0.02, help="smallest speed change worth calling, as a fraction (default 0.02)")
    p.add_argument(
        "--order-seed", type=int, default=0, help="picks which command runs first in the first pair; the order then alternates (default 0)"
    )
    p.add_argument("--quiet", action="store_true", help="don't print a progress line to stderr after each run")
    _add_window(p)

    p = sub.add_parser("report", help="write report.md and report.json for a run directory")
    p.add_argument("run_dir", nargs="?", help="profile or compare run directory (default: latest)")
    p.add_argument("--out", default=api.DEFAULT_OUT)
    p.add_argument("--json", action="store_true", help="print report.json instead of report.md")

    p = sub.add_parser("setup", help="register the MCP server and skill with Claude Code, Codex and Cursor")
    p.add_argument("--yes", action="store_true", help="apply the changes (otherwise only show them)")
    p.add_argument("--project", default=None, help="also write a project .mcp.json in this directory")
    p.add_argument("--command", dest="server_command", default=None, help="server command to register (default: uvx train-doctor mcp)")

    sub.add_parser("mcp", help="run the MCP server over stdio")
    return ap


def _print_failure(err: dict) -> None:
    print(f"error: the command exited with code {err.get('exit_code')} before its first measured step (ran {err.get('executable')})")
    if err.get("stderr_tail"):
        print("stderr (last lines):")
        for line in err["stderr_tail"].splitlines():
            print(f"  | {line}")
    print(f"fix: {err.get('hint')}")


def _print_profile(res: dict) -> None:
    ev = res["evidence"]
    if res.get("error"):
        print(f"run: {res['run_dir']}")
        _print_failure(res["error"])
        return
    tp = ev.get("throughput") or {}
    stm = ev.get("step_time_ms") or {}
    print(f"run: {res['run_dir']}")
    print(
        f"source: {ev.get('source')}, device: {(ev.get('config') or {}).get('device')}, steps measured: {(ev.get('steps') or {}).get('measured')}"
    )
    if stm.get("n"):
        print(f"step time: median {stm['median']:.2f} ms (p10 {stm['p10']:.2f}, p90 {stm['p90']:.2f})")
    if tp.get("samples_per_s"):
        print(f"throughput: {tp['samples_per_s']:.1f} samples/s")
    elif tp.get("steps_per_s"):
        print(f"throughput: {tp['steps_per_s']:.2f} steps/s")
    split = ev.get("split")
    if split:
        print(
            "split: " + ", ".join(f"{k} {split[k] * 100:.1f}%" for k in ("data_wait", "host_sync", "checkpoint_save", "log_write", "other"))
        )
    _print_findings(res["findings"])
    for lim in ev.get("limits") or []:
        print(f"limit: {lim}")
    print(f"report: {Path(res['run_dir']) / 'report.md'}")
    print(f"next: {res['next_step']}")


def _print_findings(findings: list[dict]) -> None:
    if not findings:
        print("findings: none")
        return
    print("findings:")
    for i, f in enumerate(findings, 1):
        print(f"  {i}. [{f['id']}] {f['title']} (score {f['score']:.3f}, risk {f['risk']})")
        print(f"     evidence: {f['evidence']}")
        print(f"     change: {f['suggestion']}")
        print(f"     expected: {f['expected_effect']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from train_doctor.runner import CommandError

    try:
        return _dispatch(args)
    except (CommandError, FileNotFoundError, ValueError) as e:
        print(f"train-doctor: {e}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:

    if args.command == "profile":
        cmd = _strip_dashdash(args.cmd)
        if not cmd:
            print(
                "train-doctor profile: give the training command after --, for example: train-doctor profile -- python train.py",
                file=sys.stderr,
            )
            return 2
        res = api.profile(
            cmd,
            cwd=args.cwd,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            seconds=args.seconds,
            timeout=args.timeout,
            profiler_steps=args.profiler_steps,
            py_spy=args.py_spy,
            seed=args.seed,
            out_dir=args.out,
            echo=args.echo,
            label=args.label,
        )
        if args.json:
            ev = res["evidence"]
            print(
                json.dumps(
                    {
                        "run_dir": res["run_dir"],
                        "error": res.get("error"),
                        "next_step": res["next_step"],
                        "throughput": ev.get("throughput"),
                        "step_time_ms": ev.get("step_time_ms"),
                        "split": ev.get("split"),
                        "findings": res["findings"],
                        "limits": ev.get("limits"),
                    },
                    indent=2,
                    default=str,
                )
            )
        else:
            _print_profile(res)
        return 1 if res.get("error") else 0

    if args.command == "diagnose":
        res = api.diagnose(args.run_dir, args.out)
        if args.json:
            print(json.dumps(res, indent=2, default=str))
        else:
            print(f"run: {res['run_dir']}")
            _print_findings(res["findings"])
            for lim in res["limits"]:
                print(f"limit: {lim}")
        return 0

    if args.command == "compare":
        res = api.compare(
            args.baseline,
            args.candidate,
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            cwd=args.cwd,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            seconds=args.seconds,
            timeout=args.timeout,
            seed=args.seed,
            loss_tol=args.loss_tol,
            min_effect=args.min_effect,
            order_seed=args.order_seed,
            out_dir=args.out,
            echo=args.echo,
            label=args.label,
            progress=None if args.quiet else (lambda done, total, msg: print(msg, file=sys.stderr, flush=True)),
        )
        if args.json:
            print(json.dumps({k: v for k, v in res.items() if k != "machine"}, indent=2, default=str))
        else:
            print(f"decision: {res['decision']}: {res.get('reason')}")
            if res.get("ratio"):
                ra = res["ratio"]
                tp = res["throughput"]
                unit = "samples/s" if res["metric"] == "samples_per_s" else "steps/s"
                print(f"baseline median {tp['baseline']['median']:.2f} {unit}, candidate median {tp['candidate']['median']:.2f} {unit}")
                w = res["wins"]
                print(
                    f"median pair ratio {ra['point']:.3f}x (95% interval {ra['low']:.3f} to {ra['high']:.3f}), "
                    f"candidate faster in {w['candidate_faster']} of {w['pairs']} pairs, verdict: {res['verdict']}"
                )
                st = res.get("stalls") or {}
                if st.get("set_aside"):
                    al = res["ratio_all_pairs"]
                    print(
                        f"stalled pairs set aside: {', '.join(str(i) for i in st['set_aside'])} "
                        f"(with them the interval would be {al['low']:.3f} to {al['high']:.3f})"
                    )
            lc = res.get("loss_check") or {}
            if lc:
                print(
                    f"loss check: {lc.get('status')}"
                    + (f" (mean rel diff {lc['mean_rel_diff'] * 100:.2f}%, tol {lc['tol'] * 100:.0f}%)" if "mean_rel_diff" in lc else "")
                )
            for e in res.get("errors") or []:
                print(f"error: {e}")
            for lim in res.get("limits") or []:
                print(f"limit: {lim}")
            print(f"report: {Path(res['root']) / 'report.md'}")
            if res.get("next_step"):
                print(f"next: {res['next_step']}")
        return 0 if res.get("verdict") != "error" else 1

    if args.command == "report":
        res = api.make_report(args.run_dir, args.out)
        print(Path(res["json" if args.json else "markdown"]).read_text())
        return 0

    if args.command == "setup":
        from train_doctor.setup_agents import run_setup

        return run_setup(apply=args.yes, project=args.project, server_command=args.server_command)

    if args.command == "mcp":
        from train_doctor.mcp_server import serve

        serve()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
