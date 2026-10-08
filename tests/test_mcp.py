import asyncio

from train_doctor.mcp_server import build_server


def test_tools_are_registered():
    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"profile", "diagnose", "compare", "report", "version"} <= names
    prof = next(t for t in tools if t.name == "profile")
    assert "command" in schema(prof)["properties"]
    cmp_ = next(t for t in tools if t.name == "compare")
    assert {"baseline", "candidate", "repeats"} <= set(schema(cmp_)["properties"])


def schema(tool):
    # mcp 1.x calls it inputSchema, 2.x input_schema
    return getattr(tool, "input_schema", None) or tool.inputSchema


def test_profile_tool_runs_a_command(tmp_path, script):
    cmd = script(
        "loop.py",
        """
        import time, train_doctor
        for i in range(500):
            time.sleep(0.005)
            train_doctor.step(samples=4, loss=1.0)
    """,
    )
    server = build_server()
    args = {"command": cmd, "warmup_steps": 1, "steps": 5, "seconds": 0, "profiler_steps": 0, "out_dir": str(tmp_path / "runs")}
    out = asyncio.run(server.call_tool("profile", args))
    text = str(out)
    assert "samples_per_s" in text and "report.md" in text
