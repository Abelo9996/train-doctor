"""setup runs against a temporary HOME with a fake `claude` CLI."""

import json

from train_doctor.setup_agents import run_setup, skill_text


def make_agents(home):
    (home / ".claude").mkdir()
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text('model = "o4"\n')
    (home / ".cursor").mkdir()
    (home / ".cursor" / "mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))


def fake_claude(fake_bin, tmp_path):
    state = tmp_path / "claude_registered"
    log = tmp_path / "claude_calls.log"
    fake_bin(
        "claude",
        f"""
        echo "$@" >> {log}
        if [ "$1 $2" = "mcp get" ]; then
          [ -f {state} ] && exit 0 || exit 1
        fi
        if [ "$1 $2" = "mcp add" ]; then
          touch {state}
          exit 0
        fi
        exit 2
    """,
    )
    return log


def test_dry_run_changes_nothing(_isolated_home, fake_bin, tmp_path, capsys):
    make_agents(_isolated_home)
    log = fake_claude(fake_bin, tmp_path)
    before = (_isolated_home / ".codex" / "config.toml").read_text()
    assert run_setup(apply=False) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out and "claude mcp add --scope user train-doctor -- uvx train-doctor mcp" in out
    assert (_isolated_home / ".codex" / "config.toml").read_text() == before
    assert not (_isolated_home / ".claude" / "skills").exists()
    assert "mcp add" not in log.read_text()


def test_apply_then_idempotent(_isolated_home, fake_bin, tmp_path, capsys):
    make_agents(_isolated_home)
    log = fake_claude(fake_bin, tmp_path)
    project = tmp_path / "proj"
    project.mkdir()
    assert run_setup(apply=True, project=str(project)) == 0
    codex = (_isolated_home / ".codex" / "config.toml").read_text()
    assert 'model = "o4"' in codex and "[mcp_servers.train-doctor]" in codex and 'args = ["train-doctor", "mcp"]' in codex
    cursor = json.loads((_isolated_home / ".cursor" / "mcp.json").read_text())
    assert cursor["mcpServers"]["other"] == {"command": "x"}
    assert cursor["mcpServers"]["train-doctor"] == {"command": "uvx", "args": ["train-doctor", "mcp"]}
    assert json.loads((project / ".mcp.json").read_text())["mcpServers"]["train-doctor"]["command"] == "uvx"
    assert (_isolated_home / ".claude" / "skills" / "train-doctor" / "SKILL.md").read_text() == skill_text()
    assert (_isolated_home / ".codex" / "skills" / "train-doctor" / "SKILL.md").exists()
    assert "mcp add --scope user train-doctor -- uvx train-doctor mcp" in log.read_text()
    backups = list((_isolated_home / ".codex").glob("config.toml.train-doctor-backup-*"))
    assert len(backups) == 1 and backups[0].read_text() == 'model = "o4"\n'
    capsys.readouterr()

    assert run_setup(apply=True, project=str(project)) == 0
    assert "Nothing to change." in capsys.readouterr().out
    assert (_isolated_home / ".codex" / "config.toml").read_text().count("[mcp_servers.train-doctor]") == 1


def test_no_agents_found(_isolated_home, monkeypatch, capsys):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run_setup(apply=True) == 0
    assert "No Claude Code, Codex or Cursor" in capsys.readouterr().out


def test_skill_has_frontmatter():
    text = skill_text()
    assert text.startswith("---\nname: train-doctor\ndescription: ")
