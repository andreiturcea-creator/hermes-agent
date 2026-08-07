"""Behavior tests for the terminal DOCX fail-closed guard."""

import json
from pathlib import Path

import pytest

import tools.docx_safety_guard as guard
import tools.terminal_tool as terminal_tool


def _config(cwd):
    return {
        "env_type": "local",
        "cwd": str(cwd),
        "timeout": 30,
        "lifetime_seconds": 3600,
    }


@pytest.fixture
def validator(tmp_path, monkeypatch):
    module = tmp_path / "validator.py"
    module.write_text(
        "from types import SimpleNamespace\n"
        "def inspect_docx(path):\n"
        "    data = open(path, 'rb').read(4)\n"
        "    ok = data == b'PK\\x03\\x04'\n"
        "    return SimpleNamespace(valid=ok, errors=() if ok else ('invalid package',))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_DOCX_SAFETY_MODULE", str(module))
    monkeypatch.setattr(guard, "_VALIDATOR", None)
    return module


def test_candidate_paths_resolve_quoted_spaces_and_reject_globs(tmp_path):
    expected = tmp_path / "folder with spaces" / "output.docx"
    assert guard.candidate_paths(
        "python writer.py 'folder with spaces/output.docx'", str(tmp_path)
    ) == [expected]
    assert guard.candidate_paths("ls *.docx", str(tmp_path)) == []


def test_invalid_overwrite_is_quarantined_and_original_restored(tmp_path, validator):
    target = tmp_path / "protected.docx"
    target.write_bytes(b"PK\x03\x04original")
    session = guard.prepare_terminal_guard(f"writer '{target}'", str(tmp_path))
    target.write_bytes(b"plain text")

    report = guard.finalize_terminal_guard(session)

    assert report.ok is False
    assert target.read_bytes() == b"PK\x03\x04original"
    assert report.restored == (str(target),)
    assert len(report.quarantined) == 1
    assert Path(report.quarantined[0]).read_bytes() == b"plain text"


def test_invalid_new_output_is_quarantined_and_removed(tmp_path, validator):
    target = tmp_path / "new.docx"
    session = guard.prepare_terminal_guard(f"writer '{target}'", str(tmp_path))
    target.write_bytes(b"not a package")

    report = guard.finalize_terminal_guard(session)

    assert report.ok is False
    assert not target.exists()
    assert len(report.quarantined) == 1


def _run_terminal(monkeypatch, tmp_path, command, execute, *, background=False):
    class FakeEnv:
        env = {}
        cwd = str(tmp_path)

        def execute(self, raw_command, **kwargs):
            return execute(raw_command, **kwargs)

    task_id = "docx-safety-test"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _config(tmp_path))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    return json.loads(
        terminal_tool.terminal_tool(
            command=command,
            task_id=task_id,
            background=background,
        )
    )


def test_terminal_foreground_invalid_docx_fails_closed(
    tmp_path, monkeypatch, validator
):
    target = tmp_path / "terminal.docx"

    def execute(_command, **_kwargs):
        target.write_bytes(b"plain text")
        return {"output": "writer said success", "returncode": 0}

    result = _run_terminal(
        monkeypatch,
        tmp_path,
        f"writer '{target}'",
        execute,
    )

    assert result["exit_code"] == 65
    assert "DOCX SAFETY BLOCK" in result["output"]
    assert not target.exists()


def test_terminal_background_docx_is_blocked_before_spawn(
    tmp_path, monkeypatch, validator
):
    called = False

    def execute(_command, **_kwargs):
        nonlocal called
        called = True
        return {"output": "", "returncode": 0}

    result = _run_terminal(
        monkeypatch,
        tmp_path,
        f"writer '{tmp_path / 'background.docx'}'",
        execute,
        background=True,
    )

    assert result["exit_code"] == 65
    assert result["status"] == "blocked"
    assert called is False
