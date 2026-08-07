"""Tests for the per-turn file-mutation verifier footer.

Covers the three moving pieces:

1. ``_extract_file_mutation_targets`` — pulls file paths from write_file /
   patch (replace + V4A) tool-call argument dicts.
2. ``AIAgent._record_file_mutation_result`` — builds the per-turn state
   dict, removing entries when a later success supersedes an earlier
   failure for the same path.
3. ``AIAgent._format_file_mutation_failure_footer`` — renders the dict
   as a user-visible advisory.

Regression target: the "Ben Eng llm-wiki" session where grok-4.1-fast
batched parallel patches, half failed, and the model summarised the
turn claiming every file was edited.  This verifier makes over-claiming
structurally impossible past the model: the user always sees the real
list of files that did NOT change.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from run_agent import (
    AIAgent,
    _FILE_MUTATING_TOOLS,
    _extract_error_preview,
    _extract_file_mutation_targets,
    _extract_landed_file_mutation_paths,
    _extract_tool_error_code,
)


# ---------------------------------------------------------------------------
# _extract_file_mutation_targets
# ---------------------------------------------------------------------------


class TestExtractFileMutationTargets:
    def test_non_mutating_tool_returns_empty(self):
        assert _extract_file_mutation_targets("read_file", {"path": "/x"}) == []
        assert _extract_file_mutation_targets("terminal", {"command": "ls"}) == []



    def test_patch_replace_mode_returns_path(self):
        args = {"mode": "replace", "path": "/tmp/a.md", "old_string": "x", "new_string": "y"}
        assert _extract_file_mutation_targets("patch", args) == ["/tmp/a.md"]



    def test_patch_v4a_multi_file(self):
        body = (
            "*** Begin Patch\n"
            "*** Update File: /tmp/a.md\n"
            "@@ @@\n-a\n+b\n"
            "*** Add File: /tmp/new.md\n"
            "+fresh\n"
            "*** Delete File: /tmp/old.md\n"
            "*** End Patch\n"
        )
        args = {"mode": "patch", "patch": body}
        paths = _extract_file_mutation_targets("patch", args)
        assert paths == ["/tmp/a.md", "/tmp/new.md", "/tmp/old.md"]


    def test_patch_v4a_accepts_no_space_after_asterisks(self):
        """Match patch_parser / file_tools: ``***Update File:`` (no space)."""
        body = "***Update File: nospace.py\n"
        assert _extract_file_mutation_targets(
            "patch", {"mode": "patch", "patch": body}
        ) == ["nospace.py"]


# ---------------------------------------------------------------------------
# _extract_error_preview
# ---------------------------------------------------------------------------


class TestExtractErrorPreview:
    def test_json_error_field_preferred(self):
        raw = json.dumps({"success": False, "error": "Could not find old_string in /tmp/x"})
        assert _extract_error_preview(raw) == "Could not find old_string in /tmp/x"

    def test_plain_string_falls_through(self):
        assert _extract_error_preview("Error executing tool: boom") == "Error executing tool: boom"

    def test_long_preview_truncated(self):
        long = "x" * 500
        out = _extract_error_preview(long, max_len=50)
        assert len(out) <= 50
        assert out.endswith("…")



# ---------------------------------------------------------------------------
# _record_file_mutation_result — state transitions
# ---------------------------------------------------------------------------


def _bare_agent() -> AIAgent:
    """Skip __init__ and only attach the per-turn state dict.

    AIAgent.__init__ takes ~60 parameters and touches network, auth, and
    the filesystem.  For these tests we only need the two methods —
    ``_record_file_mutation_result`` and ``_format_file_mutation_failure_footer``.
    Using ``object.__new__`` mirrors the gateway-test pattern documented in
    the agent pitfalls list.
    """
    agent = object.__new__(AIAgent)
    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()
    return agent


class TestRecordFileMutationResult:
    def test_non_mutating_tool_ignored(self):
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "read_file", {"path": "/tmp/x"}, "{}", is_error=True,
        )
        assert agent._turn_failed_file_mutations == {}

    def test_failure_recorded(self):
        agent = _bare_agent()
        result = json.dumps({"success": False, "error": "Could not find old_string"})
        agent._record_file_mutation_result(
            "patch", {"mode": "replace", "path": "/tmp/a.md", "old_string": "x", "new_string": "y"},
            result, is_error=True,
        )
        state = agent._turn_failed_file_mutations
        assert "/tmp/a.md" in state
        assert state["/tmp/a.md"]["tool"] == "patch"
        assert "Could not find old_string" in state["/tmp/a.md"]["error_preview"]

    def test_success_removes_prior_failure(self):
        agent = _bare_agent()
        # First attempt fails
        agent._record_file_mutation_result(
            "patch", {"mode": "replace", "path": "/tmp/a.md", "old_string": "x", "new_string": "y"},
            json.dumps({"error": "not found"}), is_error=True,
        )
        assert "/tmp/a.md" in agent._turn_failed_file_mutations
        # Second attempt with corrected old_string succeeds
        agent._record_file_mutation_result(
            "patch", {"mode": "replace", "path": "/tmp/a.md", "old_string": "real", "new_string": "fixed"},
            json.dumps({"success": True, "diff": "..."}), is_error=False,
        )
        assert agent._turn_failed_file_mutations == {}
        assert agent._turn_file_mutation_paths == {"/tmp/a.md"}


    def test_landed_paths_prefer_resolved_tool_result(self):
        paths = _extract_landed_file_mutation_paths(
            "patch",
            {"mode": "replace", "path": "src/app.py"},
            json.dumps({
                "success": True,
                "files_modified": ["/tmp/project/src/app.py"],
            }),
        )

        assert paths == ["/tmp/project/src/app.py"]

    def test_write_file_with_lint_error_counts_as_landed(self):
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "write_file",
            {"path": "/tmp/a.py", "content": "bad"},
            json.dumps({"error": "write failed"}),
            is_error=True,
        )
        assert "/tmp/a.py" in agent._turn_failed_file_mutations

        result = json.dumps({
            "bytes_written": 24,
            "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
        })

        agent._record_file_mutation_result(
            "write_file",
            {"path": "/tmp/a.py", "content": "def nope(:\n"},
            result,
            is_error=True,
        )

        assert agent._turn_failed_file_mutations == {}

    def test_patch_with_lsp_diagnostics_counts_as_landed(self):
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": "/tmp/a.py", "old_string": "x", "new_string": "y"},
            json.dumps({"error": "Could not find old_string"}),
            is_error=True,
        )
        assert "/tmp/a.py" in agent._turn_failed_file_mutations

        result = json.dumps({
            "success": True,
            "diff": "--- a/tmp.py\n+++ b/tmp.py\n",
            "files_modified": ["/tmp/a.py"],
            "lsp_diagnostics": "<diagnostics>ERROR [1:1] type mismatch</diagnostics>",
        })

        agent._record_file_mutation_result(
            "patch",
            {"mode": "replace", "path": "/tmp/a.py", "old_string": "x", "new_string": "y"},
            result,
            is_error=True,
        )

        assert agent._turn_failed_file_mutations == {}

    def test_repeated_failure_keeps_first_error(self):
        agent = _bare_agent()
        agent._record_file_mutation_result(
            "patch", {"mode": "replace", "path": "/tmp/a.md", "old_string": "v1", "new_string": "y"},
            json.dumps({"error": "first error"}), is_error=True,
        )
        agent._record_file_mutation_result(
            "patch", {"mode": "replace", "path": "/tmp/a.md", "old_string": "v2", "new_string": "y"},
            json.dumps({"error": "second error"}), is_error=True,
        )
        # Keep the original error — swapping to the latest would obscure
        # the initial root cause.
        assert "first error" in agent._turn_failed_file_mutations["/tmp/a.md"]["error_preview"]





# ---------------------------------------------------------------------------
# _format_file_mutation_failure_footer
# ---------------------------------------------------------------------------


class TestFormatFooter:
    def test_empty_returns_empty_string(self):
        assert AIAgent._format_file_mutation_failure_footer({}) == ""

    def test_single_failure(self):
        out = AIAgent._format_file_mutation_failure_footer(
            {"/tmp/a.md": {"tool": "patch", "error_preview": "Could not find old_string"}},
        )
        assert "1 file(s) were NOT modified" in out
        assert "/tmp/a.md" in out
        assert "Could not find old_string" in out
        assert "git status" in out  # user-actionable hint

    def test_truncation_at_10_entries(self):
        failed = {
            f"/tmp/f{i}.md": {"tool": "patch", "error_preview": "err"}
            for i in range(15)
        }
        out = AIAgent._format_file_mutation_failure_footer(failed)
        assert "15 file(s) were NOT modified" in out
        assert "… and 5 more" in out
        # Ten file bullets + header + "and X more" line
        lines = out.split("\n")
        bullet_lines = [ln for ln in lines if ln.lstrip().startswith("•")]
        assert len(bullet_lines) == 11  # 10 shown + 1 summary


    def test_footer_path_not_extracted_by_gateway(self):
        """End-to-end: the gateway's extract_local_files must NOT pull a
        config.yaml path out of the rendered footer (#35584)."""
        import os
        import tempfile
        from gateway.platforms.base import BasePlatformAdapter

        tmp = tempfile.mkdtemp(prefix="hermes_footer_")
        try:
            cfg = os.path.join(tmp, "config.yaml")
            with open(cfg, "w") as fh:
                fh.write("openrouter_api_key: sk-LEAK\n")
            footer = AIAgent._format_file_mutation_failure_footer(
                {cfg: {
                    "tool": "patch",
                    "error_preview": (
                        f"Write denied: '{cfg}' is a protected "
                        "system/credential file."
                    ),
                }},
            )
            response = "I updated your config.\n\n" + footer
            paths, _ = BasePlatformAdapter.extract_local_files(response)
            assert paths == [], f"footer leaked deliverable path(s): {paths}"
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# _file_mutation_verifier_enabled — env + config precedence
# ---------------------------------------------------------------------------


class TestVerifierEnabled:
    def test_default_is_enabled(self, monkeypatch):
        monkeypatch.delenv("HERMES_FILE_MUTATION_VERIFIER", raising=False)
        agent = _bare_agent()
        # With no env and no config present, safe default is True.
        # load_config may surface a user config.yaml in some envs — stub it.
        import hermes_cli.config as _cfg_mod
        monkeypatch.setattr(_cfg_mod, "load_config", lambda: {})
        assert agent._file_mutation_verifier_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
    def test_env_disables(self, monkeypatch, value):
        monkeypatch.setenv("HERMES_FILE_MUTATION_VERIFIER", value)
        agent = _bare_agent()
        assert agent._file_mutation_verifier_enabled() is False




# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_file_mutating_tools_set_shape():
    """write_file + patch are the only tools the verifier tracks.

    Guard rail: if someone adds a third file-mutating tool (e.g. a new
    ``append_file``), they should also audit whether the verifier should
    track it.  This test fails loudly on unilateral additions.
    """
    assert _FILE_MUTATING_TOOLS == frozenset({"write_file", "patch"})


# ---------------------------------------------------------------------------
# Temp-verifier helper classification
# ---------------------------------------------------------------------------


def _denied(error_code: str | None = "sensitive_path_denied") -> str:
    """A tool_error-shaped denial result, optionally carrying error_code."""
    body: dict = {"error": "Refusing to write to sensitive system path: x"}
    if error_code is not None:
        body["error_code"] = error_code
    return json.dumps(body)


def _temp_helper_path(name: str) -> str:
    """A path under the real OS tempdir with the given basename."""
    return str(Path(tempfile.gettempdir()) / name)


class TestTempHelperClassifierFooter:
    def test_real_target_failure_still_warns(self):
        footer = AIAgent._format_file_mutation_failure_footer(
            {"/home/u/project/src/app.py": {
                "tool": "patch", "error_preview": "Could not find old_string",
            }},
            changed_paths=["/home/u/project/src/app.py"],
            session_id="s1",
        )
        assert "NOT modified" in footer
        assert "/home/u/project/src/app.py" in footer

    def test_temp_helper_sensitive_denial_drops_not_modified(self):
        helper = _temp_helper_path("hermes-verify-demo.py")
        footer = AIAgent._format_file_mutation_failure_footer(
            {helper: {
                "tool": "write_file",
                "error_preview": "sensitive path",
                "error_code": "sensitive_path_denied",
            }},
            changed_paths=["/home/u/project/src/app.py"],
            session_id="s1",
        )
        assert "NOT modified" not in footer
        assert "temporary verification helper" in footer

    def test_helper_only_with_fresh_pass_suppresses_footer(self, monkeypatch):
        helper = _temp_helper_path("hermes-verify-demo.py")
        # Seed the freshness read via the REAL dict key "status" (NOT "state").
        import agent.verification_evidence as ve
        monkeypatch.setattr(
            ve, "verification_status",
            lambda *, session_id, cwd: {"status": "passed", "evidence": {}},
        )
        footer = AIAgent._format_file_mutation_failure_footer(
            {helper: {
                "tool": "write_file",
                "error_preview": "sensitive path",
                "error_code": "sensitive_path_denied",
            }},
            changed_paths=["/home/u/project/src/app.py"],
            session_id="s1",
        )
        assert footer == ""

    def test_helper_only_with_fresh_pass_suppresses_footer_integration(
        self, tmp_path, monkeypatch
    ):
        """End-to-end through the REAL verification_status / DB writer, so a
        rename of the "status" key would break this (a monkeypatch cannot)."""
        from agent.verification_evidence import record_terminal_result

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        # A real node project root with a canonical test command.
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "vitest"}}), encoding="utf-8"
        )
        (tmp_path / "pnpm-lock.yaml").write_text("", encoding="utf-8")
        # Record a real PASSING canonical command -> verification_status passed.
        record_terminal_result(
            command="pnpm test",
            cwd=tmp_path,
            session_id="s1",
            exit_code=0,
            output="green",
        )
        helper = _temp_helper_path("hermes-verify-demo.py")
        footer = AIAgent._format_file_mutation_failure_footer(
            {helper: {
                "tool": "write_file",
                "error_preview": "sensitive path",
                "error_code": "sensitive_path_denied",
            }},
            changed_paths=[str(tmp_path / "src" / "app.ts")],
            session_id="s1",
        )
        assert footer == ""

    def test_helper_only_without_verification_emits_info_note(self):
        helper = _temp_helper_path("hermes-verify-demo.py")
        footer = AIAgent._format_file_mutation_failure_footer(
            {helper: {
                "tool": "write_file",
                "error_preview": "sensitive path",
                "error_code": "sensitive_path_denied",
            }},
            changed_paths=["/home/u/project/src/app.py"],
            session_id="s1",
        )
        assert "NOT modified" not in footer
        assert "temporary verification helper" in footer

    def test_mixed_helper_and_real_warns_only_real(self):
        helper = _temp_helper_path("hermes-verify-demo.py")
        real = "/home/u/project/src/app.py"
        footer = AIAgent._format_file_mutation_failure_footer(
            {
                helper: {
                    "tool": "write_file",
                    "error_preview": "sensitive path",
                    "error_code": "sensitive_path_denied",
                },
                real: {"tool": "patch", "error_preview": "not found"},
            },
            changed_paths=[real],
            session_id="s1",
        )
        assert "NOT modified" in footer
        assert "1 file(s) were NOT modified" in footer
        assert real in footer
        assert helper not in footer

    def test_helper_prefix_in_project_root_still_warns(self, tmp_path):
        # A hermes-verify- file living UNDER an edited project root is a target,
        # not a disposable helper (root exclusion anchored on the file's parent).
        # changed_paths holds FILE paths, so the exclusion anchors on the PARENT
        # dir of an edited file: place both the edited file and the helper in the
        # same directory so the helper resolves under that anchor.
        edited = tmp_path / "app.py"
        helper_in_root = tmp_path / "hermes-verify-thing.py"
        footer = AIAgent._format_file_mutation_failure_footer(
            {str(helper_in_root): {
                "tool": "write_file",
                "error_preview": "sensitive path",
                "error_code": "sensitive_path_denied",
            }},
            changed_paths=[str(edited)],
            session_id="s1",
        )
        assert "NOT modified" in footer
        assert str(helper_in_root) in footer

    def test_tempdir_but_basename_verify_py_still_warns(self):
        # Under tempdir but basename is 'verify.py' (no hermes-verify- prefix).
        path = _temp_helper_path("verify.py")
        footer = AIAgent._format_file_mutation_failure_footer(
            {path: {
                "tool": "write_file",
                "error_preview": "sensitive path",
                "error_code": "sensitive_path_denied",
            }},
            changed_paths=["/home/u/project/src/app.py"],
            session_id="s1",
        )
        assert "NOT modified" in footer

    def test_temp_prefix_but_non_safety_error_still_warns(self):
        # hermes-verify- under tempdir but NO sensitive_path_denied code.
        helper = _temp_helper_path("hermes-verify-demo.py")
        footer = AIAgent._format_file_mutation_failure_footer(
            {helper: {"tool": "write_file", "error_preview": "disk full"}},
            changed_paths=["/home/u/project/src/app.py"],
            session_id="s1",
        )
        assert "NOT modified" in footer

    def test_classifier_exception_falls_back_to_warning(self, monkeypatch):
        """Fail-open fence: if classification raises, the footer degrades to
        warn-all rather than dropping a real 'NOT modified'."""
        def _boom(path, info, roots):
            raise RuntimeError("classifier blew up")

        monkeypatch.setattr(
            AIAgent, "_is_temp_verification_helper_failure",
            staticmethod(_boom),
        )
        helper = _temp_helper_path("hermes-verify-demo.py")
        real = "/home/u/project/src/app.py"
        footer = AIAgent._format_file_mutation_failure_footer(
            {
                real: {"tool": "patch", "error_preview": "not found"},
                helper: {
                    "tool": "write_file",
                    "error_preview": "sensitive path",
                    "error_code": "sensitive_path_denied",
                },
            },
            changed_paths=[real],
            session_id="s1",
        )
        assert footer != ""
        assert "NOT modified" in footer
        assert real in footer

    def test_backtick_neutralization_preserved(self):
        real = "/home/u/project/src/app.py"
        footer = AIAgent._format_file_mutation_failure_footer(
            {real: {"tool": "patch", "error_preview": "not found"}},
            changed_paths=[real],
            session_id="s1",
        )
        assert f"`{real}`" in footer

    def test_legacy_positional_call_preserves_behavior(self):
        # No changed_paths/session_id -> a real failure with no sensitive code
        # still warns.
        footer = AIAgent._format_file_mutation_failure_footer(
            {"/home/u/project/src/app.py": {
                "tool": "patch", "error_preview": "not found"}},
        )
        assert "1 file(s) were NOT modified" in footer
        assert "/home/u/project/src/app.py" in footer

    def test_config_refusal_not_classified_as_helper(self):
        # A config-refusal denial has error_code None even at a hermes-verify-
        # named path -> treated as a real target and warned.
        helper_named = _temp_helper_path("hermes-verify-cfg.py")
        footer = AIAgent._format_file_mutation_failure_footer(
            {helper_named: {
                "tool": "write_file",
                "error_preview": "Refusing to write to Hermes config file",
                "error_code": None,
            }},
            changed_paths=["/home/u/project/src/app.py"],
            session_id="s1",
        )
        assert "NOT modified" in footer
        assert helper_named in footer

    def test_helper_ad_hoc_prefix_sensitive_denial_drops_not_modified(self):
        """Second-prefix case: hermes-ad-hoc- is ALSO a disposable helper
        prefix the runtime emits. The classifier matches both via the imported
        tuple constant."""
        helper = _temp_helper_path("hermes-ad-hoc-demo.py")
        footer = AIAgent._format_file_mutation_failure_footer(
            {helper: {
                "tool": "write_file",
                "error_preview": "sensitive path",
                "error_code": "sensitive_path_denied",
            }},
            changed_paths=["/home/u/project/src/app.py"],
            session_id="s1",
        )
        assert "NOT modified" not in footer
        assert "temporary verification helper" in footer

    def test_helper_classifier_uses_authoritative_prefix_constant(self, monkeypatch):
        """Drift guard: the classifier matches against
        verification_evidence._AD_HOC_SCRIPT_NAME_PREFIXES (imported, not
        re-hardcoded). Extend the constant with a sentinel prefix and confirm a
        file with that prefix is then bucketed as a helper."""
        import agent.verification_evidence as ve

        sentinel = "hermes-zzsentinel-"
        monkeypatch.setattr(
            ve, "_AD_HOC_SCRIPT_NAME_PREFIXES",
            tuple(ve._AD_HOC_SCRIPT_NAME_PREFIXES) + (sentinel,),
        )
        helper = _temp_helper_path(sentinel + "demo.py")
        footer = AIAgent._format_file_mutation_failure_footer(
            {helper: {
                "tool": "write_file",
                "error_preview": "sensitive path",
                "error_code": "sensitive_path_denied",
            }},
            changed_paths=["/home/u/project/src/app.py"],
            session_id="s1",
        )
        # Imported-constant => sentinel now classifies as a helper (info note).
        assert "NOT modified" not in footer
        assert "temporary verification helper" in footer


# ---------------------------------------------------------------------------
# Record layer captures error_code
# ---------------------------------------------------------------------------


class TestRecordCapturesErrorCode:
    def test_record_failure_captures_error_code(self):
        agent = _bare_agent()
        helper = _temp_helper_path("hermes-verify-demo.py")
        agent._record_file_mutation_result(
            "write_file",
            {"path": helper, "content": "x"},
            _denied("sensitive_path_denied"),
            is_error=True,
        )
        assert helper in agent._turn_failed_file_mutations
        assert (
            agent._turn_failed_file_mutations[helper]["error_code"]
            == "sensitive_path_denied"
        )


# ---------------------------------------------------------------------------
# _extract_tool_error_code (suffix-tolerant)
# ---------------------------------------------------------------------------


class TestExtractToolErrorCode:
    def test_extract_error_code_tolerates_guardrail_suffix(self):
        """The decisive test: tool_executor appends a '[Tool loop warning: ...]'
        suffix on the 2nd failure. A plain json.loads of the whole string raises;
        _extract_tool_error_code slices to the last '}' and survives."""
        raw = (
            '{"error":"denied","error_code":"sensitive_path_denied"}'
            "\n\n[Tool loop warning: exact_failure; count=2; retried blocked write]"
        )
        # A naive parse of the WHOLE string fails.
        with pytest.raises(Exception):
            json.loads(raw)
        assert _extract_tool_error_code(raw) == "sensitive_path_denied"

    def test_extract_error_code_overslice_never_fabricates(self):
        """Suppression-direction pin: a result with NO error_code but a stray '}'
        in a trailing suffix must return None, NEVER a fabricated code."""
        raw = '{"ok":false}\n\n[note: see func() }]'
        out = _extract_tool_error_code(raw)
        assert out is None
        assert out != "sensitive_path_denied"
