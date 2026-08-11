"""Characterization + unit tests for the `run_one_job` shared helper (Phase 4A).

`tick`'s per-job body (`_process_job`) is the execute → save → deliver → mark
sequence that fires ONE due job. Phase 4A extracts it into a module-level
`run_one_job(job, *, adapters=None, loop=None, verbose=False)` so the external
Chronos provider's `fire_due` can reuse the IDENTICAL body — no duplicated
correctness.

The first test characterizes the sequence as driven through `tick()` (proving
the extraction didn't change `tick`'s behavior); the rest unit-test the
extracted helper directly.
"""
import cron.scheduler as s


def _patch_pipeline(monkeypatch, *, success=True, output="out", final="final response",
                    error=None, silent_marker_in=None, cfg=None):
    """Patch the job pipeline primitives and record the call order."""
    calls = []

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        calls.append(("run_job", job["id"]))
        fr = final if silent_marker_in is None else silent_marker_in
        return (success, output, fr, error)

    def fake_save(jid, out):
        calls.append(("save", jid))
        return f"/tmp/{jid}.txt"

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append(("deliver", job["id"], job.get("deliver")))
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None):
        calls.append(("mark", jid, ok))

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", fake_save)
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    monkeypatch.setattr(s, "_load_cron_delivery_config", lambda: cfg or {})
    return calls


def test_tick_process_job_sequence(monkeypatch):
    """Characterization: a single due job driven through tick() runs the
    sequence run_job → save → deliver → mark, in that order."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "get_due_jobs", lambda: [{"id": "j1", "name": "t"}])
    monkeypatch.setattr(s, "advance_next_runs", lambda ids: 1)

    s.tick(verbose=False, sync=True)

    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j1", True)


def test_run_one_job_success_sequence(monkeypatch):
    """The extracted helper runs the same execute→save→deliver→mark sequence
    for a successful job."""
    calls = _patch_pipeline(monkeypatch)

    ok = s.run_one_job({"id": "j2", "name": "t"})

    assert ok is True
    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j2", True)


def test_run_one_job_failure_uses_configured_matrix_target(monkeypatch):
    """A broken Telegram-origin job is delivered only to the Matrix override."""
    calls = _patch_pipeline(
        monkeypatch,
        success=False,
        final="",
        error="provider timeout",
        cfg={"cron": {"failure_deliver": "matrix:!errors:example.org"}},
    )
    job = {
        "id": "broken",
        "name": "broken report",
        "deliver": "telegram:1234",
        "origin": {"platform": "telegram", "chat_id": "1234"},
    }

    assert s.run_one_job(job) is True
    deliveries = [call for call in calls if call[0] == "deliver"]
    assert deliveries == [("deliver", "broken", "matrix:!errors:example.org")]
    assert job["deliver"] == "telegram:1234"  # stored job view was not mutated


def test_run_one_job_success_keeps_original_target(monkeypatch):
    calls = _patch_pipeline(
        monkeypatch,
        success=True,
        cfg={"cron": {"failure_deliver": "matrix:!errors:example.org"}},
    )
    job = {"id": "healthy", "name": "daily report", "deliver": "telegram:1234"}

    assert s.run_one_job(job) is True
    deliveries = [call for call in calls if call[0] == "deliver"]
    assert deliveries == [("deliver", "healthy", "telegram:1234")]


def test_failure_override_local_suppresses_chat_delivery(monkeypatch):
    calls = _patch_pipeline(
        monkeypatch,
        success=False,
        final="",
        error="script failed",
        cfg={"cron": {"failure_deliver": "local"}},
    )
    job = {"id": "quiet-failure", "name": "watchdog", "deliver": "telegram:1234"}

    assert s.run_one_job(job) is True
    # The normal delivery function still runs so bookkeeping stays unchanged;
    # ``deliver=local`` resolves to zero platform targets.
    assert [call for call in calls if call[0] == "deliver"] == [
        ("deliver", "quiet-failure", "local")
    ]


def test_empty_agent_response_is_routed_as_matrix_failure(monkeypatch):
    calls = _patch_pipeline(
        monkeypatch,
        success=True,
        final="   \n",
        cfg={"cron": {"failure_deliver": "matrix:!errors:example.org"}},
    )
    job = {"id": "empty", "name": "empty response", "deliver": "telegram:1234"}

    assert s.run_one_job(job) is True
    assert [call for call in calls if call[0] == "deliver"] == [
        ("deliver", "empty", "matrix:!errors:example.org")
    ]
    assert calls[-1] == ("mark", "empty", False)


def test_processing_exception_is_routed_as_matrix_failure(monkeypatch):
    calls = _patch_pipeline(
        monkeypatch,
        cfg={"cron": {"failure_deliver": "matrix:!errors:example.org"}},
    )

    def explode(*args, **kwargs):
        raise RuntimeError("executor exploded")

    monkeypatch.setattr(s, "run_job", explode)
    job = {"id": "exception", "name": "exception path", "deliver": "telegram:1234"}

    assert s.run_one_job(job) is False
    assert [call for call in calls if call[0] == "deliver"] == [
        ("deliver", "exception", "matrix:!errors:example.org")
    ]
    assert calls[-1] == ("mark", "exception", False)


def test_unresolvable_failure_override_never_falls_back(monkeypatch):
    calls = _patch_pipeline(
        monkeypatch,
        success=False,
        final="",
        error="broken",
        cfg={"cron": {"failure_deliver": "not-a-platform"}},
    )
    recorded = {}

    def capture_mark(jid, ok, err=None, delivery_error=None):
        recorded.update(ok=ok, delivery_error=delivery_error)

    monkeypatch.setattr(s, "mark_job_run", capture_mark)
    job = {"id": "invalid-route", "name": "invalid route", "deliver": "telegram:1234"}

    assert s.run_one_job(job) is True
    assert [call for call in calls if call[0] == "deliver"] == []
    assert recorded["ok"] is False
    assert "could not be resolved" in recorded["delivery_error"]


def test_run_one_job_installs_secret_scope_under_multiplex(monkeypatch, tmp_path):
    """Regression: under profile isolation (multiplex active), run_one_job must
    execute run_job inside a profile secret scope so credential reads
    (resolve_runtime_provider -> get_secret) don't fail-close with
    UnscopedSecretError, and must tear the scope down afterward.

    Behavior contract: a scope is present during run_job and absent after,
    regardless of the concrete secret values.
    """
    from agent import secret_scope as ss

    # Point cron's home resolution at a profile whose .env carries a secret.
    (tmp_path / ".env").write_text("OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n")
    monkeypatch.setattr(s, "_get_hermes_home", lambda: tmp_path)

    scope_during_run = {}

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        # This is where resolve_runtime_provider() would read a secret. Prove a
        # scope is installed and the profile's secret resolves without raising.
        scope_during_run["scope"] = ss.current_secret_scope()
        scope_during_run["base_url"] = ss.get_secret("OPENROUTER_BASE_URL")
        return (True, "out", "final", None)

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)

    ss.set_multiplex_active(True)
    try:
        ok = s.run_one_job({"id": "j7", "name": "t"})
    finally:
        ss.set_multiplex_active(False)

    assert ok is True
    # Scope was installed during run_job and the profile secret resolved.
    assert scope_during_run["scope"] is not None
    assert scope_during_run["base_url"] == "https://openrouter.ai/api/v1"
    # And it was torn down after run_one_job returned (no leak).
    assert ss.current_secret_scope() is None
