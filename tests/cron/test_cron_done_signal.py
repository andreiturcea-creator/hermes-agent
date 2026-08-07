"""Tests for the recurring-cron [DONE] completion signal (cron/scheduler.py).

RC2 of the 2026-06-26 cron work: a recurring monitor cron that emits the
``[DONE]`` marker in its output is delivered one final time and then STOPPED,
instead of running forever. Extends the existing ``[SILENT]`` sentinel pattern.
Mirrors the monkeypatch pipeline used in test_run_one_job.py.
"""

import cron.scheduler as s


def _patch(monkeypatch, *, success=True, final="final response"):
    """Patch run_one_job's pipeline primitives; record deliver content + pauses."""
    calls = []
    delivered = []
    paused = []

    monkeypatch.setattr(
        s,
        "run_job",
        lambda job, **_kwargs: (success, "out", final, None),
    )
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append("deliver")
        delivered.append(content)
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None):
        calls.append("mark")

    def fake_pause(jid, reason=None):
        calls.append("pause")
        paused.append((jid, reason))
        return {"id": jid, "enabled": False}

    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    monkeypatch.setattr(s, "pause_job", fake_pause)
    return calls, delivered, paused


def _recurring(**extra):
    return {"id": "j1", "name": "monitor", "schedule": {"kind": "interval", "minutes": 10}, **extra}


def _oneshot(**extra):
    return {"id": "j1", "name": "once", "schedule": {"kind": "once", "run_at": "x"}, **extra}


# ---------------------------------------------------------------------------
# Termination on [DONE]
# ---------------------------------------------------------------------------

def test_recurring_done_pauses_job(monkeypatch):
    calls, delivered, paused = _patch(monkeypatch, final="All finished, the parts arrived. [DONE]")
    s.run_one_job(_recurring())
    assert paused and paused[0][0] == "j1"
    assert "[DONE]" in (paused[0][1] or "")  # reason references the signal


def test_done_marker_stripped_from_delivery(monkeypatch):
    calls, delivered, paused = _patch(monkeypatch, final="All finished, the parts arrived. [DONE]")
    s.run_one_job(_recurring())
    assert "deliver" in calls
    assert delivered[0].upper().find("[DONE]") == -1  # marker gone
    assert "All finished" in delivered[0]             # real content kept


def test_recurring_without_done_not_paused(monkeypatch):
    calls, delivered, paused = _patch(monkeypatch, final="Still waiting, no change yet.")
    s.run_one_job(_recurring())
    assert paused == []
    assert "deliver" in calls


def test_oneshot_with_done_not_paused(monkeypatch):
    # One-shots already self-terminate; [DONE] must not trigger an extra pause.
    calls, delivered, paused = _patch(monkeypatch, final="did the thing [DONE]")
    s.run_one_job(_oneshot())
    assert paused == []


def test_done_only_response_terminates_without_delivery(monkeypatch):
    # Bare "[DONE]" => nothing to deliver after stripping, but the job still stops.
    calls, delivered, paused = _patch(monkeypatch, final="[DONE]")
    s.run_one_job(_recurring())
    assert "deliver" not in calls
    assert paused and paused[0][0] == "j1"


def test_done_marker_case_insensitive(monkeypatch):
    calls, delivered, paused = _patch(monkeypatch, final="all set [done]")
    s.run_one_job(_recurring())
    assert paused and paused[0][0] == "j1"


def test_done_mentioned_midreport_does_not_terminate(monkeypatch):
    # A monitor whose report merely MENTIONS "[DONE]" (not as a trailing marker)
    # must NOT stop itself, and the text must be delivered intact.
    calls, delivered, paused = _patch(
        monkeypatch, final="Status:\n- Item A: [DONE]\n- Item B: still pending")
    s.run_one_job(_recurring())
    assert paused == []
    assert "deliver" in calls
    assert "[DONE]" in delivered[0]  # not stripped — it's legitimate content


def test_trailing_done_on_own_line_terminates(monkeypatch):
    calls, delivered, paused = _patch(
        monkeypatch, final="All items complete.\n\n[DONE]\n")
    s.run_one_job(_recurring())
    assert paused and paused[0][0] == "j1"
    assert "[DONE]" not in delivered[0]
    assert "All items complete." in delivered[0]


def test_trailing_done_with_punctuation_terminates(monkeypatch):
    # Natural LLM phrasing 'All resolved. [DONE].' / '...[DONE]!' must still stop.
    for final in ("All resolved. [DONE].", "Everything handled [DONE]!", "fixed. [DONE] ."):
        calls, delivered, paused = _patch(monkeypatch, final=final)
        s.run_one_job(_recurring())
        assert paused and paused[0][0] == "j1", f"failed to terminate on {final!r}"


def test_silent_unchanged_no_pause(monkeypatch):
    # Regression: [SILENT] suppresses delivery and must NOT stop the job.
    calls, delivered, paused = _patch(monkeypatch, final="[SILENT]")
    s.run_one_job(_recurring())
    assert "deliver" not in calls
    assert paused == []


def test_failed_run_not_terminated(monkeypatch):
    # A failed run that happens to contain the marker text must not be treated
    # as completion (done requires success).
    calls, delivered, paused = _patch(monkeypatch, success=False, final="error [DONE]")
    s.run_one_job(_recurring())
    assert paused == []


# ---------------------------------------------------------------------------
# Prompt guidance
# ---------------------------------------------------------------------------

def test_hint_includes_done_for_recurring():
    hint = s._cron_execution_hint(_recurring())
    assert "[DONE]" in hint
    assert "[SILENT]" in hint


def test_hint_omits_done_for_oneshot():
    hint = s._cron_execution_hint(_oneshot())
    assert "[DONE]" not in hint
    assert "[SILENT]" in hint  # one-shots still get the base guidance


def test_hint_base_text_preserved():
    # The base cron guidance text must be byte-identical to the original so
    # existing behaviour/tests that depend on it are unaffected.
    hint = s._cron_execution_hint(_oneshot())
    assert hint.startswith("[IMPORTANT: You are running as a scheduled cron job. ")
    assert hint.endswith("say [SILENT] and nothing more.]\n\n")
