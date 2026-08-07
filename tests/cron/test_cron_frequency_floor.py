"""Tests for the recurring-cadence frequency floor in cron/jobs.py.

Root-cause regression suite for the 2026-06-26 voice-line bug where a request
meant as a one-shot reminder ("fire once in 1 minute") was created as an
every-minute *forever* cron. The floor rejects pathologically frequent
*recurring* schedules at create time, while leaving one-shot schedules and
all legitimate cadences untouched.
"""

import pytest

import cron.jobs as jobs_mod
from cron.jobs import (
    create_job,
    update_job,
    pause_job,
    resume_job,
    get_job,
    parse_schedule,
    _effective_cadence_seconds,
    MIN_RECURRING_CADENCE_SECONDS,
)


@pytest.fixture(autouse=True)
def _isolate_cron_storage(tmp_path, monkeypatch):
    """Keep these create/update regression tests off the real cron registry.

    cron.jobs resolves JOBS_FILE at module import time. This test module imports
    create_job before the global HERMES_HOME fixture has a chance to run, so
    without patching the module constants here the regression cases append test
    jobs to ~/.hermes/cron/jobs.json when the file is run directly or by an
    agent.  Point storage at a per-test temp cron dir instead.
    """
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs_mod, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", cron_dir / "output")


# ---------------------------------------------------------------------------
# The floor REJECTS pathological recurring schedules
# ---------------------------------------------------------------------------

class TestFloorRejectsEveryMinute:
    def test_every_minute_interval_rejected(self):
        with pytest.raises(ValueError) as exc:
            create_job(prompt="spam", schedule="every 1m", name="bad-interval")
        assert "one-shot" in str(exc.value).lower() or "minimum" in str(exc.value).lower()

    def test_every_minute_cron_rejected(self):
        pytest.importorskip("croniter")
        with pytest.raises(ValueError):
            create_job(prompt="spam", schedule="* * * * *", name="bad-cron")

    def test_step_one_minute_cron_rejected(self):
        pytest.importorskip("croniter")
        with pytest.raises(ValueError):
            create_job(prompt="spam", schedule="*/1 * * * *", name="bad-step")

    def test_every_second_sixfield_not_created(self):
        # 6-field (seconds) every-second cron must never become a live forever job,
        # whether it's parse_schedule or the floor that rejects it.
        with pytest.raises(ValueError):
            create_job(prompt="spam", schedule="* * * * * *", name="bad-seconds")

    def test_rejection_message_points_to_one_shot(self):
        with pytest.raises(ValueError) as exc:
            create_job(prompt="x", schedule="every 1m", name="msg")
        msg = str(exc.value)
        # The error must teach the correct one-shot alternative.
        assert "allow_high_frequency" in msg
        assert "once" in msg.lower()


# ---------------------------------------------------------------------------
# One-shot schedules are NEVER affected (the actual fix for the reported bug)
# ---------------------------------------------------------------------------

class TestOneShotUnaffected:
    @pytest.mark.parametrize("sched", ["1m", "5m", "30m"])
    def test_bare_duration_is_one_shot_fire_once(self, sched):
        job = create_job(prompt="remind me", schedule=sched, name=f"oneshot-{sched}")
        assert job["schedule"]["kind"] == "once"
        assert job["repeat"]["times"] == 1  # fires exactly once

    def test_iso_timestamp_is_one_shot(self):
        job = create_job(prompt="remind me", schedule="2026-12-31T09:00:00", name="iso")
        assert job["schedule"]["kind"] == "once"
        assert job["repeat"]["times"] == 1

    def test_one_minute_oneshot_allowed_even_though_a_minute_away(self):
        # "in one minute, once" is the exact intent that the voice bug mangled.
        # Expressed correctly (bare '1m') it MUST succeed as a one-shot.
        job = create_job(prompt="Test reminder", schedule="1m", name="voice-correct")
        assert job["schedule"]["kind"] == "once"
        assert job["repeat"]["times"] == 1


# ---------------------------------------------------------------------------
# Legitimate recurring cadences (>= floor) still work — no regression
# ---------------------------------------------------------------------------

class TestLegitRecurringStillWorks:
    @pytest.mark.parametrize("sched", ["every 2m", "every 3m", "every 5m", "every 20m", "every 1h"])
    def test_interval_at_or_above_floor_allowed(self, sched):
        job = create_job(prompt="monitor", schedule=sched, name=f"ok-{sched}")
        assert job["schedule"]["kind"] == "interval"

    @pytest.mark.parametrize("sched", ["0 9 * * *", "*/15 * * * *", "*/2 * * * *", "0 11 * * 1"])
    def test_cron_at_or_above_floor_allowed(self, sched):
        pytest.importorskip("croniter")
        job = create_job(prompt="report", schedule=sched, name=f"okcron-{sched}")
        assert job["schedule"]["kind"] == "cron"

    def test_floor_boundary_every_two_minutes_allowed(self):
        # Exactly at the 120s floor must be allowed (strict < comparison).
        job = create_job(prompt="x", schedule="every 2m", name="boundary")
        assert _effective_cadence_seconds(job["schedule"]) == MIN_RECURRING_CADENCE_SECONDS

    def test_adjacent_minute_pair_cron_allowed_not_false_positive(self):
        # '0,1 0 * * *' fires twice a DAY (00:00 and 00:01) — it has a 60s gap
        # between its two fires but is NOT a high-frequency job. The cadence is
        # measured as a 24h average so this is correctly ALLOWED.
        pytest.importorskip("croniter")
        job = create_job(prompt="twice daily", schedule="0,1 0 * * *", name="adjacent")
        assert job["schedule"]["kind"] == "cron"
        assert _effective_cadence_seconds(job["schedule"]) > MIN_RECURRING_CADENCE_SECONDS


# ---------------------------------------------------------------------------
# The explicit opt-out for genuine infrastructure watchdogs
# ---------------------------------------------------------------------------

class TestHighFrequencyOptIn:
    def test_allow_high_frequency_permits_every_minute_interval(self):
        job = create_job(prompt="watchdog", schedule="every 1m", name="wd",
                         allow_high_frequency=True)
        assert job["schedule"]["kind"] == "interval"
        assert job["schedule"]["minutes"] == 1

    def test_allow_high_frequency_permits_every_minute_cron(self):
        pytest.importorskip("croniter")
        job = create_job(prompt="watchdog", schedule="* * * * *", name="wdc",
                         allow_high_frequency=True)
        assert job["schedule"]["kind"] == "cron"


# ---------------------------------------------------------------------------
# The cadence helper itself
# ---------------------------------------------------------------------------

class TestUpdateBypassClosed:
    """The floor must also fire on schedule CHANGE, not just create — otherwise
    an agent could create 'every 5m' then update to 'every 1m'."""

    def test_update_to_every_minute_interval_rejected(self):
        job = create_job(prompt="x", schedule="every 5m", name="upd1")
        with pytest.raises(ValueError):
            update_job(job["id"], {"schedule": "every 1m"})

    def test_update_to_every_minute_cron_rejected(self):
        pytest.importorskip("croniter")
        job = create_job(prompt="x", schedule="every 5m", name="upd2")
        with pytest.raises(ValueError):
            update_job(job["id"], {"schedule": "* * * * *"})

    def test_update_to_legit_cadence_allowed(self):
        job = create_job(prompt="x", schedule="every 5m", name="upd3")
        out = update_job(job["id"], {"schedule": "every 10m"})
        assert out["schedule"]["minutes"] == 10

    def test_update_with_override_allows_high_frequency(self):
        job = create_job(prompt="x", schedule="every 5m", name="upd4")
        out = update_job(job["id"], {"schedule": "every 1m"}, allow_high_frequency=True)
        assert out["schedule"]["minutes"] == 1

    def test_pause_resume_existing_job_not_blocked(self):
        # pause/resume don't change the schedule, so the floor must NOT fire even
        # for a job at/above the floor — a regression guard for pause_job/resume_job
        # which both route through update_job.
        job = create_job(prompt="x", schedule="every 3m", name="upd5")
        paused = pause_job(job["id"])
        assert paused["enabled"] is False
        resumed = resume_job(job["id"])
        assert resumed["enabled"] is True


class TestEffectiveCadence:
    def test_interval_cadence(self):
        assert _effective_cadence_seconds({"kind": "interval", "minutes": 5}) == 300.0
        assert _effective_cadence_seconds({"kind": "interval", "minutes": 1}) == 60.0

    def test_once_has_no_cadence(self):
        assert _effective_cadence_seconds({"kind": "once", "run_at": "x"}) is None

    def test_cron_cadence_every_minute(self):
        pytest.importorskip("croniter")
        assert _effective_cadence_seconds({"kind": "cron", "expr": "* * * * *"}) == 60.0

    def test_cron_cadence_daily(self):
        pytest.importorskip("croniter")
        assert _effective_cadence_seconds({"kind": "cron", "expr": "0 9 * * *"}) == 86400.0

    def test_unknown_kind_fails_open(self):
        # Unknown / uncomputable cadence => None => caller does NOT block.
        assert _effective_cadence_seconds({"kind": "weird"}) is None
