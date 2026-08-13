# PRD — failure-only cron delivery

| | |
|---|---|
| Date | 2026-08-11 |
| Author | [Codex] |
| Status | APPROVED 2026-08-11 (Paul: “stop Aster from sending me broken cron jobs in telegram… errors land [in Matrix] and get picked up automatically”) |
| Component | `cron/scheduler.py` |

## Goal and end state

A cron job keeps its configured success destination. If that run fails, the
scheduler sends only the failure summary to `cron.failure_deliver`. Paul's
configuration points that at Hermes Errors in Matrix, so Telegram remains quiet
without changing useful successful briefs. An unavailable/invalid override
records a delivery error and never falls back to the original chat.

## Must / won't

- Must apply after the true run outcome is known, including interrupted runs.
- Must use an ephemeral delivery copy and never mutate the stored job or origin.
- Must support explicit Matrix targets and `local` while preserving empty-config
  compatibility.
- Won't change successful delivery, silently fall back, read credentials, or
  restart the gateway.

## Acceptance evals

1. A failed Telegram job reaches only the configured Matrix target.
2. The identical successful job still reaches Telegram.
3. `local` resolves to zero chat targets and empty config keeps old behavior.
4. The focused scheduler tests and complete cron suite pass at the exact SHA.
5. After the explicit restart gate, a synthetic failed cron produces no
   Telegram message and one Hermes Errors message.

## Risk and rollback

The code/config change is reversible, but loading it requires a protected
gateway restart. Keep activation gated until the exact merged SHA and config are
verified. Rollback restores the previous scheduler release/config and restarts
once through the guarded gateway path; cron output/history is never deleted.

## Assumptions

- `failure_deliver` is global because Paul wants one dedicated errors room.
- Success delivery remains unchanged because the request targets broken runs.
- No fallback is safer than reintroducing Telegram error spam.
