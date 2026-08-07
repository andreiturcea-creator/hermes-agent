"""Regression guard for the venice custom-provider 401 (local-patch: venice-pool-seed-guard).

Root cause it locks down: hermes_cli.config._expand_env_vars keeps an UNEXPANDED
"${VAR}" literal whenever the env var is absent from os.environ at config-cache
time (a cron/worker thread with a stripped env, or load_config caching before
env_loader.load_hermes_dotenv). _seed_custom_pool then seeded that truthy literal
as the pool's access_token, which _try_resolve_from_custom_pool returns *before*
the correct key_env fallback -> the gateway sent "Authorization: Bearer
${VENICE_API_KEY}" -> HTTP 401 "Authentication failed" -> the entry was marked
exhausted and persisted, recurring across restarts.

The fix: the seeder never seeds a "${...}" template, and instead resolves key_env
directly from ~/.hermes/.env (export-stripped, os.environ/scope-independent) via
get_env_value_prefer_dotenv; has_usable_secret also rejects "${...}" so the
secondary (non-pool) candidate path can't select the literal either.
"""

import agent.credential_pool as cp
import hermes_cli.auth as auth
import hermes_cli.config as cfg


# A stand-in "real" 63-char venice-shaped key. Never a real secret.
FAKE_KEY = "VE_test_realkey_0123456789_abcdefghij_KLMNOPQRSTUV_63charsxxxx"


def test_has_usable_secret_rejects_unexpanded_template():
    assert auth.has_usable_secret("${VENICE_API_KEY}") is False
    assert auth.has_usable_secret("${ANYTHING}") is False


def test_has_usable_secret_still_accepts_real_keys():
    assert auth.has_usable_secret(FAKE_KEY) is True
    assert auth.has_usable_secret("sk-abc1234567") is True


def test_has_usable_secret_still_rejects_placeholders():
    assert auth.has_usable_secret("changeme") is False
    assert auth.has_usable_secret("***") is False
    assert auth.has_usable_secret("") is False


def _poisoned_cfg(_pool_key):
    return {
        "name": "venice",
        "base_url": "https://api.venice.ai/api/v1",
        "api_key": "${VENICE_API_KEY}",  # unexpanded literal survives config expansion
        "key_env": "VENICE_API_KEY",
    }


def test_seed_falls_back_to_dotenv_when_api_key_is_unexpanded(monkeypatch):
    monkeypatch.setattr(cp, "_get_custom_provider_config", _poisoned_cfg)
    monkeypatch.setattr(
        cfg, "get_env_value_prefer_dotenv",
        lambda k: FAKE_KEY if k == "VENICE_API_KEY" else None,
    )
    entries = []
    cp._seed_custom_pool("custom:venice", entries)
    assert entries, "a healthy entry should be seeded from key_env"
    assert entries[0].access_token == FAKE_KEY
    assert not entries[0].access_token.startswith("${")


def test_seed_never_stores_literal_when_key_unresolvable(monkeypatch):
    monkeypatch.setattr(cp, "_get_custom_provider_config", _poisoned_cfg)
    monkeypatch.setattr(cfg, "get_env_value_prefer_dotenv", lambda k: None)
    entries = []
    cp._seed_custom_pool("custom:venice", entries)
    # Either no entry, or definitely not the poisoned literal -> resolution then
    # falls through to the correct runtime key_env fallback.
    assert all(not (e.access_token or "").startswith("${") for e in entries)


def test_seed_with_already_expanded_key_still_works(monkeypatch):
    monkeypatch.setattr(
        cp, "_get_custom_provider_config",
        lambda pk: {
            "name": "venice",
            "base_url": "https://api.venice.ai/api/v1",
            "api_key": FAKE_KEY,
            "key_env": "VENICE_API_KEY",
        },
    )
    entries = []
    cp._seed_custom_pool("custom:venice", entries)
    assert entries and entries[0].access_token == FAKE_KEY
