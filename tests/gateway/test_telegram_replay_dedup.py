"""Durable Telegram update-offset replay guard."""

import asyncio
import types
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter(name="telegram"):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._telegram_offset_key = lambda: name
    return adapter


def _event(update_id):
    return types.SimpleNamespace(platform_update_id=update_id)


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    return tmp_path


def test_offset_persists_across_adapter_restart(profile_home):
    first = _adapter()
    first._note_platform_update_processed(_event(100))

    offset_file = profile_home / ".telegram_offset.json"
    assert offset_file.exists()
    assert offset_file.stat().st_mode & 0o777 == 0o600

    restarted = _adapter()
    assert restarted._load_telegram_offset() == 100
    assert restarted._is_replayed_platform_update(_event(100)) is True
    assert restarted._is_replayed_platform_update(_event(99)) is True
    assert restarted._is_replayed_platform_update(_event(101)) is False


def test_offset_only_advances_and_bot_keys_are_isolated(profile_home):
    first = _adapter("bot-a")
    second = _adapter("bot-b")
    first._note_platform_update_processed(_event(50))
    first._note_platform_update_processed(_event(40))
    second._note_platform_update_processed(_event(900))

    assert _adapter("bot-a")._load_telegram_offset() == 50
    assert _adapter("bot-b")._load_telegram_offset() == 900


def test_missing_or_non_integer_update_id_is_ignored(profile_home):
    adapter = _adapter()
    adapter._note_platform_update_processed(_event(None))
    assert adapter._telegram_offset_high is None
    assert adapter._is_replayed_platform_update(_event(None)) is False
    assert not (profile_home / ".telegram_offset.json").exists()


def test_merged_batch_retains_highest_constituent_update_id():
    existing = _event(300)

    TelegramAdapter._merge_platform_update_high_watermark(existing, _event(302))
    TelegramAdapter._merge_platform_update_high_watermark(existing, _event(301))

    assert existing.platform_update_id == 302


@pytest.mark.asyncio
async def test_text_batch_retains_highest_update_id(profile_home, monkeypatch):
    adapter = _adapter()
    adapter._text_batch_delay_seconds = 60
    adapter._text_batch_split_delay_seconds = 60
    monkeypatch.setattr(adapter, "_text_batch_key", lambda _event: "session")

    first = types.SimpleNamespace(
        text="part one",
        platform_update_id=400,
        media_urls=[],
        media_types=[],
    )
    second = types.SimpleNamespace(
        text="part two",
        platform_update_id=401,
        media_urls=[],
        media_types=[],
    )
    adapter._enqueue_text_event(first)
    adapter._enqueue_text_event(second)

    assert adapter._pending_text_batches["session"].platform_update_id == 401
    task = adapter._pending_text_batch_tasks.pop("session")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_replayed_update_never_reaches_message_handler(profile_home):
    first = _adapter()
    first._note_platform_update_processed(_event(200))

    restarted = _adapter()
    restarted._message_handler = AsyncMock()
    await restarted.handle_message(_event(200))
    restarted._message_handler.assert_not_called()


def test_base_adapter_hooks_remain_inert():
    event = _event(1)
    assert BasePlatformAdapter._is_replayed_platform_update(object(), event) is False
    assert BasePlatformAdapter._note_platform_update_processed(object(), event) is None
