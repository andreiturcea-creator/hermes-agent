"""Write-time defense against replayed platform message ids."""

from hermes_state import SessionDB, _normalize_platform_message_id


def _db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s1", source="telegram")
    return db


def _count(db):
    with db._lock:
        return db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 's1'"
        ).fetchone()[0]


def test_normalizer_rejects_empty_values_and_canonicalizes_ids():
    assert _normalize_platform_message_id(None) is None
    assert _normalize_platform_message_id("") is None
    assert _normalize_platform_message_id("  ") is None
    assert _normalize_platform_message_id(42) == "42"
    assert _normalize_platform_message_id(" 42 ") == "42"


def test_single_message_replay_returns_existing_row_without_counter_increment(tmp_path):
    db = _db(tmp_path)
    try:
        first = db.append_message(
            "s1", role="user", content="original", platform_message_id="100"
        )
        replay = db.append_message(
            "s1", role="user", content="duplicate", platform_message_id="100"
        )
        assert replay == first
        assert _count(db) == 1
        assert db.get_session("s1")["message_count"] == 1
    finally:
        db.close()


def test_batch_replay_skips_the_entire_turn_tail(tmp_path):
    db = _db(tmp_path)
    try:
        original = [
            {"role": "user", "content": "hello", "platform_message_id": "200"},
            {"role": "assistant", "content": "first reply"},
        ]
        replay = [
            {"role": "user", "content": "hello again", "platform_message_id": "200"},
            {"role": "assistant", "content": "duplicate reply"},
        ]
        assert db.append_messages_batch("s1", original) == 2
        assert db.append_messages_batch("s1", replay) == 0
        assert _count(db) == 2
        assert db.get_session("s1")["message_count"] == 2
        contents = [m["content"] for m in db.get_messages("s1")]
        assert "duplicate reply" not in contents
    finally:
        db.close()


def test_empty_ids_store_as_null_and_do_not_collide(tmp_path):
    db = _db(tmp_path)
    try:
        db.append_message("s1", role="user", content="one", platform_message_id="")
        db.append_message("s1", role="user", content="two", platform_message_id="  ")
        assert _count(db) == 2
        with db._lock:
            empties = db._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE platform_message_id = ''"
            ).fetchone()[0]
        assert empties == 0
    finally:
        db.close()


def test_same_id_in_different_sessions_is_not_a_replay(tmp_path):
    db = _db(tmp_path)
    try:
        db.create_session(session_id="s2", source="telegram")
        db.append_message("s1", role="user", content="one", platform_message_id="300")
        db.append_message("s2", role="user", content="two", platform_message_id="300")
        assert db.message_count("s1") == 1
        assert db.message_count("s2") == 1
    finally:
        db.close()
