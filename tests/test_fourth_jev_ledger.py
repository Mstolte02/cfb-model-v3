from fourth_jev.ledger import ledger_key, state_hash


def test_state_hash_is_order_independent():
    assert state_hash({"a": 1, "b": 2}) == state_hash({"b": 2, "a": 1})


def test_ledger_key_changes_with_as_of():
    state = {"game": {"home": "A", "away": "B"}}
    a = ledger_key("1", "2026-09-01T12:00:00Z", state, "jev-latest")
    b = ledger_key("1", "2026-09-01T13:00:00Z", state, "jev-latest")
    assert a != b
