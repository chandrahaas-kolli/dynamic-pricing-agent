import json
import sqlite3

import pytest

from src.staging import (
    init_db, record_decision, find_decision, upsert_monthly_state, get_monthly_state,
    open_escalation, resolve_escalation, has_open_escalation,
)


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    init_db(connection)
    yield connection
    connection.close()


def test_init_db_twice_raises_nothing(conn):
    init_db(conn)


def test_record_decision_round_trip(conn):
    brief = {"recommendation": "multi_month_path", "rationale": "Gradual is safer."}
    record_decision(
        conn, "g4", "01-09-2018 10:00", "escalated", None,
        "cap_exceeded", "medium", "Escalated (medium): target beyond the cap.", brief,
    )
    row = conn.execute(
        "SELECT product_id, observed_at, outcome, price, escalation_cause, tier, reason, brief, created_at "
        "FROM decisions"
    ).fetchone()
    product_id, observed_at, outcome, price, escalation_cause, tier, reason, brief_json, created_at = row
    assert product_id == "g4"
    assert outcome == "escalated"
    assert price is None
    assert escalation_cause == "cap_exceeded"
    assert tier == "medium"
    assert reason == "Escalated (medium): target beyond the cap."
    assert json.loads(brief_json) == brief
    assert created_at is not None


def test_record_decision_no_brief_stores_null_and_keeps_price(conn):
    record_decision(
        conn, "g4", "01-09-2018 10:00", "priced", 42.5,
        None, "low", "Priced at 42.50: target is the mid of the competitor band (42.50).", None,
    )
    row = conn.execute("SELECT price, brief FROM decisions").fetchone()
    price, brief_json = row
    assert price == 42.5
    assert brief_json is None


def test_record_decision_duplicate_product_and_observed_at_raises(conn):
    record_decision(conn, "g4", "01-09-2018 10:00", "priced", 42.5, None, "low", "Priced at 42.50.", None)
    with pytest.raises(sqlite3.IntegrityError):
        record_decision(conn, "g4", "01-09-2018 10:00", "priced", 43.0, None, "low", "Priced at 43.00.", None)


def test_record_decision_same_product_different_observed_at_inserts_fine(conn):
    record_decision(conn, "g4", "01-09-2018 10:00", "priced", 42.5, None, "low", "Priced at 42.50.", None)
    record_decision(conn, "g4", "02-09-2018 10:00", "priced", 43.0, None, "low", "Priced at 43.00.", None)
    count = conn.execute("SELECT COUNT(*) FROM decisions WHERE product_id = ?", ("g4",)).fetchone()[0]
    assert count == 2


def test_record_decision_same_date_different_time_inserts_fine(conn):
    record_decision(conn, "g4", "01-09-2018 10:00", "priced", 42.5, None, "low", "Priced at 42.50.", None)
    record_decision(conn, "g4", "01-09-2018 14:30", "priced", 43.0, None, "low", "Priced at 43.00.", None)
    count = conn.execute("SELECT COUNT(*) FROM decisions WHERE product_id = ?", ("g4",)).fetchone()[0]
    assert count == 2


def test_find_decision_round_trip(conn):
    brief = {"recommendation": "multi_month_path", "rationale": "Gradual is safer."}
    record_decision(
        conn, "g4", "01-09-2018 10:00", "escalated", None,
        "cap_exceeded", "medium", "Escalated (medium): target beyond the cap.", brief,
    )
    decision = find_decision(conn, "g4", "01-09-2018 10:00")
    assert decision["product_id"] == "g4"
    assert decision["outcome"] == "escalated"
    assert decision["brief"] == brief


def test_find_decision_missing_returns_none(conn):
    assert find_decision(conn, "missing", "01-09-2018 10:00") is None


def test_find_decision_no_brief_returns_none(conn):
    record_decision(conn, "g4", "01-09-2018 10:00", "priced", 42.5, None, "low", "Priced at 42.50.", None)
    decision = find_decision(conn, "g4", "01-09-2018 10:00")
    assert decision["brief"] is None


def test_upsert_monthly_state_updates_current_price_not_anchor(conn):
    upsert_monthly_state(conn, "g4", "2018-09", 30.0, 30.0)
    upsert_monthly_state(conn, "g4", "2018-09", 999.0, 32.5)
    state = get_monthly_state(conn, "g4", "2018-09")
    assert state["anchor_price"] == 30.0
    assert state["current_price"] == 32.5


def test_get_monthly_state_missing_returns_none(conn):
    assert get_monthly_state(conn, "missing", "2018-09") is None


def test_has_open_escalation_false_with_no_rows(conn):
    assert has_open_escalation(conn, "g4") is False


def test_has_open_escalation_true_after_open_false_after_resolve(conn):
    escalation_id = open_escalation(conn, "g4", "cap_exceeded")
    assert has_open_escalation(conn, "g4") is True
    resolve_escalation(conn, escalation_id)
    assert has_open_escalation(conn, "g4") is False


def test_has_open_escalation_does_not_count_another_product(conn):
    open_escalation(conn, "g4", "cap_exceeded")
    assert has_open_escalation(conn, "other") is False
