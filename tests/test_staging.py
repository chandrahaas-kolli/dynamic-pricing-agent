import json
import sqlite3

import pytest

from src.staging import init_db, record_decision, upsert_monthly_state, get_monthly_state


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
        conn, "g4", "01-09-2018", "escalated", None,
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
        conn, "g4", "01-09-2018", "priced", 42.5,
        None, "low", "Priced at 42.50: target is the mid of the competitor band (42.50).", None,
    )
    row = conn.execute("SELECT price, brief FROM decisions").fetchone()
    price, brief_json = row
    assert price == 42.5
    assert brief_json is None


def test_upsert_monthly_state_updates_current_price_and_pending_not_anchor(conn):
    upsert_monthly_state(conn, "g4", "2018-09", 30.0, 30.0, 0)
    upsert_monthly_state(conn, "g4", "2018-09", 999.0, 32.5, 1)
    state = get_monthly_state(conn, "g4", "2018-09")
    assert state["anchor_price"] == 30.0
    assert state["current_price"] == 32.5
    assert state["pending_escalation"] is True


def test_get_monthly_state_missing_returns_none(conn):
    assert get_monthly_state(conn, "missing", "2018-09") is None


def test_upsert_monthly_state_pending_escalation_never_clears_on_its_own(conn):
    upsert_monthly_state(conn, "g4", "2018-09", 30.0, 30.0, 1)
    upsert_monthly_state(conn, "g4", "2018-09", 30.0, 31.0, 0)
    state = get_monthly_state(conn, "g4", "2018-09")
    assert state["pending_escalation"] is True
