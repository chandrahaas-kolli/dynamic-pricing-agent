import json
import os
import sqlite3
from datetime import datetime

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from src import staging
from src.agent import (
    app, escalate_node, competitor_move_check_node, check_move_size_node,
    reason_node, apply_dominance_clamp_node, llm_step_size_node,
    build_competitors_text, describe_target, build_market_move_text,
    _cap_brief, _market_brief, get_connection, gate_node, route_gate,
    validate_input_node, pair_competitors_node, analyze_position_node,
    resolve_target_price_node, compute_step_node, enforce_bounds_node,
    no_action_node, persist_node,
    route_comp_move_check, route_check_move_size, route_llm_step_size,
)
from src.llm_schemas import StepDecision, CapExceededBrief, MarketMoveBrief
from src.pricing import observation_fingerprint
from src.prompts import STEP_SIZE_SYSTEM_PROMPT


def _client_error(code, message="failure"):
    return ClientError({"Error": {"Code": code, "Message": message}}, "Converse")


class _FakeStepLLM:
    """Stand-in for get_step_llm()'s return value: records the messages it
    was invoked with and returns a preset with_structured_output-style dict.

    Pass a single dict/result to return it on every call, or a list to pop
    one outcome per call (an outcome that's an Exception instance is raised
    instead of returned) — mirrors Mock's side_effect for a sequence of
    retry attempts.
    """
    def __init__(self, result):
        if isinstance(result, list):
            self._side_effects = iter(result)
            self._result = None
        else:
            self._side_effects = None
            self._result = result
        self.messages = None
        self.call_count = 0

    def invoke(self, messages):
        self.messages = messages
        self.call_count += 1
        if self._side_effects is not None:
            outcome = next(self._side_effects)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return self._result


# escalate_node now calls _cap_brief/_market_brief whenever needs_brief is
# True, which would otherwise call Bedrock for real. Autouse so every test
# in this file gets a default success, not just the ones that mention it.
@pytest.fixture(autouse=True)
def _mock_brief_llms(monkeypatch):
    cap_fake = _FakeStepLLM({
        "parsed": CapExceededBrief(recommendation="multi_month_path", rationale="Gradual is safer."),
        "parsing_error": None, "raw": None,
    })
    market_fake = _FakeStepLLM({
        "parsed": MarketMoveBrief(likely_cause="market_shift", rationale="Both movers rose together."),
        "parsing_error": None, "raw": None,
    })
    monkeypatch.setattr("src.agent.get_cap_brief_llm", lambda: cap_fake)
    monkeypatch.setattr("src.agent.get_market_brief_llm", lambda: market_fake)


# persist_node now calls get_connection() on every run, which would otherwise
# open (and create) data/staging.db. Autouse so every test gets a fresh
# in-memory DB instead, and no test writes to the real file.
@pytest.fixture(autouse=True)
def _mock_staging_db(monkeypatch):
    conn = sqlite3.connect(":memory:")
    staging.init_db(conn)
    monkeypatch.setattr("src.agent.get_connection", lambda: conn)
    yield conn
    conn.close()


# --- get_connection ---
def test_get_connection_lazy_then_creates_all_tables(tmp_path, monkeypatch):
    db_path = str(tmp_path / "staging.db")
    monkeypatch.setattr("src.agent.STAGING_DB_PATH", db_path)
    get_connection.cache_clear()

    assert not os.path.exists(db_path)

    conn = get_connection()
    try:
        assert os.path.exists(db_path)
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert tables == {"decisions", "monthly_state", "escalations"}
    finally:
        conn.close()
        get_connection.cache_clear()


# --- gate_node ---
def _gate_state(**overrides):
    state = {
        "product_id": "g4",
        "comp_prices": [80, 40, 35],
        "comp_ratings": [4.5, 4.0, 3.8],
        "our_rating": 4.0,
        "observed_at": "01-09-2018 10:00",
    }
    state.update(overrides)
    return state

def test_gate_node_empty_db_proceeds(_mock_staging_db):
    result = gate_node(_gate_state())
    assert result == {"gate": "proceed"}

def test_gate_node_same_key_same_inputs_duplicate(_mock_staging_db):
    conn = _mock_staging_db
    state = _gate_state()
    input_hash = observation_fingerprint(state["comp_prices"], state["comp_ratings"], state["our_rating"])
    inputs = {
        "comp_prices": state["comp_prices"], "comp_ratings": state["comp_ratings"], "our_rating": state["our_rating"],
    }
    staging.record_decision(
        conn, state["product_id"], state["observed_at"], "priced", 42.5,
        None, "low", "Priced at 42.50.", None, input_hash, inputs,
    )
    result = gate_node(state)
    assert result["gate"] == "duplicate"
    assert result["stored_decision"]["product_id"] == "g4"
    assert result["stored_decision"]["outcome"] == "priced"

def test_gate_node_same_key_different_comp_prices_conflict(_mock_staging_db):
    conn = _mock_staging_db
    state = _gate_state()
    stored_comp_prices = [80, 41, 35]
    stored_inputs = {
        "comp_prices": stored_comp_prices, "comp_ratings": state["comp_ratings"], "our_rating": state["our_rating"],
    }
    stored_input_hash = observation_fingerprint(stored_comp_prices, state["comp_ratings"], state["our_rating"])
    staging.record_decision(
        conn, state["product_id"], state["observed_at"], "priced", 42.5,
        None, "low", "Priced at 42.50.", None, stored_input_hash, stored_inputs,
    )
    result = gate_node(state)
    assert result["gate"] == "conflict"
    assert result["escalation_cause"] == "data_conflict"
    assert result["tier"] == "medium"
    assert result["conflict"]["stored"] == stored_inputs
    assert result["conflict"]["incoming"] == {
        "comp_prices": state["comp_prices"], "comp_ratings": state["comp_ratings"], "our_rating": state["our_rating"],
    }
    assert result["conflict"]["stored_outcome"] == "priced"

def test_gate_node_new_key_open_escalation_held(_mock_staging_db):
    conn = _mock_staging_db
    staging.open_escalation(conn, "g4", "cap_exceeded")
    result = gate_node(_gate_state())
    assert result == {"gate": "held"}

def test_gate_node_same_key_same_inputs_and_open_escalation_still_duplicate(_mock_staging_db):
    conn = _mock_staging_db
    state = _gate_state()
    input_hash = observation_fingerprint(state["comp_prices"], state["comp_ratings"], state["our_rating"])
    inputs = {
        "comp_prices": state["comp_prices"], "comp_ratings": state["comp_ratings"], "our_rating": state["our_rating"],
    }
    staging.record_decision(
        conn, state["product_id"], state["observed_at"], "priced", 42.5,
        None, "low", "Priced at 42.50.", None, input_hash, inputs,
    )
    staging.open_escalation(conn, "g4", "cap_exceeded")
    result = gate_node(state)
    assert result["gate"] == "duplicate"


# --- route_gate ---
def test_route_gate_duplicate():
    assert route_gate({"gate": "duplicate"}) == "duplicate"

def test_route_gate_conflict():
    assert route_gate({"gate": "conflict"}) == "conflict"

def test_route_gate_held():
    assert route_gate({"gate": "held"}) == "held"

def test_route_gate_proceed():
    assert route_gate({"gate": "proceed"}) == "proceed"

def test_route_gate_unknown_raises():
    with pytest.raises(ValueError):
        route_gate({"gate": "something_else"})


# --- escalate_node ---
def test_escalate_node_competitor_move_single_trigger():
    state = {
        "escalation_cause": "competitor_move",
        "triggered": [{"index": 0, "comp_price_new": 100, "comp_price_prev": 74, "percent_change": 35.1}],
        "comp_prices": [100, 39.5, 39.0],
        "prev_comp_prices": [74, 39.24, 39.24],
    }
    result = escalate_node(state)
    assert result["needs_brief"] is False
    assert result["brief"] is None
    assert set(result["detail_pack"].keys()) == {"triggered", "comp_prices", "prev_comp_prices"}

def test_escalate_node_competitor_move_double_trigger():
    state = {
        "escalation_cause": "competitor_move",
        "triggered": [
            {"index": 0, "comp_price_new": 100, "comp_price_prev": 74, "percent_change": 35.1},
            {"index": 1, "comp_price_new": 55, "comp_price_prev": 39.24, "percent_change": 40.2},
        ],
        "comp_prices": [100, 55, 39.0],
        "prev_comp_prices": [74, 39.24, 39.24],
        "comp_ratings": [4.5, 4.0, 3.8],
        "percent_diff": [35.1, 40.2, -0.6],
    }
    result = escalate_node(state)
    assert result["needs_brief"] is True

def _cap_escalation_state(**overrides):
    state = {
        "escalation_cause": "cap_exceeded",
        "target_price": 40,
        "target_label": "high",
        "target_source": "band",
        "anchor": 30,
        "x_anchor": 33.33,
        "current_price": 39,
        "band": {"low": 35, "mid": 57.5, "high": 80},
        "gap": -0.1,
        "min_price": 1,
        "max_price": 1000,
        "our_rating": 4.0,
        "comp_details": [
            {"comp_price": 35, "comp_rating": 4.0},
            {"comp_price": 80, "comp_rating": 3.5},
        ],
    }
    state.update(overrides)
    return state

def test_escalate_node_cap_exceeded():
    result = escalate_node(_cap_escalation_state())
    assert result["needs_brief"] is True
    assert set(result["detail_pack"].keys()) == {
        "target_price", "target_source", "anchor", "x_anchor", "current_price", "band", "gap",
    }

def test_escalate_node_unknown_cause():
    with pytest.raises(ValueError):
        escalate_node({"escalation_cause": "something_else"})

def test_escalate_node_missing_cause():
    with pytest.raises(ValueError):
        escalate_node({})

def test_escalate_node_cap_exceeded_brief_filled():
    result = escalate_node(_cap_escalation_state())
    assert result["brief"] == {"recommendation": "multi_month_path", "rationale": "Gradual is safer."}
    assert result["brief_errors"] == []
    assert result["brief_attempts"] == 1

def test_escalate_node_competitor_move_double_trigger_brief_filled():
    state = {
        "escalation_cause": "competitor_move",
        "triggered": [
            {"index": 0, "comp_price_new": 100, "comp_price_prev": 74, "percent_change": 35.1},
            {"index": 1, "comp_price_new": 55, "comp_price_prev": 39.24, "percent_change": 40.2},
        ],
        "comp_prices": [100, 55, 39.0],
        "prev_comp_prices": [74, 39.24, 39.24],
        "comp_ratings": [4.5, 4.0, 3.8],
        "percent_diff": [35.1, 40.2, -0.6],
    }
    result = escalate_node(state)
    assert result["brief"] == {"likely_cause": "market_shift", "rationale": "Both movers rose together."}
    assert result["brief_errors"] == []
    assert result["brief_attempts"] == 1

def test_escalate_node_competitor_move_single_trigger_no_brief_keys():
    state = {
        "escalation_cause": "competitor_move",
        "triggered": [{"index": 0, "comp_price_new": 100, "comp_price_prev": 74, "percent_change": 35.1}],
        "comp_prices": [100, 39.5, 39.0],
        "prev_comp_prices": [74, 39.24, 39.24],
    }
    result = escalate_node(state)
    assert result["brief"] is None
    assert "brief_errors" not in result
    assert "brief_attempts" not in result

def test_escalate_node_llm_failed_no_brief_keys():
    state = {
        "escalation_cause": "llm_failed",
        "llm_errors": ["ThrottlingException: failure"] * 3,
        "llm_attempts": 3,
    }
    result = escalate_node(state)
    assert result["needs_brief"] is False
    assert result["brief"] is None
    assert "brief_errors" not in result
    assert "brief_attempts" not in result

def test_escalate_node_data_conflict_no_brief():
    conflict = {
        "stored": {"comp_prices": [80, 40, 35], "comp_ratings": [4.5, 4.0, 3.8], "our_rating": 4.0},
        "incoming": {"comp_prices": [80, 41, 35], "comp_ratings": [4.5, 4.0, 3.8], "our_rating": 4.0},
        "stored_outcome": "priced",
    }
    state = {"escalation_cause": "data_conflict", "conflict": conflict}
    result = escalate_node(state)
    assert result["detail_pack"] == conflict
    assert result["needs_brief"] is False
    assert result["brief"] is None
    assert "brief_errors" not in result
    assert "brief_attempts" not in result

def test_escalate_node_cap_exceeded_brief_fails_detail_pack_unchanged(monkeypatch):
    fake = _FakeStepLLM([{"parsed": None, "parsing_error": None, "raw": None}] * 3)
    monkeypatch.setattr("src.agent.get_cap_brief_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    result = escalate_node(_cap_escalation_state())
    assert result["brief"] is None
    assert result["brief_errors"] == ["invalid output: no structured answer"] * 3
    assert result["brief_attempts"] == 3
    assert set(result["detail_pack"].keys()) == {
        "target_price", "target_source", "anchor", "x_anchor", "current_price", "band", "gap",
    }


# --- competitor_move_check_node ---
def test_competitor_move_check_node_sets_cause_on_trigger():
    state = {"product_id": "g4", "comp_prices": [100, 39.5, 39.0], "prev_comp_prices": [74, 39.24, 39.24]}
    result = competitor_move_check_node(state)
    assert result["escalation_cause"] == "competitor_move"
    assert result["action"] == "trigger"

def test_competitor_move_check_node_no_cause_on_proceed():
    state = {"product_id": "g4", "comp_prices": [105, 120, 90], "prev_comp_prices": [100, 100, 100]}
    result = competitor_move_check_node(state)
    assert "escalation_cause" not in result
    assert result["action"] == "proceed"

def test_competitor_move_check_node_no_cause_on_none():
    state = {"product_id": "g4", "comp_prices": [101, 100, 99], "prev_comp_prices": [100, 100, 100]}
    result = competitor_move_check_node(state)
    assert "escalation_cause" not in result
    assert result["action"] == "none"


# --- check_move_size_node ---
def test_check_move_size_node_sets_cause_on_escalate():
    state = {"target_price": 40, "anchor": 30, "target_source": "band"}
    result = check_move_size_node(state)
    assert result["escalation_cause"] == "cap_exceeded"
    assert result["path"] == "escalate"

def test_check_move_size_node_no_cause_on_llm():
    state = {"target_price": 40, "anchor": 39, "target_source": "band"}
    result = check_move_size_node(state)
    assert "escalation_cause" not in result
    assert result["path"] == "llm"

def test_check_move_size_node_no_cause_on_direct():
    state = {"target_price": 40, "anchor": 37, "target_source": "band"}
    result = check_move_size_node(state)
    assert "escalation_cause" not in result
    assert result["path"] == "direct"

def test_check_move_size_node_tie_source_still_escalates():
    state = {"target_price": 40, "anchor": 30, "target_source": "tie_match"}
    result = check_move_size_node(state)
    assert result["path"] == "escalate"
    assert result["escalation_cause"] == "cap_exceeded"


# --- llm_step_size_node (get_step_llm mocked; no test may call Bedrock) ---
def _llm_state(**overrides):
    state = {
        "current_price": 90,
        "target_price": 96,
        "anchor": 92,
        "our_rating": 4.0,
        "comp_details": [
            {"comp_price": 98, "comp_rating": 4.5},
            {"comp_price": 92, "comp_rating": 4.0},
            {"comp_price": 80, "comp_rating": 3.5},
        ],
    }
    state.update(overrides)
    return state

def test_llm_step_size_node_success(monkeypatch):
    fake = _FakeStepLLM({
        "parsed": StepDecision(step_pct=0.02, rationale="Room below the higher-rated competitor at 98."),
        "parsing_error": None, "raw": None,
    })
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    result = llm_step_size_node(_llm_state())
    assert result == {
        "llm_step_pct": 0.02,
        "llm_rationale": "Room below the higher-rated competitor at 98.",
        "llm_attempts": 1,
        "llm_errors": [],
    }

def test_llm_step_size_node_parsing_error_escalates(monkeypatch):
    fake = _FakeStepLLM({"parsed": None, "parsing_error": ValueError("bad output"), "raw": None})
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    result = llm_step_size_node(_llm_state())
    assert result["escalation_cause"] == "llm_failed"
    assert fake.call_count == 3
    assert result["llm_errors"] == ["invalid output: failed schema validation"] * 3

def test_llm_step_size_node_parsed_none_no_error_escalates(monkeypatch):
    fake = _FakeStepLLM({"parsed": None, "parsing_error": None, "raw": None})
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    result = llm_step_size_node(_llm_state())
    assert result["escalation_cause"] == "llm_failed"
    assert fake.call_count == 3
    assert result["llm_errors"] == ["invalid output: no structured answer"] * 3

def test_llm_step_size_node_at_target_no_call(monkeypatch):
    fake = _FakeStepLLM({"parsed": StepDecision(step_pct=0.02, rationale="x"), "parsing_error": None, "raw": None})
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    result = llm_step_size_node(_llm_state(current_price=96, target_price=96))
    assert result == {}
    assert fake.messages is None

def test_llm_step_size_node_human_text_upward(monkeypatch):
    fake = _FakeStepLLM({"parsed": StepDecision(step_pct=0.02, rationale="x"), "parsing_error": None, "raw": None})
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    llm_step_size_node(_llm_state(current_price=90, target_price=96, anchor=92))
    human_text = fake.messages[1].content
    assert "Direction: up" in human_text
    assert "Current price: 90.00" in human_text
    assert "Target price: 96.00" in human_text
    assert "Anchor price: 92.00" in human_text
    assert f"{abs(96 - 90) / 92:.4f}" in human_text

def test_llm_step_size_node_human_text_downward(monkeypatch):
    fake = _FakeStepLLM({"parsed": StepDecision(step_pct=0.02, rationale="x"), "parsing_error": None, "raw": None})
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    llm_step_size_node(_llm_state(current_price=96, target_price=90, anchor=92))
    human_text = fake.messages[1].content
    assert "Direction: down" in human_text

def test_llm_step_size_node_system_prompt_sent(monkeypatch):
    fake = _FakeStepLLM({"parsed": StepDecision(step_pct=0.02, rationale="x"), "parsing_error": None, "raw": None})
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    llm_step_size_node(_llm_state())
    assert fake.messages[0].content == STEP_SIZE_SYSTEM_PROMPT

def test_build_competitors_text_labels():
    comp_details = [
        {"comp_price": 98, "comp_rating": 4.5},
        {"comp_price": 92, "comp_rating": 4.0},
        {"comp_price": 80, "comp_rating": 3.5},
    ]
    text = build_competitors_text(4.0, comp_details)
    lines = text.split("\n")
    assert lines[0] == "- Competitor 1: price 98.00, rating 4.5 (rated higher than us)"
    assert lines[1] == "- Competitor 2: price 92.00, rating 4.0 (rated the same as us)"
    assert lines[2] == "- Competitor 3: price 80.00, rating 3.5 (rated lower than us)"


# --- describe_target ---
def test_describe_target_band():
    assert describe_target("band", "high") == "the high point of the competitor price band, set by our rating position"

def test_describe_target_tie_match():
    assert describe_target("tie_match", "mid") == "matches the price of a competitor rated the same as us"

def test_describe_target_tie_undercut():
    assert describe_target("tie_undercut", "mid") == "just below the cheapest competitor rated higher than us"

def test_describe_target_top_premium():
    assert describe_target("top_premium", "high") == "a premium above the band's high end, because we are rated above every competitor"

def test_describe_target_unknown_raises():
    with pytest.raises(ValueError):
        describe_target("mystery", "mid")


# --- build_market_move_text ---
def test_build_market_move_text_two_triggered():
    comp_prices = [100, 55, 48]
    prev_comp_prices = [74, 39.24, 50]
    comp_ratings = [4.5, 4.0, 3.8]
    percent_diff = [35.1, 40.2, -4.0]
    triggered = [
        {"index": 0, "comp_price_new": 100, "comp_price_prev": 74, "percent_change": 35.1},
        {"index": 1, "comp_price_new": 55, "comp_price_prev": 39.24, "percent_change": 40.2},
    ]
    text = build_market_move_text(comp_prices, prev_comp_prices, comp_ratings, percent_diff, triggered)
    lines = text.split("\n")
    assert lines[0] == "- Competitor 1: 74.00 -> 100.00 (+35.1%), rating 4.5 [moved 30%+]"
    assert lines[1] == "- Competitor 2: 39.24 -> 55.00 (+40.2%), rating 4.0 [moved 30%+]"
    assert lines[2] == "- Competitor 3: 50.00 -> 48.00 (-4.0%), rating 3.8"


# --- _cap_brief (get_cap_brief_llm mocked; no test may call Bedrock) ---
def _cap_brief_state(**overrides):
    state = {
        "current_price": 90,
        "anchor": 90,
        "target_price": 120,
        "x_anchor": 33.3,
        "min_price": 50,
        "max_price": 150,
        "our_rating": 4.0,
        "target_source": "band",
        "target_label": "high",
        "band": {"low": 100, "mid": 110, "high": 120},
        "comp_details": [
            {"comp_price": 100, "comp_rating": 4.5},
            {"comp_price": 120, "comp_rating": 3.5},
        ],
    }
    state.update(overrides)
    return state

def test_cap_brief_success(monkeypatch):
    fake = _FakeStepLLM({
        "parsed": CapExceededBrief(recommendation="multi_month_path", rationale="Gradual is safer."),
        "parsing_error": None, "raw": None,
    })
    monkeypatch.setattr("src.agent.get_cap_brief_llm", lambda: fake)
    brief, errors, attempts = _cap_brief(_cap_brief_state())
    assert brief == {"recommendation": "multi_month_path", "rationale": "Gradual is safer."}
    assert errors == []
    assert attempts == 1

def test_cap_brief_three_failures_returns_none(monkeypatch):
    fake = _FakeStepLLM([{"parsed": None, "parsing_error": None, "raw": None}] * 3)
    monkeypatch.setattr("src.agent.get_cap_brief_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    brief, errors, attempts = _cap_brief(_cap_brief_state())
    assert brief is None
    assert errors == ["invalid output: no structured answer"] * 3
    assert attempts == 3

def test_cap_brief_human_message_contains_months_and_target_reason(monkeypatch):
    fake = _FakeStepLLM({
        "parsed": CapExceededBrief(recommendation="one_time_jump", rationale="Close enough."),
        "parsing_error": None, "raw": None,
    })
    monkeypatch.setattr("src.agent.get_cap_brief_llm", lambda: fake)
    state = _cap_brief_state()
    _cap_brief(state)
    human_text = fake.messages[1].content
    assert "Months needed at the cap: 4" in human_text
    assert describe_target(state["target_source"], state["target_label"]) in human_text


# --- _market_brief (get_market_brief_llm mocked; no test may call Bedrock) ---
def _market_brief_state(**overrides):
    state = {
        "comp_prices": [100, 55, 48],
        "prev_comp_prices": [74, 39.24, 50],
        "comp_ratings": [4.5, 4.0, 3.8],
        "percent_diff": [35.1, 40.2, -4.0],
        "triggered": [
            {"index": 0, "comp_price_new": 100, "comp_price_prev": 74, "percent_change": 35.1},
            {"index": 1, "comp_price_new": 55, "comp_price_prev": 39.24, "percent_change": 40.2},
        ],
    }
    state.update(overrides)
    return state

def test_market_brief_success(monkeypatch):
    fake = _FakeStepLLM({
        "parsed": MarketMoveBrief(likely_cause="market_shift", rationale="Both movers rose together."),
        "parsing_error": None, "raw": None,
    })
    monkeypatch.setattr("src.agent.get_market_brief_llm", lambda: fake)
    brief, errors, attempts = _market_brief(_market_brief_state())
    assert brief == {"likely_cause": "market_shift", "rationale": "Both movers rose together."}
    assert errors == []
    assert attempts == 1

def test_market_brief_three_failures_returns_none(monkeypatch):
    fake = _FakeStepLLM([{"parsed": None, "parsing_error": None, "raw": None}] * 3)
    monkeypatch.setattr("src.agent.get_market_brief_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    brief, errors, attempts = _market_brief(_market_brief_state())
    assert brief is None
    assert errors == ["invalid output: no structured answer"] * 3
    assert attempts == 3

def test_market_brief_human_message_contains_counts_and_flagged_line(monkeypatch):
    fake = _FakeStepLLM({
        "parsed": MarketMoveBrief(likely_cause="market_shift", rationale="Both movers rose together."),
        "parsing_error": None, "raw": None,
    })
    monkeypatch.setattr("src.agent.get_market_brief_llm", lambda: fake)
    _market_brief(_market_brief_state())
    human_text = fake.messages[1].content
    assert "Competitors that moved 30% or more in this observation: 2 of 3" in human_text
    assert "- Competitor 1: 74.00 -> 100.00 (+35.1%), rating 4.5 [moved 30%+]" in human_text


# --- app.invoke, graph-level ---
base = {
    "product_id": "g4",
    "comp_ratings": [3.9, 4, 4],
    "our_rating": 4.0,
    "observed_at": "01-09-2018 10:00",
    "last_observed_mon_yr": "01-08-2018",
    "cleared_comp": None,
}

def test_app_invoke_double_trigger():
    result = app.invoke({**base, "comp_prices": [100, 55, 39.0],
                          "prev_comp_prices": [74, 39.24, 39.24]})
    assert result["escalation_cause"] == "competitor_move"
    assert result["needs_brief"] is True

def test_app_invoke_double_trigger_brief_filled():
    result = app.invoke({**base, "comp_prices": [100, 55, 39.0],
                          "prev_comp_prices": [74, 39.24, 39.24]})
    assert result["brief"] == {"likely_cause": "market_shift", "rationale": "Both movers rose together."}
    assert result["brief_errors"] == []
    assert result["brief_attempts"] == 1

def test_app_invoke_single_trigger_no_brief():
    result = app.invoke({**base, "comp_prices": [100, 39.5, 39.0],
                          "prev_comp_prices": [74, 39.24, 39.24]})
    assert result["brief"] is None
    assert "brief_errors" not in result

def test_app_invoke_cap_exceeded():
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.0, 3.8],
               "our_rating": 4.0,
               "current_price": 39,
               "min_price": 1,
               "max_price": 1000,
               "anchor": 30}
    result = app.invoke(proceed)
    assert result["escalation_cause"] == "cap_exceeded"
    assert result["needs_brief"] is True
    assert "price" not in result

def test_app_invoke_cap_exceeded_brief_filled():
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.0, 3.8],
               "our_rating": 4.0,
               "current_price": 39,
               "min_price": 1,
               "max_price": 1000,
               "anchor": 30}
    result = app.invoke(proceed)
    assert result["brief"] == {"recommendation": "multi_month_path", "rationale": "Gradual is safer."}
    assert result["brief_errors"] == []
    assert result["brief_attempts"] == 1
    assert "Brief unavailable" not in result["reason"]
    assert "price" not in result

def test_app_invoke_cap_exceeded_brief_fails_reason_unavailable(monkeypatch):
    fake = _FakeStepLLM([{"parsed": None, "parsing_error": None, "raw": None}] * 3)
    monkeypatch.setattr("src.agent.get_cap_brief_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.0, 3.8],
               "our_rating": 4.0,
               "current_price": 39,
               "min_price": 1,
               "max_price": 1000,
               "anchor": 30}
    result = app.invoke(proceed)
    assert result["reason"].endswith("Brief unavailable.")
    assert "price" not in result


# --- reason_node ---
def _priced_state(**overrides):
    state = {
        "action": "proceed",
        "price": 50,
        "current_price": 40,
        "target_price": 50,
        "target_label": "mid",
        "target_source": "band",
        "cleared_comp": None,
        "path": "direct",
        "dominance_clamped": False,
        "dominance_side": None,
        "bounds_clamped": False,
        "min_price": 1,
        "max_price": 1000,
    }
    state.update(overrides)
    return state


# --- apply_dominance_clamp_node ---
def test_apply_dominance_clamp_node_downward_above_lower_rated():
    state = {
        "price": 102, "current_price": 105, "our_rating": 4.0,
        "comp_details": [{"comp_price": 103, "comp_rating": 3.5}],
    }
    result = apply_dominance_clamp_node(state)
    assert round(result["price"], 2) == 103.01
    assert result["dominance_clamped"] is True
    assert result["dominance_side"] == "above_lower_rated"

def test_apply_dominance_clamp_node_upward_below_higher_rated():
    state = {
        "price": 104, "current_price": 100, "our_rating": 4.0,
        "comp_details": [{"comp_price": 102, "comp_rating": 4.5}],
    }
    result = apply_dominance_clamp_node(state)
    assert round(result["price"], 2) == 101.99
    assert result["dominance_clamped"] is True
    assert result["dominance_side"] == "below_higher_rated"

def test_apply_dominance_clamp_node_no_move():
    state = {
        "price": 100, "current_price": 100, "our_rating": 4.0,
        "comp_details": [{"comp_price": 102, "comp_rating": 4.5}],
    }
    result = apply_dominance_clamp_node(state)
    assert result["price"] == 100
    assert result["dominance_clamped"] is False
    assert result["dominance_side"] is None

def test_reason_node_held_exact_text():
    result = reason_node({"gate": "held"})
    assert result["reason"] == "Held: an open escalation is awaiting human review; price unchanged."

def test_reason_node_data_conflict_includes_stored_outcome():
    state = {
        "escalation_cause": "data_conflict",
        "conflict": {
            "stored": {"comp_prices": [80, 40, 35], "comp_ratings": [4.5, 4.0, 3.8], "our_rating": 4.0},
            "incoming": {"comp_prices": [80, 41, 35], "comp_ratings": [4.5, 4.0, 3.8], "our_rating": 4.0},
            "stored_outcome": "priced",
        },
    }
    result = reason_node(state)
    assert "stored outcome: priced" in result["reason"]
    assert "Escalated (medium)" in result["reason"]

def test_reason_node_competitor_move_two_triggered():
    state = {
        "escalation_cause": "competitor_move",
        "triggered": [
            {"index": 0, "comp_price_new": 100, "comp_price_prev": 74, "percent_change": 35.1},
            {"index": 1, "comp_price_new": 55, "comp_price_prev": 39.24, "percent_change": 40.2},
        ],
    }
    result = reason_node(state)
    assert "Escalated (high)" in result["reason"]
    assert "2 competitor(s)" in result["reason"]

def test_reason_node_cap_exceeded_negative_x_anchor():
    state = {
        "escalation_cause": "cap_exceeded",
        "target_price": 30,
        "anchor": 40,
        "x_anchor": -25.0,
    }
    result = reason_node(state)
    assert "-25.0%" in result["reason"]
    assert "±10%" in result["reason"]

def test_reason_node_cap_exceeded_brief_unavailable():
    state = {
        "escalation_cause": "cap_exceeded",
        "target_price": 30,
        "anchor": 40,
        "x_anchor": -25.0,
        "needs_brief": True,
        "brief": None,
    }
    result = reason_node(state)
    assert result["reason"].endswith("Brief unavailable.")

def test_reason_node_cap_exceeded_brief_present_no_unavailable_clause():
    state = {
        "escalation_cause": "cap_exceeded",
        "target_price": 30,
        "anchor": 40,
        "x_anchor": -25.0,
        "needs_brief": True,
        "brief": {"recommendation": "multi_month_path", "rationale": "x"},
    }
    result = reason_node(state)
    assert "Brief unavailable" not in result["reason"]

def test_reason_node_competitor_move_two_triggered_brief_unavailable():
    state = {
        "escalation_cause": "competitor_move",
        "triggered": [
            {"index": 0, "comp_price_new": 100, "comp_price_prev": 74, "percent_change": 35.1},
            {"index": 1, "comp_price_new": 55, "comp_price_prev": 39.24, "percent_change": 40.2},
        ],
        "needs_brief": True,
        "brief": None,
    }
    result = reason_node(state)
    assert result["reason"].endswith("Brief unavailable.")

def test_reason_node_competitor_move_one_triggered_no_unavailable_clause():
    state = {
        "escalation_cause": "competitor_move",
        "triggered": [
            {"index": 0, "comp_price_new": 100, "comp_price_prev": 74, "percent_change": 35.1},
        ],
        "needs_brief": False,
        "brief": None,
    }
    result = reason_node(state)
    assert "Brief unavailable" not in result["reason"]

def test_reason_node_llm_failed_exact_text():
    state = {
        "escalation_cause": "llm_failed",
        "llm_errors": ["AccessDeniedException: denied"],
        "llm_attempts": 1,
    }
    result = reason_node(state)
    assert result["reason"] == "Escalated (medium): step-size model failed after 1 attempt (AccessDeniedException)."

def test_reason_node_unknown_escalation_cause():
    with pytest.raises(ValueError):
        reason_node({"escalation_cause": "something_else"})

def test_reason_node_unknown_target_source():
    state = _priced_state(target_source="mystery")
    with pytest.raises(ValueError):
        reason_node(state)

def test_reason_node_none_exact_text():
    result = reason_node({"action": "none"})
    assert result["reason"] == "No change: competitor moves below the 2% deadband."

def test_reason_node_band_main_sentence():
    state = _priced_state(target_source="band", price=57.5, current_price=39, target_price=57.5, target_label="mid")
    result = reason_node(state)
    assert result["reason"] == "Priced at 57.50: target is the mid of the competitor band (57.50)."

def test_reason_node_tie_match_main_sentence():
    state = _priced_state(target_source="tie_match", price=40, current_price=39, target_price=40)
    result = reason_node(state)
    assert result["reason"] == "Priced at 40.00: target matches an equally rated competitor at 40.00."

def test_reason_node_tie_undercut_main_sentence():
    state = _priced_state(target_source="tie_undercut", price=44.99, current_price=44, target_price=44.99,
                           cleared_comp={"comp_price": 45, "comp_rating": 4.5})
    result = reason_node(state)
    assert result["reason"] == "Priced at 44.99: target undercuts the cheapest higher-rated competitor at 45.00."

def test_reason_node_top_premium_main_sentence():
    state = _priced_state(target_source="top_premium", price=113.10, current_price=100, target_price=113.10)
    result = reason_node(state)
    assert result["reason"] == "Priced at 113.10: target is a top-rated premium over the band high (113.10)."

def test_reason_node_no_change_price_equals_current():
    state = _priced_state(price=39, current_price=39, target_source="band", target_price=57.5)
    result = reason_node(state)
    assert result["reason"] == "No change: price stays at 39.00."

def test_reason_node_llm_partial_step():
    state = _priced_state(path="llm", price=39.39, current_price=39, target_price=40, target_source="tie_match")
    result = reason_node(state)
    assert "Partial step toward" in result["reason"]

def test_reason_node_llm_with_clamp_no_partial():
    state = _priced_state(path="llm", price=97.99, current_price=105, target_price=96, target_source="tie_match",
                           dominance_clamped=True, dominance_side="below_higher_rated")
    result = reason_node(state)
    assert "Partial step toward" not in result["reason"]

def test_reason_node_dominance_clamp_only():
    state = _priced_state(price=101.99, current_price=100, target_price=104, target_source="band",
                           dominance_clamped=True, dominance_side="below_higher_rated", bounds_clamped=False, path="direct")
    result = reason_node(state)
    assert "Clamped below a higher-rated competitor" in result["reason"]

def test_reason_node_ceiling_clamp():
    state = _priced_state(price=55, current_price=56, target_price=57.5, target_source="band",
                           bounds_clamped=True, dominance_clamped=False, max_price=55, min_price=1)
    result = reason_node(state)
    assert "Lowered to the ceiling" in result["reason"]

def test_reason_node_floor_overrides_dominance():
    state = _priced_state(price=1, current_price=105, target_price=96, target_source="tie_match",
                           dominance_clamped=True, dominance_side="below_higher_rated", bounds_clamped=True,
                           min_price=1, max_price=1000)
    result = reason_node(state)
    assert "Raised to the floor" in result["reason"]
    assert "Floor overrode the dominance clamp" in result["reason"]
    assert "Clamped below" not in result["reason"]

def test_reason_node_dominance_clamp_above_lower_rated():
    state = _priced_state(price=90.01, current_price=88, target_price=90, target_source="band",
                           dominance_clamped=True, dominance_side="above_lower_rated", bounds_clamped=False)
    result = reason_node(state)
    assert "Clamped above a lower-rated competitor." in result["reason"]

def test_reason_node_ceiling_overrides_dominance():
    state = _priced_state(price=55, current_price=40, target_price=60, target_source="band",
                           dominance_clamped=True, dominance_side="above_lower_rated", bounds_clamped=True,
                           min_price=1, max_price=55)
    result = reason_node(state)
    assert "Ceiling overrode the dominance clamp." in result["reason"]
    assert "Clamped" not in result["reason"]

def test_reason_node_unknown_dominance_side():
    state = _priced_state(dominance_clamped=True, dominance_side="sideways")
    with pytest.raises(ValueError):
        reason_node(state)

def test_reason_node_llm_rationale_clause():
    state = _priced_state(path="llm", price=96, current_price=90, target_price=96, target_source="band",
                           llm_rationale="Room below the higher-rated competitor at 98.")
    result = reason_node(state)
    assert result["reason"].endswith("Model rationale: Room below the higher-rated competitor at 98.")

def test_reason_node_direct_path_no_rationale_clause():
    state = _priced_state(path="direct", price=96, current_price=90, target_price=96, target_source="band",
                           llm_rationale="Should not appear.")
    result = reason_node(state)
    assert "Model rationale" not in result["reason"]

def test_reason_node_epsilon_formatting():
    state = _priced_state(price=45 - 0.01, current_price=44, target_price=45 - 0.01, target_source="tie_undercut",
                           cleared_comp={"comp_price": 45, "comp_rating": 4.5})
    result = reason_node(state)
    assert "44.99" in result["reason"]


# --- reason_node, graph-level via app.invoke ---
def test_app_invoke_reason_none():
    result = app.invoke({**base, "comp_prices": [75, 39.5, 39.0],
                          "prev_comp_prices": [74, 39.24, 39.24]})
    assert result["reason"]

def test_app_invoke_reason_single_trigger():
    result = app.invoke({**base, "comp_prices": [100, 39.5, 39.0],
                          "prev_comp_prices": [74, 39.24, 39.24]})
    assert result["reason"]

def test_app_invoke_reason_cap_exceeded():
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.0, 3.8],
               "our_rating": 4.0,
               "current_price": 39,
               "min_price": 1,
               "max_price": 1000,
               "anchor": 30}
    result = app.invoke(proceed)
    assert result["reason"]

def test_app_invoke_reason_direct():
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.0, 3.8],
               "our_rating": 4.0,
               "current_price": 39,
               "min_price": 1,
               "max_price": 1000,
               "anchor": 37}
    result = app.invoke(proceed)
    assert result["reason"]

def test_app_invoke_reason_llm(monkeypatch):
    fake = _FakeStepLLM({
        "parsed": StepDecision(step_pct=0.02, rationale="Test rationale."),
        "parsing_error": None, "raw": None,
    })
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.2, 3.8],
               "our_rating": 4.3,
               "current_price": 56,
               "min_price": 1,
               "max_price": 55,
               "anchor": 56}
    result = app.invoke(proceed)
    assert result["llm_step_pct"] == 0.02
    assert "Model rationale: Test rationale." in result["reason"]
    
    
# --- llm_step_size_node: retries and failure -> escalate ---
def _llm_proceed_state(**overrides):
    state = {**base, "comp_prices": [80, 40, 35],
             "prev_comp_prices": [74, 39.24, 39.24],
             "comp_ratings": [4.5, 4.2, 3.8],
             "our_rating": 4.3,
             "current_price": 56,
             "min_price": 1,
             "max_price": 100,
             "anchor": 56}
    state.update(overrides)
    return state

def _step_decision_result(step_pct=0.02, rationale="ok"):
    return {"parsed": StepDecision(step_pct=step_pct, rationale=rationale),
            "parsing_error": None, "raw": None}

# 1. Success on first try -> invoke called once, llm_attempts == 1, graph prices.
def test_llm_step_size_success_first_try_reaches_compute_step(monkeypatch):
    fake = _FakeStepLLM(_step_decision_result())
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    result = app.invoke(_llm_proceed_state())
    assert fake.call_count == 1
    assert result["llm_attempts"] == 1
    assert result["llm_step_pct"] == 0.02
    assert "price" in result

# 2. ThrottlingException x2, then success -> invoke called 3 times, priced normally.
def test_llm_step_size_throttle_twice_then_success(monkeypatch):
    fake = _FakeStepLLM([
        _client_error("ThrottlingException"),
        _client_error("ThrottlingException"),
        _step_decision_result(),
    ])
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    result = llm_step_size_node(_llm_state())
    assert fake.call_count == 3
    assert result["llm_attempts"] == 3
    assert result["llm_step_pct"] == 0.02
    assert result["llm_errors"] == ["ThrottlingException: failure"] * 2

# 3. ThrottlingException x3 -> escalation_cause == llm_failed, no price in state,
#    graph routes through escalate_node/reason_node instead of compute_step.
def test_llm_step_size_throttle_three_times_escalates(monkeypatch):
    fake = _FakeStepLLM([_client_error("ThrottlingException")] * 3)
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    result = app.invoke(_llm_proceed_state())
    assert result["escalation_cause"] == "llm_failed"
    assert result["tier"] == "medium"
    assert "price" not in result
    assert result["reason"].startswith("Escalated")
    assert len(result["llm_errors"]) == 3

# 4. None (no parsed, no error) x3 -> counts as a failure, not a crash.
def test_llm_step_size_none_three_times_escalates(monkeypatch):
    fake = _FakeStepLLM([{"parsed": None, "parsing_error": None, "raw": None}] * 3)
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    result = app.invoke(_llm_proceed_state())
    assert result["escalation_cause"] == "llm_failed"
    assert result["tier"] == "medium"
    assert "price" not in result
    assert len(result["llm_errors"]) == 3

# 5. Non-retryable code -> stops after one attempt instead of retrying.
def test_llm_step_size_non_retryable_code_stops_after_one_attempt(monkeypatch):
    fake = _FakeStepLLM([_client_error("AccessDeniedException", "denied")])
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    result = llm_step_size_node(_llm_state())
    assert fake.call_count == 1
    assert result["llm_attempts"] == 1
    assert result["escalation_cause"] == "llm_failed"
    assert result["llm_errors"] == ["AccessDeniedException: denied"]

# 6. Schema failure, then success -> invoke called 2 times, priced.
def test_llm_step_size_schema_failure_then_success(monkeypatch):
    fake = _FakeStepLLM([
        {"parsed": None, "parsing_error": ValueError("bad"), "raw": None},
        _step_decision_result(),
    ])
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    result = llm_step_size_node(_llm_state())
    assert fake.call_count == 2
    assert result["llm_attempts"] == 2
    assert result["llm_step_pct"] == 0.02

# 7. EndpointConnectionError, then success -> invoke called 2 times, priced.
def test_llm_step_size_connection_error_then_success(monkeypatch):
    fake = _FakeStepLLM([
        EndpointConnectionError(endpoint_url="https://bedrock.amazonaws.com"),
        _step_decision_result(),
    ])
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)
    result = llm_step_size_node(_llm_state())
    assert fake.call_count == 2
    assert result["llm_attempts"] == 2
    assert result["llm_step_pct"] == 0.02

# 8. Three failures -> sleep called exactly twice, with 1 then 2 (never after the last attempt).
def test_llm_step_size_backoff_schedule(monkeypatch):
    fake = _FakeStepLLM([_client_error("ThrottlingException")] * 3)
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)
    sleep_calls = []
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: sleep_calls.append(seconds))
    llm_step_size_node(_llm_state())
    assert sleep_calls == [1, 2]

# 9. llm_failed -> escalate sets needs_brief False, detail_pack has llm_errors and llm_attempts.
def test_escalate_node_llm_failed():
    state = {
        "escalation_cause": "llm_failed",
        "llm_errors": ["ThrottlingException: rate", "ThrottlingException: rate", "ServiceUnavailableException: down"],
        "llm_attempts": 3,
    }
    result = escalate_node(state)
    assert result["needs_brief"] is False
    assert result["detail_pack"] == {"llm_errors": state["llm_errors"], "llm_attempts": 3}


# --- persist_node, graph-level via app.invoke (staging DB mocked by _mock_staging_db) ---
def test_app_invoke_persists_priced_decision_and_monthly_state(_mock_staging_db):
    conn = _mock_staging_db
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.0, 3.8],
               "our_rating": 4.0,
               "current_price": 39,
               "min_price": 1,
               "max_price": 1000,
               "anchor": 37}
    result = app.invoke(proceed)
    outcome, price = conn.execute("SELECT outcome, price FROM decisions").fetchone()
    assert outcome == "priced"
    assert price == result["price"]
    state = conn.execute(
        "SELECT anchor_price, current_price FROM monthly_state WHERE product_id = ? AND month = ?",
        (result["product_id"], "2018-09"),
    ).fetchone()
    assert state == (37.0, result["price"])

def test_app_invoke_persists_input_hash_matches_fingerprint(_mock_staging_db):
    conn = _mock_staging_db
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.0, 3.8],
               "our_rating": 4.0,
               "current_price": 39,
               "min_price": 1,
               "max_price": 1000,
               "anchor": 37}
    app.invoke(proceed)
    input_hash, inputs_json = conn.execute("SELECT input_hash, inputs FROM decisions").fetchone()
    expected = observation_fingerprint(proceed["comp_prices"], proceed["comp_ratings"], proceed["our_rating"])
    assert input_hash == expected
    assert json.loads(inputs_json) == {
        "comp_prices": proceed["comp_prices"],
        "comp_ratings": proceed["comp_ratings"],
        "our_rating": proceed["our_rating"],
    }

def test_app_invoke_persists_cap_exceeded_escalation(_mock_staging_db):
    conn = _mock_staging_db
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.0, 3.8],
               "our_rating": 4.0,
               "current_price": 39,
               "min_price": 1,
               "max_price": 1000,
               "anchor": 30}
    result = app.invoke(proceed)
    outcome, price, brief_json = conn.execute("SELECT outcome, price, brief FROM decisions").fetchone()
    assert outcome == "escalated"
    assert price is None
    assert json.loads(brief_json) == {"recommendation": "multi_month_path", "rationale": "Gradual is safer."}
    assert staging.has_open_escalation(conn, "g4") is True
    detail_json = conn.execute(
        "SELECT detail FROM escalations WHERE product_id = ?", ("g4",)
    ).fetchone()[0]
    assert json.loads(detail_json) == result["detail_pack"]

def test_app_invoke_persists_no_change_decision(_mock_staging_db):
    conn = _mock_staging_db
    app.invoke({**base, "comp_prices": [101, 100, 99], "prev_comp_prices": [100, 100, 100]})
    outcome, price = conn.execute("SELECT outcome, price FROM decisions").fetchone()
    assert outcome == "no_change"
    assert price is None

def test_app_invoke_escalated_run_opens_escalation_priced_run_does_not(_mock_staging_db):
    conn = _mock_staging_db
    priced_proceed = {**base, "comp_prices": [80, 40, 35],
                       "prev_comp_prices": [74, 39.24, 39.24],
                       "comp_ratings": [4.5, 4.0, 3.8],
                       "our_rating": 4.0,
                       "current_price": 39,
                       "min_price": 1,
                       "max_price": 1000,
                       "anchor": 37}
    app.invoke(priced_proceed)
    assert staging.has_open_escalation(conn, "g4") is False

    cap_exceeded_proceed = {**base, "observed_at": "02-09-2018 10:00",
                             "comp_prices": [80, 40, 35],
                             "prev_comp_prices": [74, 39.24, 39.24],
                             "comp_ratings": [4.5, 4.0, 3.8],
                             "our_rating": 4.0,
                             "current_price": 39,
                             "min_price": 1,
                             "max_price": 1000,
                             "anchor": 30}
    app.invoke(cap_exceeded_proceed)
    assert staging.has_open_escalation(conn, "g4") is True

def test_app_invoke_escalate_then_new_observation_held(_mock_staging_db):
    conn = _mock_staging_db
    escalate_proceed = {**base, "comp_prices": [80, 40, 35],
                         "prev_comp_prices": [74, 39.24, 39.24],
                         "comp_ratings": [4.5, 4.0, 3.8],
                         "our_rating": 4.0,
                         "current_price": 39,
                         "min_price": 1,
                         "max_price": 1000,
                         "anchor": 30}
    app.invoke(escalate_proceed)

    # A later observed_at: a distinct observation of the same product, not a
    # re-run of the same one (which the decisions UNIQUE constraint rejects).
    # With an open escalation already on this product, gate_node holds it
    # instead of letting it through to the normal pricing flow.
    held_proceed = {**base, "observed_at": "05-09-2018 10:00",
                     "comp_prices": [80, 40, 35],
                     "prev_comp_prices": [74, 39.24, 39.24],
                     "comp_ratings": [4.5, 4.0, 3.8],
                     "our_rating": 4.0,
                     "current_price": 40,
                     "min_price": 1,
                     "max_price": 1000,
                     "anchor": 38}
    result = app.invoke(held_proceed)
    assert result["gate"] == "held"
    assert result["reason"].startswith("Held")

    anchor_price = conn.execute(
        "SELECT anchor_price FROM monthly_state WHERE product_id = ? AND month = ?",
        ("g4", "2018-09"),
    ).fetchone()[0]
    assert anchor_price == 30.0
    # Nothing in persist_node resolves escalations; only a human does (step 12).
    assert staging.has_open_escalation(conn, "g4") is True


# --- gate wiring, graph-level via app.invoke ---
def test_app_invoke_duplicate_resend_returns_stored_decision(_mock_staging_db):
    conn = _mock_staging_db
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.0, 3.8],
               "our_rating": 4.0,
               "current_price": 39,
               "min_price": 1,
               "max_price": 1000,
               "anchor": 37}
    app.invoke(proceed)
    result = app.invoke(proceed)
    assert result["gate"] == "duplicate"
    assert result["stored_decision"]["product_id"] == "g4"
    count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert count == 1

def test_app_invoke_conflict_same_key_different_comp_prices(_mock_staging_db):
    conn = _mock_staging_db
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.0, 3.8],
               "our_rating": 4.0,
               "current_price": 39,
               "min_price": 1,
               "max_price": 1000,
               "anchor": 37}
    app.invoke(proceed)

    conflicting_proceed = {**proceed, "comp_prices": [80, 41, 35]}
    result = app.invoke(conflicting_proceed)
    assert result["escalation_cause"] == "data_conflict"
    count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert count == 1

    status, cause, detail_json = conn.execute(
        "SELECT status, cause, detail FROM escalations WHERE product_id = ?", ("g4",)
    ).fetchone()
    assert status == "open"
    assert cause == "data_conflict"
    detail = json.loads(detail_json)
    assert "stored" in detail
    assert "incoming" in detail

def test_app_invoke_held_run_records_held_decision_and_skips_llm(_mock_staging_db, monkeypatch):
    conn = _mock_staging_db
    fake = _FakeStepLLM(_step_decision_result())
    monkeypatch.setattr("src.agent.get_step_llm", lambda: fake)

    cap_exceeded_proceed = {**base, "comp_prices": [80, 40, 35],
                             "prev_comp_prices": [74, 39.24, 39.24],
                             "comp_ratings": [4.5, 4.0, 3.8],
                             "our_rating": 4.0,
                             "current_price": 39,
                             "min_price": 1,
                             "max_price": 1000,
                             "anchor": 30}
    app.invoke(cap_exceeded_proceed)

    held_proceed = {**base, "observed_at": "05-09-2018 10:00",
                     "comp_prices": [80, 40, 35],
                     "prev_comp_prices": [74, 39.24, 39.24],
                     "comp_ratings": [4.5, 4.0, 3.8],
                     "our_rating": 4.0,
                     "current_price": 40,
                     "min_price": 1,
                     "max_price": 1000,
                     "anchor": 38}
    result = app.invoke(held_proceed)
    assert result["reason"].startswith("Held")
    assert "band" not in result
    assert "target_price" not in result

    outcome, price = conn.execute(
        "SELECT outcome, price FROM decisions WHERE observed_at = ?", ("05-09-2018 10:00",)
    ).fetchone()
    assert outcome == "held"
    assert price is None
    assert fake.call_count == 0

    current_price = conn.execute(
        "SELECT current_price FROM monthly_state WHERE product_id = ? AND month = ?", ("g4", "2018-09"),
    ).fetchone()[0]
    assert current_price == 39.0


# --- validate_input_node ---
def test_validate_input_node_valid_state():
    state = {
        "product_id": "g4",
        "comp_prices": [80, 40, 35],
        "comp_ratings": [4.5, 4.0, 3.8],
        "our_rating": 4.0,
        "observed_at": "01-09-2018 10:00",
        "last_observed_mon_yr": "01-08-2018",
        "prev_comp_prices": [74, 39.24, 39.24],
    }
    result = validate_input_node(state)
    assert result == {"observed_date": datetime(2018, 9, 1, 10, 0)}

def test_validate_input_node_bad_our_rating_raises():
    state = {
        "product_id": "g4",
        "comp_prices": [80, 40, 35],
        "comp_ratings": [4.5, 4.0, 3.8],
        "our_rating": 7.5,
        "observed_at": "01-09-2018 10:00",
        "last_observed_mon_yr": "01-08-2018",
        "prev_comp_prices": [74, 39.24, 39.24],
    }
    with pytest.raises(ValueError):
        validate_input_node(state)


# --- pair_competitors_node ---
def test_pair_competitors_node():
    state = {"product_id": "g4", "comp_prices": [80, 40, 35], "comp_ratings": [4.5, 4.0, 3.8]}
    result = pair_competitors_node(state)
    assert result == {
        "comp_details": [
            {"comp_price": 80, "comp_rating": 4.5},
            {"comp_price": 40, "comp_rating": 4.0},
            {"comp_price": 35, "comp_rating": 3.8},
        ]
    }


# --- analyze_position_node ---
def test_analyze_position_node_rated_above_average():
    state = {"comp_prices": [80, 40, 35], "comp_ratings": [4.0, 4.0, 4.0], "our_rating": 4.5}
    result = analyze_position_node(state)
    assert result["band"] == {"low": 35, "high": 80, "mid": 57.5}
    assert result["gap"] == pytest.approx(0.5)
    assert result["target_label"] == "high"
    assert set(result.keys()) == {"band", "gap", "target_label"}

def test_analyze_position_node_rated_at_average():
    state = {"comp_prices": [80, 40, 35], "comp_ratings": [4.0, 4.0, 4.0], "our_rating": 4.0}
    result = analyze_position_node(state)
    assert result["band"] == {"low": 35, "high": 80, "mid": 57.5}
    assert result["gap"] == pytest.approx(0.0)
    assert result["target_label"] == "mid"
    assert set(result.keys()) == {"band", "gap", "target_label"}

def test_analyze_position_node_rated_below_average():
    state = {"comp_prices": [80, 40, 35], "comp_ratings": [4.0, 4.0, 4.0], "our_rating": 3.5}
    result = analyze_position_node(state)
    assert result["band"] == {"low": 35, "high": 80, "mid": 57.5}
    assert result["gap"] == pytest.approx(-0.5)
    assert result["target_label"] == "low"
    assert set(result.keys()) == {"band", "gap", "target_label"}


# --- resolve_target_price_node ---
def test_resolve_target_price_node_no_tie():
    comp_details = [
        {"comp_price": 80, "comp_rating": 4.5},
        {"comp_price": 40, "comp_rating": 4.2},
        {"comp_price": 35, "comp_rating": 3.8},
    ]
    state = {
        "target_label": "mid",
        "gap": -0.16666666666666696,
        "band": {"low": 35, "high": 80, "mid": 57.5},
        "our_rating": 4.0,
        "comp_details": comp_details,
    }
    result = resolve_target_price_node(state)
    assert result == {"target_price": 57.5, "cleared_comp": None, "target_source": "band"}

def test_resolve_target_price_node_tie():
    comp_details = [
        {"comp_price": 80, "comp_rating": 4.5},
        {"comp_price": 40, "comp_rating": 4.0},
        {"comp_price": 35, "comp_rating": 3.8},
    ]
    state = {
        "target_label": "mid",
        "gap": -0.10000000000000053,
        "band": {"low": 35, "high": 80, "mid": 57.5},
        "our_rating": 4.0,
        "comp_details": comp_details,
    }
    result = resolve_target_price_node(state)
    assert result == {
        "target_price": 40,
        "cleared_comp": {"comp_price": 80, "comp_rating": 4.5},
        "target_source": "tie_match",
    }


# --- compute_step_node ---
def test_compute_step_node_direct_ignores_llm_step_pct():
    state = {"path": "direct", "current_price": 39, "target_price": 41, "anchor": 40, "llm_step_pct": 0.02}
    result = compute_step_node(state)
    assert result == {"price": 41}

def test_compute_step_node_llm_partial_step():
    state = {"path": "llm", "current_price": 39, "target_price": 41, "anchor": 40, "llm_step_pct": 0.02}
    result = compute_step_node(state)
    assert result["price"] == pytest.approx(39.8)


# --- enforce_bounds_node ---
def test_enforce_bounds_node_within_bounds():
    state = {"product_id": "g4", "price": 50, "min_price": 10, "max_price": 100}
    result = enforce_bounds_node(state)
    assert result == {"price": 50, "bounds_clamped": False}

def test_enforce_bounds_node_below_floor():
    state = {"product_id": "g4", "price": 5, "min_price": 10, "max_price": 100}
    result = enforce_bounds_node(state)
    assert result == {"price": 10, "bounds_clamped": True}

def test_enforce_bounds_node_above_ceiling():
    state = {"product_id": "g4", "price": 150, "min_price": 10, "max_price": 100}
    result = enforce_bounds_node(state)
    assert result == {"price": 100, "bounds_clamped": True}

def test_enforce_bounds_node_exactly_on_floor_not_clamped():
    state = {"product_id": "g4", "price": 10, "min_price": 10, "max_price": 100}
    result = enforce_bounds_node(state)
    assert result == {"price": 10, "bounds_clamped": False}


# --- no_action_node ---
def test_no_action_node_returns_empty_dict():
    assert no_action_node({"action": "none"}) == {}


# --- persist_node, direct ---
def test_persist_node_rounding_branch_no_change(_mock_staging_db):
    conn = _mock_staging_db
    state = {
        "product_id": "g4",
        "observed_at": "01-09-2018 10:00",
        "observed_date": datetime(2018, 9, 1, 10, 0),
        "comp_prices": [80, 40, 35],
        "comp_ratings": [4.5, 4.0, 3.8],
        "our_rating": 4.0,
        "price": 39.004,
        "current_price": 39,
        "anchor": 39,
        "action": "proceed",
        "reason": "No change: price stays at 39.00.",
    }
    result = persist_node(state)
    assert result == {}

    outcome = conn.execute("SELECT outcome FROM decisions WHERE product_id = ?", ("g4",)).fetchone()[0]
    assert outcome == "no_change"

    current_price = conn.execute(
        "SELECT current_price FROM monthly_state WHERE product_id = ? AND month = ?", ("g4", "2018-09"),
    ).fetchone()[0]
    assert current_price == 39.0

def test_persist_node_no_anchor_or_current_price_skips_monthly_state(_mock_staging_db):
    conn = _mock_staging_db
    state = {
        "product_id": "g4",
        "observed_at": "01-09-2018 10:00",
        "comp_prices": [100, 55, 39.0],
        "comp_ratings": [4.5, 4.0, 3.8],
        "our_rating": 4.0,
        "escalation_cause": "competitor_move",
        "tier": "high",
        "reason": "Escalated (high): 2 competitor(s) moved 30% or more in one observation.",
        "detail_pack": {
            "triggered": [{"index": 0, "comp_price_new": 100, "comp_price_prev": 74, "percent_change": 35.1}],
            "comp_prices": [100, 55, 39.0],
            "prev_comp_prices": [74, 39.24, 39.24],
        },
    }
    result = persist_node(state)
    assert result == {}

    count = conn.execute("SELECT COUNT(*) FROM decisions WHERE product_id = ?", ("g4",)).fetchone()[0]
    assert count == 1

    monthly_rows = conn.execute("SELECT COUNT(*) FROM monthly_state WHERE product_id = ?", ("g4",)).fetchone()[0]
    assert monthly_rows == 0


# --- route_comp_move_check ---
def test_route_comp_move_check_trigger():
    assert route_comp_move_check({"action": "trigger"}) == "trigger"

def test_route_comp_move_check_none():
    assert route_comp_move_check({"action": "none"}) == "none"

def test_route_comp_move_check_proceed():
    assert route_comp_move_check({"action": "proceed"}) == "proceed"

def test_route_comp_move_check_unknown_raises():
    with pytest.raises(ValueError):
        route_comp_move_check({"action": "something_else"})


# --- route_check_move_size ---
def test_route_check_move_size_escalate():
    assert route_check_move_size({"path": "escalate"}) == "escalate"

def test_route_check_move_size_llm():
    assert route_check_move_size({"path": "llm"}) == "llm"

def test_route_check_move_size_direct():
    assert route_check_move_size({"path": "direct"}) == "direct"

def test_route_check_move_size_unknown_raises():
    with pytest.raises(ValueError):
        route_check_move_size({"path": "something_else"})


# --- route_llm_step_size ---
def test_route_llm_step_size_llm_failed_escalates():
    assert route_llm_step_size({"escalation_cause": "llm_failed"}) == "escalate"

def test_route_llm_step_size_no_escalation_cause_computes_step():
    assert route_llm_step_size({}) == "compute_step"

def test_route_llm_step_size_other_escalation_cause_computes_step():
    assert route_llm_step_size({"escalation_cause": "cap_exceeded"}) == "compute_step"
