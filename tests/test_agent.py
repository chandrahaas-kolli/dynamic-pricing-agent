import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from src.agent import (
    app, escalate_node, competitor_move_check_node, check_move_size_node,
    reason_node, apply_dominance_clamp_node, llm_step_size_node,
    build_competitors_text, describe_target, build_market_move_text,
    _cap_brief, _market_brief,
)
from src.llm_schemas import StepDecision, CapExceededBrief, MarketMoveBrief
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
    "observed_at": "01-09-2018",
    "last_observed_mon_yr": "01-08-2018",
    "cleared_comp": None,
}

def test_app_invoke_double_trigger():
    result = app.invoke({**base, "comp_prices": [100, 55, 39.0],
                          "prev_comp_prices": [74, 39.24, 39.24]})
    assert result["escalation_cause"] == "competitor_move"
    assert result["needs_brief"] is True

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
