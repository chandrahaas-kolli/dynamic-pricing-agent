import pytest
from src.agent import app, escalate_node, competitor_move_check_node, check_move_size_node, reason_node


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
    }
    result = escalate_node(state)
    assert result["needs_brief"] is True

def test_escalate_node_cap_exceeded():
    state = {
        "escalation_cause": "cap_exceeded",
        "target_price": 40,
        "target_source": "band",
        "anchor": 30,
        "x_anchor": 33.33,
        "current_price": 39,
        "band": {"low": 35, "mid": 57.5, "high": 80},
        "gap": -0.1,
    }
    result = escalate_node(state)
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
        "bounds_clamped": False,
        "min_price": 1,
        "max_price": 1000,
    }
    state.update(overrides)
    return state

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
                           dominance_clamped=True)
    result = reason_node(state)
    assert "Partial step toward" not in result["reason"]

def test_reason_node_dominance_clamp_only():
    state = _priced_state(price=101.99, current_price=100, target_price=104, target_source="band",
                           dominance_clamped=True, bounds_clamped=False, path="direct")
    result = reason_node(state)
    assert "Clamped below a higher-rated competitor" in result["reason"]

def test_reason_node_ceiling_clamp():
    state = _priced_state(price=55, current_price=56, target_price=57.5, target_source="band",
                           bounds_clamped=True, dominance_clamped=False, max_price=55, min_price=1)
    result = reason_node(state)
    assert "Lowered to the ceiling" in result["reason"]

def test_reason_node_floor_overrides_dominance():
    state = _priced_state(price=1, current_price=105, target_price=96, target_source="tie_match",
                           dominance_clamped=True, bounds_clamped=True, min_price=1, max_price=1000)
    result = reason_node(state)
    assert "Raised to the floor" in result["reason"]
    assert "Floor overrode the dominance clamp" in result["reason"]
    assert "Clamped below" not in result["reason"]

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

def test_app_invoke_reason_llm():
    proceed = {**base, "comp_prices": [80, 40, 35],
               "prev_comp_prices": [74, 39.24, 39.24],
               "comp_ratings": [4.5, 4.2, 3.8],
               "our_rating": 4.3,
               "current_price": 56,
               "min_price": 1,
               "max_price": 55,
               "anchor": 56,
               "llm_step_pct": 0.01}
    result = app.invoke(proceed)
    assert result["reason"]
