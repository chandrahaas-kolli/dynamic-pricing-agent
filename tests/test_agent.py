import pytest
from src.agent import app, escalate_node, competitor_move_check_node, check_move_size_node


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
