"""
LangGraph orchestration for the dynamic pricing agent.

pricing.py holds the pure calculation functions; this file wraps them as
graph nodes and wires them together in execution order.

Flow:
    START -> validate_input -> competitor_move_check -> route on action:
        trigger -> trigger_node (placeholder, becomes escalate)
        none    -> no_action_node -> END
        proceed -> pair_competitors -> END (later: analyze_position)
"""

from langgraph.graph import StateGraph, START, END
from src.state import PipelineState
from src.pricing import validate_input, pair_competitors, competitor_move_check
from typing import Dict


# ---------- Nodes ----------
# A node reads the state and returns a dict of only the keys it changes.
# LangGraph merges that dict into the state.

def validate_input_node(state: PipelineState) -> Dict:
    """M1: validate inputs and store the parsed datetime in observed_date.

    observed_at is left untouched.
    """
    parsed = validate_input(
        state["product_id"], state["comp_prices"], state["comp_ratings"],
        state["observed_at"], state["last_observed_mon_yr"]
    )
    return {"observed_date": parsed}


def competitor_move_check_node(state: PipelineState) -> Dict:
    """
    M2: compare new vs previous competitor prices.

    Returns the pricing function's dict directly, since its keys vary by
    outcome: 'triggered' only on trigger, 'percent_diff' only on proceed.
    """
    return competitor_move_check(
        state["product_id"], state["comp_prices"], state["prev_comp_prices"]
    )


def pair_competitors_node(state: PipelineState) -> Dict:
    """Pair each competitor price with its rating. Runs only on the proceed path."""
    comp_details = pair_competitors(
        state["product_id"], state["comp_prices"], state["comp_ratings"]
    )
    return {"comp_details": comp_details}


def trigger_node(state: PipelineState) -> Dict:
    """Placeholder for the escalate node (30%+ move). Changes nothing yet."""
    return {}


def no_action_node(state: PipelineState) -> Dict:
    """Move below deadband: nothing to do, run ends here."""
    return {}


# ---------- Routing ----------

def route_comp_move_check(state: PipelineState) -> str:
    """Return the branch label for the action set by the move check."""
    for route in ("trigger", "none", "proceed"):
        if state["action"] == route:
            return route
    raise ValueError(f"unexpected action: {state['action']}")


# ---------- Graph wiring ----------

graph = StateGraph(PipelineState)

graph.add_node("validate_input_node", validate_input_node)
graph.add_node("competitor_move_check_node", competitor_move_check_node)
graph.add_node("pair_competitors_node", pair_competitors_node)
graph.add_node("trigger_node", trigger_node)
graph.add_node("no_action_node", no_action_node)

graph.add_edge(START, "validate_input_node")
graph.add_edge("validate_input_node", "competitor_move_check_node")

# Move check runs before pairing: cheapest place to stop before computing anything.
graph.add_conditional_edges(
    "competitor_move_check_node",
    route_comp_move_check,
    {"trigger": "trigger_node", "none": "no_action_node", "proceed": "pair_competitors_node"},
)

graph.add_edge("trigger_node", END)
graph.add_edge("no_action_node", END)
graph.add_edge("pair_competitors_node", END)  # later: -> analyze_position_node

app = graph.compile()