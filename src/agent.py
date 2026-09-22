"""
LangGraph orchestration for the dynamic pricing agent.

pricing.py holds the pure calculation functions; this file wraps them as
graph nodes and wires them together in execution order.

Flow:
    START -> validate_input -> competitor_move_check -> route on action:
        trigger -> trigger_node (placeholder, becomes escalate)
        none    -> no_action_node -> END
        proceed -> pair_competitors -> analyze_position -> resolve_target_price
                   -> check_move_size -> route on path:
                       escalate -> escalate_node (placeholder) -> END
                       llm      -> llm_step_size (placeholder) -> compute_step -> END
                       direct   -> compute_step -> END
"""

from typing import Dict

from langgraph.graph import StateGraph, START, END

from src.state import PipelineState
from src.pricing import (
    validate_input,
    pair_competitors,
    competitor_move_check,
    build_band,
    rating_gap,
    choose_target,
    resolve_target_price,
    check_move_size,
    compute_step,
)


# ---------- Nodes ----------
# A node reads the state and returns a dict of only the keys it changes.
# LangGraph merges that dict into the state.

def validate_input_node(state: PipelineState) -> Dict:
    """Validate inputs and store the parsed datetime in observed_date.

    observed_at is left untouched.
    """
    parsed = validate_input(
        state["product_id"], state["comp_prices"], state["comp_ratings"],
        state["observed_at"], state["last_observed_mon_yr"]
    )
    return {"observed_date": parsed}


def competitor_move_check_node(state: PipelineState) -> Dict:
    """Compare new vs previous competitor prices.

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


def analyze_position_node(state: PipelineState) -> Dict:
    """Build the competitor price band, compute the rating gap, and choose the target.

    Kept as one node: the three steps always run together with no branching.
    Runs only on the proceed path. Requires our_rating in the state.
    """
    band = build_band(state["comp_prices"])
    ratings_gap = rating_gap(state["our_rating"], state["comp_ratings"])
    target_label = choose_target(ratings_gap)
    return {"band": band, "gap": ratings_gap, "target_label": target_label}


def resolve_target_price_node(state: PipelineState) -> Dict:
    """Set the target price and the competitor already handled, if any.

    cleared_comp is set only when we are rating-tied with a competitor and
    another is rated above us; kept for logging.
    Runs only on the proceed path.
    """
    target_price, cleared_comp = resolve_target_price(
        state["target_label"], state["gap"], state["band"],
        state["our_rating"], state["comp_details"]
    )
    return {"target_price": target_price, "cleared_comp": cleared_comp}


def check_move_size_node(state: PipelineState) -> Dict:
    """Measure the target's distance from the anchor and choose the path.

    path is 'escalate' (beyond the monthly cap), 'llm' (small move) or
    'direct' (medium move). On escalate, also sets escalate=True and
    tier='medium'. Requires anchor in the state.
    """
    path, x_anchor = check_move_size(state["target_price"], state["anchor"])
    if path == "escalate":
        return {
            "path": path,
            "x_anchor": x_anchor,
            "escalate": True,
            "tier": "medium",
        }
    return {"path": path, "x_anchor": x_anchor}


def llm_step_size_node(state: PipelineState) -> Dict:
    """Placeholder for the LLM step-size choice (llm path). Changes nothing yet."""
    return {}


def compute_step_node(state: PipelineState) -> Dict:
    """Move the price toward the target and store it in price.

    llm_step_pct is set only on the llm path; on the direct path it is
    missing, so .get() returns None and the price lands on the target.
    Requires current_price in the state.
    """
    price = compute_step(
        state["current_price"], state["target_price"],
        state["anchor"], state.get("llm_step_pct")
    )
    return {"price": price}


def trigger_node(state: PipelineState) -> Dict:
    """Placeholder for the escalate node (30%+ move). Changes nothing yet."""
    return {}


def escalate_node(state: PipelineState) -> Dict:
    """Placeholder for escalation when the target exceeds the monthly cap.

    To be merged with trigger_node once the escalate logic is built.
    """
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


def route_check_move_size(state: PipelineState) -> str:
    """Return the branch label for the path set by the move-size check."""
    for route in ("escalate", "llm", "direct"):
        if state["path"] == route:
            return route
    raise ValueError(f"unexpected path: {state['path']}")


# ---------- Graph wiring ----------

graph = StateGraph(PipelineState)

graph.add_node("validate_input_node", validate_input_node)
graph.add_node("competitor_move_check_node", competitor_move_check_node)
graph.add_node("pair_competitors_node", pair_competitors_node)
graph.add_node("analyze_position_node", analyze_position_node)
graph.add_node("resolve_target_price_node", resolve_target_price_node)
graph.add_node("check_move_size_node", check_move_size_node)
graph.add_node("llm_step_size_node", llm_step_size_node)
graph.add_node("compute_step_node", compute_step_node)
graph.add_node("trigger_node", trigger_node)
graph.add_node("escalate_node", escalate_node)
graph.add_node("no_action_node", no_action_node)

graph.add_edge(START, "validate_input_node")
graph.add_edge("validate_input_node", "competitor_move_check_node")

# Move check runs before pairing: cheapest place to stop before computing anything.
graph.add_conditional_edges(
    "competitor_move_check_node",
    route_comp_move_check,
    {"trigger": "trigger_node", "none": "no_action_node", "proceed": "pair_competitors_node"},
)

# Proceed path
graph.add_edge("pair_competitors_node", "analyze_position_node")
graph.add_edge("analyze_position_node", "resolve_target_price_node")
graph.add_edge("resolve_target_price_node", "check_move_size_node")

graph.add_conditional_edges(
    "check_move_size_node",
    route_check_move_size,
    {"escalate": "escalate_node", "llm": "llm_step_size_node", "direct": "compute_step_node"},
)

graph.add_edge("llm_step_size_node", "compute_step_node")
graph.add_edge("compute_step_node", END)

# Terminal branches
graph.add_edge("trigger_node", END)
graph.add_edge("escalate_node", END)
graph.add_edge("no_action_node", END)

app = graph.compile()