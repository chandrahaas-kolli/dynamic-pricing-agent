"""
LangGraph orchestration for the dynamic pricing agent.

pricing.py holds the pure calculation functions; this file wraps them as
graph nodes and wires them together in execution order.

Flow:
    START -> validate_input -> competitor_move_check -> route on action:
        trigger -> escalate_node (escalation_cause=competitor_move) -> reason -> END
        none    -> no_action_node -> reason -> END
        proceed -> pair_competitors -> analyze_position -> resolve_target_price
                   -> check_move_size -> route on path
                      (llm is overridden to direct for tie targets):
                       escalate -> escalate_node (escalation_cause=cap_exceeded) -> reason -> END
                       llm      -> llm_step_size (placeholder) -> compute_step
                                   -> apply_dominance_clamp -> enforce_bounds -> reason -> END
                       direct   -> compute_step -> apply_dominance_clamp -> enforce_bounds -> reason -> END

A persist node will later sit between reason and END on every path.
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
    apply_dominance_clamp,
    enforce_bounds,
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
        state["observed_at"], state["last_observed_mon_yr"], state["prev_comp_prices"]
    )
    return {"observed_date": parsed}


def competitor_move_check_node(state: PipelineState) -> Dict:
    """Compare new vs previous competitor prices.

    Returns the pricing function's dict directly, since its keys vary by
    outcome: 'triggered' only on trigger, 'percent_diff' only on proceed.
    On trigger, also sets escalation_cause='competitor_move'.
    """
    result = competitor_move_check(
        state["product_id"], state["comp_prices"], state["prev_comp_prices"]
    )
    if result["action"] == "trigger":
        result["escalation_cause"] = "competitor_move"
    return result


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
    target_source records which branch in resolve_target_price set the target.
    Runs only on the proceed path.
    """
    target_price, cleared_comp, target_source = resolve_target_price(
        state["target_label"], state["gap"], state["band"],
        state["our_rating"], state["comp_details"]
    )
    return {
        "target_price": target_price,
        "cleared_comp": cleared_comp,
        "target_source": target_source,
    }


def check_move_size_node(state: PipelineState) -> Dict:
    """Measure the target's distance from the anchor and choose the path.

    path is 'escalate' (beyond the monthly cap), 'llm' (small move) or
    'direct' (medium move). On escalate, also sets escalate=True,
    tier='medium', and escalation_cause='cap_exceeded'. Requires anchor in
    the state.

    If path is 'llm' and target_source is 'tie_match' or 'tie_undercut',
    path is overridden to 'direct' so the price lands exactly on the tie
    target. Escalate is unchanged and takes priority over this override.
    """
    path, x_anchor = check_move_size(state["target_price"], state["anchor"])
    if path == "escalate":
        return {
            "path": path,
            "x_anchor": x_anchor,
            "escalate": True,
            "tier": "medium",
            "escalation_cause": "cap_exceeded",
        }
    if path == "llm" and state["target_source"] in ("tie_match", "tie_undercut"):
        path = "direct"
    return {"path": path, "x_anchor": x_anchor}


def llm_step_size_node(state: PipelineState) -> Dict:
    """Placeholder for the LLM step-size choice (llm path). Changes nothing yet."""
    return {}


def compute_step_node(state: PipelineState) -> Dict:
    """Move the price toward the target and store it in price.

    llm_step_pct is set only on the llm path; on the direct path it is
    missing, so .get() returns None and the price lands on the target.
    On the direct path llm_step_pct is ignored even if present.
    Requires current_price in the state.
    """
    step_pct = state.get("llm_step_pct") if state["path"] == "llm" else None
    price = compute_step(
        state["current_price"], state["target_price"],
        state["anchor"], step_pct
    )
    return {"price": price}


def apply_dominance_clamp_node(state: PipelineState) -> Dict:
    """Lower the price below any higher-rated competitor it would land at or above.

    Checks every higher-rated competitor. dominance_clamped records whether
    the price changed, so the reason survives into logging.
    """
    clamped_price = apply_dominance_clamp(
        state["price"], state["our_rating"], state["comp_details"]
    )
    clamp_flag = clamped_price != state["price"]
    return {
        "price": clamped_price,
        "dominance_clamped": clamp_flag,
    }


def enforce_bounds_node(state: PipelineState) -> Dict:
    """Clamp price to the frozen floor/ceiling; bounds win over the dominance
    clamp if they conflict. bounds_clamped records whether it fired.

    Requires min_price and max_price in the state.
    """
    bounded_price = enforce_bounds(
        state["product_id"], state["price"], state["min_price"], state["max_price"]
    )
    return {
        "price": bounded_price,
        "bounds_clamped": bounded_price != state["price"],
    }


def escalate_node(state: PipelineState) -> Dict:
    """Merged escalation for competitor_move and cap_exceeded.

    Decides the detail_pack and whether a brief is needed by
    escalation_cause, and for competitor_move, further by len(triggered)
    (2+ simultaneous triggers -> needs_brief). The llm_failed cause is added
    with the LLM step-size node; brief is filled in once escalation briefs
    are built.
    """
    cause = state.get("escalation_cause")
    if cause == "competitor_move":
        triggered = state["triggered"]
        detail_pack = {
            "triggered": triggered,
            "comp_prices": state["comp_prices"],
            "prev_comp_prices": state["prev_comp_prices"],
        }
        needs_brief = len(triggered) >= 2
    elif cause == "cap_exceeded":
        detail_pack = {
            "target_price": state["target_price"],
            "target_source": state["target_source"],
            "anchor": state["anchor"],
            "x_anchor": state["x_anchor"],
            "current_price": state["current_price"],
            "band": state["band"],
            "gap": state["gap"],
        }
        needs_brief = True
    else:
        raise ValueError(f"unexpected escalation_cause: {cause}")
    return {"detail_pack": detail_pack, "needs_brief": needs_brief, "brief": None}


def no_action_node(state: PipelineState) -> Dict:
    """Move below deadband: nothing to price; goes to reason."""
    return {}


def reason_node(state: PipelineState) -> Dict:
    """Compose a deterministic, LLM-free explanation of the outcome.

    Runs on every path (escalated, no-change, and priced) as the graph's
    last node. Checked in order: escalation_cause (competitor_move or
    cap_exceeded escalation text, or ValueError on any other cause that is
    present), then action == 'none' (below the deadband), then the priced
    case (direct/llm paths) — which explains target_source and appends
    clauses for any partial step or clamp that fired. Invalid input never
    reaches here, since validate_input raises before any node runs.
    """
    cause = state.get("escalation_cause")
    if cause == "competitor_move":
        triggered = state["triggered"]
        text = f"Escalated (high): {len(triggered)} competitor(s) moved 30% or more in one observation."
        return {"reason": text}
    elif cause == "cap_exceeded":
        target_price = state["target_price"]
        x_anchor = state["x_anchor"]
        anchor = state["anchor"]
        text = f"Escalated (medium): target {target_price:.2f} is {x_anchor:+.1f}% from anchor {anchor:.2f}, beyond the ±10% monthly cap."
        return {"reason": text}
    elif cause is not None:
        raise ValueError(f"unexpected escalation_cause: {cause}")
    elif state["action"] == "none":
        return {"reason": "No change: competitor moves below the 2% deadband."}

    price = state["price"]
    current_price = state["current_price"]
    target_price = state["target_price"]
    target_label = state["target_label"]
    target_source = state["target_source"]
    cleared_comp = state["cleared_comp"]
    path = state["path"]
    dominance_clamped = state["dominance_clamped"]
    bounds_clamped = state["bounds_clamped"]
    min_price = state["min_price"]
    max_price = state["max_price"]

    if round(price, 2) == round(current_price, 2):
        text = f"No change: price stays at {price:.2f}."
    elif target_source == "band":
        text = f"Priced at {price:.2f}: target is the {target_label} of the competitor band ({target_price:.2f})."
    elif target_source == "tie_match":
        text = f"Priced at {price:.2f}: target matches an equally rated competitor at {target_price:.2f}."
    elif target_source == "tie_undercut":
        text = f"Priced at {price:.2f}: target undercuts the cheapest higher-rated competitor at {cleared_comp['comp_price']:.2f}."
    elif target_source == "top_premium":
        text = f"Priced at {price:.2f}: target is a top-rated premium over the band high ({target_price:.2f})."
    else:
        raise ValueError(f"unexpected target_source: {target_source}")

    clauses = []
    if path == "llm" and round(price, 2) != round(target_price, 2) and not dominance_clamped and not bounds_clamped:
        clauses.append(f"Partial step toward {target_price:.2f}.")
    if dominance_clamped and not (bounds_clamped and price == min_price):
        clauses.append("Clamped below a higher-rated competitor.")
    if bounds_clamped and price == min_price:
        clauses.append(f"Raised to the floor {min_price:.2f}.")
    if bounds_clamped and price == max_price:
        clauses.append(f"Lowered to the ceiling {max_price:.2f}.")
    if dominance_clamped and bounds_clamped and price == min_price:
        clauses.append("Floor overrode the dominance clamp.")

    if clauses:
        text = text + " " + " ".join(clauses)
    return {"reason": text}


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
graph.add_node("apply_dominance_clamp_node", apply_dominance_clamp_node)
graph.add_node("enforce_bounds_node", enforce_bounds_node)
graph.add_node("escalate_node", escalate_node)
graph.add_node("no_action_node", no_action_node)
graph.add_node("reason_node", reason_node)

graph.add_edge(START, "validate_input_node")
graph.add_edge("validate_input_node", "competitor_move_check_node")

# Move check runs before pairing: cheapest place to stop before computing anything.
graph.add_conditional_edges(
    "competitor_move_check_node",
    route_comp_move_check,
    {"trigger": "escalate_node", "none": "no_action_node", "proceed": "pair_competitors_node"},
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
graph.add_edge("compute_step_node", "apply_dominance_clamp_node")
graph.add_edge("apply_dominance_clamp_node", "enforce_bounds_node")
graph.add_edge("enforce_bounds_node", "reason_node")

# Branches into reason
graph.add_edge("escalate_node", "reason_node")
graph.add_edge("no_action_node", "reason_node")

graph.add_edge("reason_node", END)

app = graph.compile()