from datetime import datetime
from typing import NotRequired, TypedDict


class CompEntry(TypedDict):
    """One competitor: price + rating, from pair_competitors."""

    comp_price: float
    comp_rating: float


class PipelineState(TypedDict):
    # --- always present ---
    product_id: str
    comp_prices: list[float]
    comp_ratings: list[float]
    our_rating: float
    observed_at: str
    last_observed_mon_yr: str
    prev_comp_prices: list[float]
    cleared_comp: CompEntry | None

    # --- optional: present only on some paths ---
    observed_date: NotRequired[datetime]
    # canonical ISO 8601 UTC key set by validate_input_node; observed_at stays the raw caller string
    observed_key: NotRequired[str]
    comp_details: NotRequired[list[CompEntry]]
    anchor: NotRequired[float]
    tier: NotRequired[str]
    escalate: NotRequired[bool]
    action: NotRequired[str]
    triggered: NotRequired[list[dict]]
    percent_diff: NotRequired[list[float]]
    band: NotRequired[dict]
    gap: NotRequired[float]
    target_label: NotRequired[str]
    target_price: NotRequired[float]
    target_source: NotRequired[str]
    reason: NotRequired[str]
    path: NotRequired[str]
    x_anchor: NotRequired[float]
    current_price: NotRequired[float]
    llm_step_pct: NotRequired[float]
    llm_attempts: NotRequired[int]
    llm_errors: NotRequired[list[str]]
    price: NotRequired[float]
    dominance_clamped: NotRequired[bool]
    dominance_side: NotRequired[str | None]
    bounds_clamped: NotRequired[bool]
    llm_rationale: NotRequired[str]
    min_price: NotRequired[float]
    max_price: NotRequired[float]
    escalation_cause: NotRequired[str]
    detail_pack: NotRequired[dict]
    needs_brief: NotRequired[bool]
    # model_dump() of the escalation brief (CapExceededBrief/MarketMoveBrief), or None
    brief: NotRequired[dict | None]
    # errors and attempt count from the escalation-brief LLM call; kept out of detail_pack
    brief_errors: NotRequired[list[str]]
    brief_attempts: NotRequired[int]
    gate: NotRequired[str]
    stored_decision: NotRequired[dict]
    conflict: NotRequired[dict]
