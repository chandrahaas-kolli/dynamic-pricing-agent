from datetime import datetime
from typing import TypedDict, List, Dict, Optional
from typing_extensions import NotRequired


class CompEntry(TypedDict):
    """One competitor: price + rating, from pair_competitors."""
    comp_price: float
    comp_rating: float


class PipelineState(TypedDict):
    # --- always present ---
    product_id: str
    comp_prices: List[float]
    comp_ratings: List[float]
    observed_at: str
    last_observed_mon_yr: str
    prev_comp_prices: List[float]
    cleared_comp: Optional[CompEntry]

    # --- optional: present only on some paths ---
    observed_date: NotRequired[datetime]
    comp_details: NotRequired[List[CompEntry]]
    anchor: NotRequired[float]
    our_rating: NotRequired[float]
    tier: NotRequired[str]
    escalate: NotRequired[bool]
    action: NotRequired[str]
    triggered: NotRequired[List[Dict]]
    percent_diff: NotRequired[List[float]]
    band: NotRequired[Dict]
    gap: NotRequired[float]
    target_label: NotRequired[str]
    target_price: NotRequired[float]
    path: NotRequired[str]
    x_anchor: NotRequired[float]
    current_price: NotRequired[float]
    llm_step_pct: NotRequired[float]
    price: NotRequired[float]
    min_price: NotRequired[float]
    max_price: NotRequired[float]