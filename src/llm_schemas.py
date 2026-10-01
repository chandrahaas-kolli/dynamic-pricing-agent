"""Pydantic schemas for structured LLM outputs used by the pricing agent."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StepDecision(BaseModel):
    """The price step to take toward the target, as a fraction of the anchor price, and why."""

    model_config = ConfigDict(str_strip_whitespace=True)

    step_pct: float = Field(
        gt=0,
        lt=0.05,
        description="Step as a fraction of the anchor price, from 0 to 0.05 exclusive; e.g. 0.03 for 3%",
    )
    rationale: str = Field(
        min_length=1,
        max_length=600,
        description="Reason for choosing this step size, considering each competitor's price and rating. 2-4 plain sentences 80 words max, no Markdown",
    )


class CapExceededBrief(BaseModel):
    """Recommendation for a price target beyond the monthly cap; a human makes the final decision."""

    model_config = ConfigDict(str_strip_whitespace=True)

    recommendation: Literal["one_time_jump", "multi_month_path"] = Field(
        description="one_time_jump = move to the target in one human-approved step; multi_month_path = converge over several months within the ±10% monthly cap.",
    )
    rationale: str = Field(
        min_length=1,
        max_length=600,
        description="Reason for the recommendation, considering how far the target is beyond the monthly cap and the months needed. 2-4 plain sentences, 80 words max, no Markdown.",
    )


class MarketMoveBrief(BaseModel):
    """Assessment of a 30%+ move by two or more competitors at once; a human makes the final decision."""

    model_config = ConfigDict(str_strip_whitespace=True)

    likely_cause: Literal["market_shift", "data_error", "unclear"] = Field(
        description="market_shift = competitors genuinely repriced; data_error = the price feed is likely wrong; unclear = the evidence supports neither.",
    )
    rationale: str = Field(
        min_length=1,
        max_length=600,
        description="Reason for the likely_cause. Must address whether this could be a data error. 2-4 plain sentences, 80 words max, no Markdown.",
    )