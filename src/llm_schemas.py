"""Pydantic schemas for structured LLM outputs used by the pricing agent."""
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