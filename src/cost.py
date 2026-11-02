"""Estimated USD cost of a Bedrock call, from token counts and list prices."""


def _validate_token_count(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
) -> float:
    """Estimate the USD cost of one call from its token counts and list prices.

    input_price_per_mtok and output_price_per_mtok are USD per 1M tokens.
    The result is an estimate at list price, not billing truth, and is not
    rounded here -- round only when displaying it.
    """
    _validate_token_count("input_tokens", input_tokens)
    _validate_token_count("output_tokens", output_tokens)

    return (
        input_price_per_mtok * input_tokens + output_price_per_mtok * output_tokens
    ) / 1_000_000
