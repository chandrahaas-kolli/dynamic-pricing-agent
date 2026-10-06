from datetime import datetime
from math import ceil, log

from dateutil.relativedelta import relativedelta


def validate_prices(product_id, comp_prices):
    """Check competitor prices are well-formed. Raises ValueError if not."""

    if len(comp_prices) != 3:
        raise ValueError(f'Invalid competitor prices for {product_id}: {comp_prices}, must have 3 values')

    for comp in comp_prices:
        if not isinstance(comp, (int, float)):
            raise TypeError(f'Invalid datatype for competitor price for {product_id}: {comp!r}, must be a number')
        elif comp <= 0:
            raise ValueError(f'Invalid entry for competitor price for {product_id}: {comp}, must be greater than 0')

    return comp_prices


def validate_ratings(product_id, comp_ratings):
    """Check competitor ratings are well-formed. Raises ValueError if not."""

    if len(comp_ratings) != 3:
        raise ValueError(f'Invalid number of ratings for {product_id}: {len(comp_ratings)}, must be exactly 3')

    for rating in comp_ratings:
        if not isinstance(rating, (int, float)):
            raise TypeError(f'Invalid rating type for {product_id}: {rating!r}, must be a number')
        elif rating <= 0 or rating > 5.0:
            raise ValueError(f'Invalid rating value for {product_id}: {rating}, must be in range (0, 5.0]')

    return comp_ratings


def validate_observed_at(product_id, observed_at):
    """Check observed_at is a parseable date in DD-MM-YYYY format. Raises ValueError if not."""
    try:
        return datetime.strptime(observed_at, "%d-%m-%Y")
    except ValueError:
        raise ValueError(
            f'Invalid observed_at for {product_id}: {observed_at!r}, expected format DD-MM-YYYY'
        )


def check_continuity(product_id, last_observed_mon_yr, observed_at):
    """Check observed_at is the same month as the last observation or the next one.

    Raises ValueError if not. Assumes observed_at is already validated.
    """
    prev = datetime.strptime(last_observed_mon_yr, "%d-%m-%Y")
    new = datetime.strptime(observed_at, "%d-%m-%Y")
    next_month = prev + relativedelta(months=1)

    same_month = (new.year == prev.year and new.month == prev.month)
    is_next_month = (new.year == next_month.year and new.month == next_month.month)

    if not (same_month or is_next_month):
        raise ValueError(
            f'Invalid observed_at for {product_id}: {observed_at}, '
            f'must be the same month as {last_observed_mon_yr} or the next one'
        )
        
    return new


def validate_input(product_id, comp_prices, comp_ratings, observed_at, last_observed_mon_yr, prev_comp_prices):
    validate_prices(product_id, comp_prices)
    validate_prices(product_id, prev_comp_prices)
    validate_ratings(product_id, comp_ratings)
    validate_observed_at(product_id, observed_at)
    validate_observed_at(product_id, last_observed_mon_yr)
    parsed_date = check_continuity(product_id, last_observed_mon_yr, observed_at)

    return parsed_date


def pair_competitors(product_id, comp_prices, comp_ratings):
    return [{"comp_price" : comp_price, "comp_rating" : comp_rating}
            for comp_price, comp_rating in zip(comp_prices, comp_ratings)]


def competitor_move_check(product_id, new_comp_prices, prev_comp_prices):
    percent_diff = [(new - prev) * 100 / prev for new, prev in zip(new_comp_prices, prev_comp_prices)]

    high_hits = [i for i, val in enumerate(percent_diff) if abs(val) >= 30]
    if high_hits:
        return {
            "tier": "high",
            "escalate": True,
            "action": "trigger",
            "triggered": [
                {
                    "index": i,
                    "comp_price_new": new_comp_prices[i],
                    "comp_price_prev": prev_comp_prices[i],
                    "percent_change": percent_diff[i],
                }
                for i in high_hits
            ],
            "percent_diff": percent_diff,
        }

    moved = [i for i, val in enumerate(percent_diff) if abs(val) >= 2]
    if not moved:
        return {"tier": "low", "escalate": False, "action": "none"}

    return {"tier": "low", "escalate": False, "action": "proceed", "percent_diff": percent_diff}


def build_band(comp_prices):
    """Compute the competitor price band.

    low  = cheapest competitor price
    high = most expensive competitor price
    mid  = midpoint of the range (not the average of all three prices)
    """
    low = min(comp_prices)
    high = max(comp_prices)
    mid = (low + high) / 2
    band = {
        "low": low,
        "high": high,
        "mid": mid,
    }
    return band    


def rating_gap(our_rating, comp_ratings):
    """Compute own rating minus the average of the 3 competitor ratings. Unrounded."""
    avg_rating = sum(comp_ratings) / len(comp_ratings)
    return our_rating - avg_rating


def choose_target(gap):
    """Decide which edge of the price band to target, based on the rating gap.

    round(gap, 6) kills floating-point noise (e.g. 0.20000000000000018)
    without changing the real 0.2 threshold.
    """
    if round(gap, 6) >= 0.2:
        return "high"
    elif round(gap, 6) <= -0.2:
        return "low"
    else:
        return "mid"


def resolve_target_price(target_label, gap, band, our_rating, comp_details):
    """Decide the target price and which competitor (if any) is already
    accounted for; cleared_comp is returned for logging and traceability.

    Also returns target_source, a string naming which branch set the
    target: "tie_match", "tie_undercut", "top_premium", or "band".
    """
    epsilon = 0.01
    tied_with_us = [c for c in comp_details if c["comp_rating"] == our_rating]
    rated_above_us = [c for c in comp_details if c["comp_rating"] > our_rating]

    if tied_with_us and rated_above_us:
        highest_tied = max(tied_with_us, key=lambda c: c["comp_price"])
        cheapest_above_us = min(rated_above_us, key=lambda c: c["comp_price"])
        if highest_tied["comp_price"] < cheapest_above_us["comp_price"]:
            return highest_tied["comp_price"], cheapest_above_us, "tie_match"
        else:
            return cheapest_above_us["comp_price"] - epsilon, cheapest_above_us, "tie_undercut"

    is_top_rated = not rated_above_us
    if target_label == "high" and is_top_rated:
        premium_pct = min(gap / 0.2, 2) * 0.05
        return band["high"] * (1 + premium_pct), None, "top_premium"

    return band[target_label], None, "band"


def check_move_size(target_price, anchor):
    """Decide the path BEFORE agent.py calls the LLM or not.

    escalate: target unreachable this month.
    llm: small move, LLM decides how much of it to take now.
    direct: 5-10% move, no LLM needed, go straight to target.
    """
    x_anchor = (target_price - anchor) / anchor * 100
    if abs(x_anchor) > 10:
        return "escalate", x_anchor
    elif abs(x_anchor) <= 5:
        return "llm", x_anchor
    else:
        return "direct", x_anchor


def compute_step(current_price, target_price, anchor, llm_step_pct=None):
    """Move toward target_price. llm_step_pct=None means the direct path —
    agent.py already confirmed 5-10% is safe, so just land on target.
    """
    if llm_step_pct is None:
        return target_price
    step_dollar = llm_step_pct * anchor
    direction = 1 if target_price > current_price else -1
    distance = abs(target_price - current_price)
    return current_price + direction * min(step_dollar, distance)


def apply_dominance_clamp(new_price, our_rating, comp_details, direction):
    """Keep the price on the correct side of same-direction-relevant competitors,
    unless the absolute floor/ceiling overrides it (see enforce_bounds).

    Moving up (direction > 0): never land at/above a competitor rated higher
    than us -- clamp to comp_price - epsilon. Checks every higher-rated
    competitor, no skip.

    Moving down (direction < 0): never land at/below a competitor rated lower
    than us -- clamp to comp_price + epsilon. Checks every lower-rated
    competitor, no skip. Then, if the price is still at/above any competitor
    rated higher than us, clamp to (the lowest such higher-rated price) -
    epsilon; the higher-rated rule always wins this conflict. The price may
    land on either side of current_price.

    No move (direction == 0): neither rule applies, price is returned
    unchanged.

    direction is the sign of this step's price movement (current_price to
    the pre-clamp new_price), supplied by the caller.

    Returns (price, side): side is "below_higher_rated", "above_lower_rated",
    or None if the price did not change. The conflict case reports
    "below_higher_rated".
    """
    epsilon = 0.01
    side = None
    if direction > 0:
        for comp in comp_details:
            if comp["comp_rating"] > our_rating and new_price >= comp["comp_price"]:
                new_price = comp["comp_price"] - epsilon
                side = "below_higher_rated"
    elif direction < 0:
        for comp in comp_details:
            if comp["comp_rating"] < our_rating and new_price <= comp["comp_price"]:
                new_price = comp["comp_price"] + epsilon
                side = "above_lower_rated"
        for comp in comp_details:
            if comp["comp_rating"] > our_rating and new_price >= comp["comp_price"]:
                new_price = comp["comp_price"] - epsilon
                side = "below_higher_rated"
    return new_price, side


def enforce_bounds(product_id, price, min_price, max_price):
    """Clamp price to the frozen absolute floor/ceiling. Never escalates — log-only.

    min_price = 1.0 x historical minimum unit_price for this product.
    max_price = 1.2 x historical maximum unit_price for this product.
    Both computed once from the original dataset and frozen, not recalculated.
    """
    if price < min_price:
        return min_price
    elif price > max_price:
        return max_price
    else:
        return price


def months_to_reach(anchor, target, cap=0.10):
    """Minimum whole months to move from anchor to target, moving at most
    `cap` per month, where the anchor resets each month to that month's new
    price.

    Because the anchor resets monthly, the move compounds: each month's cap
    applies to the previous month's price, not the original one. A linear
    estimate (distance / cap) overcounts a rising target and undercounts a
    falling one for exactly that reason (e.g. +46%: 5 vs 4; -28%: 3 vs 4).
    The number of compounding steps needed is log(target / anchor) base
    (1 + cap) (or base (1 - cap) for a falling target).

    The raw ratio is rounded to 9 decimal places before ceil(): floating-
    point noise can land a result that's mathematically exact (e.g. 2) just
    above the integer (e.g. 2.0000000000000004), which would otherwise push
    the answer up by a spurious month.

    Args:
        anchor: current anchor price, must be greater than 0.
        target: price to reach, must be greater than 0.
        cap: maximum fractional move per month (e.g. 0.10 for 10%).

    Returns:
        Minimum whole number of months (int); 0 if target equals anchor.

    Raises:
        ValueError: if anchor or target is not greater than 0.
    """
    if anchor <= 0:
        raise ValueError(f'Invalid anchor: {anchor}, must be greater than 0')
    if target <= 0:
        raise ValueError(f'Invalid target: {target}, must be greater than 0')
    if target == anchor:
        return 0

    ratio = target / anchor
    base = 1 + cap if ratio > 1 else 1 - cap
    return ceil(round(log(ratio) / log(base), 9))