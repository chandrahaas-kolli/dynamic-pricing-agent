from datetime import datetime
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


def validate_input(product_id, comp_prices, comp_ratings, observed_at, last_obs_mon_yr):
    validate_prices(product_id, comp_prices)
    validate_ratings(product_id, comp_ratings)
    validate_observed_at(product_id, observed_at)
    parsed_date = check_continuity(product_id, last_obs_mon_yr, observed_at)

    return parsed_date


def pair_competitors(product_id, comp_prices, comp_ratings):
    return [{"comp_price" : comp_price, "comp_rating" : comp_rating}
            for comp_price, comp_rating in zip(comp_prices, comp_ratings)]