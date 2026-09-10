def validate_prices(product_id, comp_prices):
    """Check competitor prices are well-formed. Raises ValueError if not."""

    if len(comp_prices) != 3:
        raise ValueError(f'Invalid competitor prices for {product_id}: {comp_prices}, must have 3 values')

    for comp in comp_prices:
        if not isinstance(comp, (int, float)):
            raise ValueError(f'Invalid datatype for competitor price for {product_id}: {comp!r}, must be a number')
        elif comp <= 0:
            raise ValueError(f'Invalid entry for competitor price for {product_id}: {comp}, must be greater than 0')

    return comp_prices


def validate_ratings(product_id, comp_ratings):
    """Check competitor ratings are well-formed. Raises ValueError if not."""

    if len(comp_ratings) != 3:
        raise ValueError(f'Invalid number of ratings for {product_id}: {len(comp_ratings)}, must be exactly 3')

    for rating in comp_ratings:
        if not isinstance(rating, (int, float)):
            raise ValueError(f'Invalid rating type for {product_id}: {rating!r}, must be a number')
        elif rating <= 0 or rating > 5.0:
            raise ValueError(f'Invalid rating value for {product_id}: {rating}, must be in range (0, 5.0]')

    return comp_ratings