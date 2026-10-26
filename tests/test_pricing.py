from datetime import UTC, date, datetime

import pytest

from src.pricing import (
    apply_dominance_clamp,
    build_band,
    canonical_timestamp,
    check_continuity,
    check_move_size,
    choose_target,
    competitor_move_check,
    compute_step,
    enforce_bounds,
    months_to_reach,
    observation_fingerprint,
    pair_competitors,
    rating_gap,
    resolve_target_price,
    validate_input,
    validate_last_observed_mon_yr,
    validate_observed_at,
    validate_our_rating,
    validate_prices,
    validate_ratings,
)


# --- validate_prices ---
def test_validate_prices_valid():
    assert validate_prices("g4", [105, 120, 90]) == [105, 120, 90]


def test_validate_prices_wrong_count():
    with pytest.raises(ValueError):
        validate_prices("g4", [105, 120])


def test_validate_prices_negative():
    with pytest.raises(ValueError):
        validate_prices("g4", [105, -5, 90])


def test_validate_prices_wrong_type():
    with pytest.raises(TypeError):
        validate_prices("g4", [105, "90", 90])


def test_validate_prices_bool_raises_type_error():
    with pytest.raises(TypeError):
        validate_prices("g4", [True, 100, 95])


# --- validate_ratings ---
def test_validate_ratings_valid():
    assert validate_ratings("g4", [4.2, 4.0, 4.1]) == [4.2, 4.0, 4.1]


def test_validate_ratings_wrong_count():
    with pytest.raises(ValueError):
        validate_ratings("g4", [4.2, 4.0])


def test_validate_ratings_out_of_range():
    with pytest.raises(ValueError):
        validate_ratings("g4", [4.2, 0, 4.1])


def test_validate_ratings_wrong_type():
    with pytest.raises(TypeError):
        validate_ratings("g4", [4.2, "one", 4.1])


def test_validate_ratings_bool_raises_type_error():
    with pytest.raises(TypeError):
        validate_ratings("g4", [True, 4.0, 4.1])


def test_validate_ratings_above_max():
    with pytest.raises(ValueError):
        validate_ratings("g4", [4.2, 5.1, 4.1])


def test_validate_ratings_exactly_five_valid():
    assert validate_ratings("g4", [4.2, 5.0, 4.1]) == [4.2, 5.0, 4.1]


# --- validate_our_rating ---
def test_validate_our_rating_valid():
    assert validate_our_rating("g4", 4.0) == 4.0


def test_validate_our_rating_exactly_five_valid():
    assert validate_our_rating("g4", 5.0) == 5.0


def test_validate_our_rating_zero_raises():
    with pytest.raises(ValueError):
        validate_our_rating("g4", 0)


def test_validate_our_rating_above_max_raises():
    with pytest.raises(ValueError):
        validate_our_rating("g4", 5.1)


def test_validate_our_rating_none_raises_type_error():
    with pytest.raises(TypeError):
        validate_our_rating("g4", None)


def test_validate_our_rating_string_raises_type_error():
    with pytest.raises(TypeError):
        validate_our_rating("g4", "4")


def test_validate_our_rating_bool_raises_type_error():
    with pytest.raises(TypeError):
        validate_our_rating("g4", True)


# --- validate_observed_at ---
def test_validate_observed_at_valid():
    result = validate_observed_at("g4", "2017-08-01T10:00:00Z")
    assert result.year == 2017 and result.month == 8


def test_validate_observed_at_bad_format():
    with pytest.raises(ValueError):
        validate_observed_at("g4", "not-a-date")


def test_validate_observed_at_date_only_raises():
    # date-only parses as naive (no timezone) -- rejected like any naive timestamp
    with pytest.raises(ValueError):
        validate_observed_at("g4", "2018-09-01")


def test_validate_observed_at_invalid_time_raises():
    with pytest.raises(ValueError):
        validate_observed_at("g4", "2018-09-01T25:00:00Z")


def test_validate_observed_at_bad_format_chains_cause():
    with pytest.raises(ValueError) as exc_info:
        validate_observed_at("g4", "not-a-date")
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_validate_observed_at_naive_raises():
    with pytest.raises(ValueError):
        validate_observed_at("g4", "2018-09-01T10:00:00")


def test_validate_observed_at_microseconds_raises():
    with pytest.raises(ValueError):
        validate_observed_at("g4", "2018-09-01T10:00:00.5Z")


def test_validate_observed_at_non_string_raises_type_error():
    with pytest.raises(TypeError):
        validate_observed_at("g4", 12345)


def test_validate_observed_at_converts_to_utc():
    result = validate_observed_at("g4", "2018-09-01T15:30:00+05:30")
    assert result == datetime(2018, 9, 1, 10, 0, tzinfo=UTC)


# --- canonical_timestamp ---
def test_canonical_timestamp_equivalent_instants_same_key():
    a = canonical_timestamp(validate_observed_at("g4", "2018-09-01T10:00:00Z"))
    b = canonical_timestamp(validate_observed_at("g4", "2018-09-01T10:00:00+00:00"))
    c = canonical_timestamp(validate_observed_at("g4", "2018-09-01T15:30:00+05:30"))
    assert a == b == c == "2018-09-01T10:00:00Z"


def test_canonical_timestamp_naive_raises():
    with pytest.raises(ValueError):
        canonical_timestamp(datetime(2018, 9, 1, 10, 0))


def test_canonical_timestamp_keys_sort_chronologically():
    early = canonical_timestamp(validate_observed_at("g4", "2018-08-01T10:00:00Z"))
    mid = canonical_timestamp(validate_observed_at("g4", "2018-09-01T10:00:00Z"))
    late = canonical_timestamp(validate_observed_at("g4", "2018-10-01T10:00:00Z"))
    assert sorted([late, early, mid]) == [early, mid, late]


# --- validate_last_observed_mon_yr ---
def test_validate_last_observed_mon_yr_valid():
    result = validate_last_observed_mon_yr("g4", "2026-08-01")
    assert result == date(2026, 8, 1)


def test_validate_last_observed_mon_yr_bad_format_raises():
    with pytest.raises(ValueError):
        validate_last_observed_mon_yr("g4", "not-a-date")


def test_validate_last_observed_mon_yr_non_string_raises_type_error():
    with pytest.raises(TypeError):
        validate_last_observed_mon_yr("g4", 20260801)


# --- check_continuity ---
def test_check_continuity_same_month():
    result = check_continuity(
        "g4", date(2026, 8, 1), datetime(2026, 8, 1, 10, 0, tzinfo=UTC)
    )
    assert result.month == 8


def test_check_continuity_next_month():
    result = check_continuity(
        "g4", date(2026, 8, 1), datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    )
    assert result.month == 9


def test_check_continuity_year_rollover():
    result = check_continuity(
        "g4", date(2026, 12, 1), datetime(2027, 1, 1, 10, 0, tzinfo=UTC)
    )
    assert result.year == 2027 and result.month == 1


def test_check_continuity_skip_rejected():
    with pytest.raises(ValueError):
        check_continuity(
            "g4", date(2026, 8, 1), datetime(2026, 11, 1, 10, 0, tzinfo=UTC)
        )


def test_check_continuity_same_month_different_year_rejected():
    with pytest.raises(ValueError):
        check_continuity(
            "g4", date(2025, 8, 1), datetime(2026, 8, 1, 10, 0, tzinfo=UTC)
        )


def test_check_continuity_across_month_boundary_with_timestamps():
    result = check_continuity(
        "g4", date(2026, 8, 31), datetime(2026, 9, 1, 0, 5, tzinfo=UTC)
    )
    assert (
        result.year == 2026
        and result.month == 9
        and result.day == 1
        and result.hour == 0
        and result.minute == 5
    )


def test_check_continuity_judged_in_utc_allows_next_month():
    # 2018-10-01T02:00:00+05:00 == 2018-09-30T21:00:00Z: September, the
    # month after last_observed's August -- allowed, even though the wall-
    # clock date is already in October.
    observed = validate_observed_at("g4", "2018-10-01T02:00:00+05:00")
    result = check_continuity("g4", date(2018, 8, 1), observed)
    assert result.year == 2018 and result.month == 9


def test_check_continuity_judged_in_utc_rejects_month_after_next():
    # 2018-10-01T10:00:00Z is genuinely October in UTC: two months after
    # last_observed's August, not the next one.
    observed = validate_observed_at("g4", "2018-10-01T10:00:00Z")
    with pytest.raises(ValueError):
        check_continuity("g4", date(2018, 8, 1), observed)


# --- validate_input ---
def test_validate_input_success():
    result = validate_input(
        "bed4",
        [105, 120, 90],
        [4.2, 4.0, 4.1],
        4.0,
        "2026-09-01T10:00:00Z",
        "2026-08-01",
        [100, 110, 95],
    )
    assert result.month == 9


def test_validate_input_bad_prices_raises():
    with pytest.raises(ValueError):
        validate_input(
            "bed4",
            [105, 120],
            [4.2, 4.0, 4.1],
            4.0,
            "2026-09-01T10:00:00Z",
            "2026-08-01",
            [100, 110, 95],
        )


def test_validate_input_bad_continuity_raises():
    with pytest.raises(ValueError):
        validate_input(
            "bed4",
            [105, 120, 90],
            [4.2, 4.0, 4.1],
            4.0,
            "2026-11-01T10:00:00Z",
            "2026-08-01",
            [100, 110, 95],
        )


def test_validate_input_prev_prices_contains_zero():
    with pytest.raises(ValueError, match="must be greater than 0"):
        validate_input(
            "bed4",
            [105, 120, 90],
            [4.2, 4.0, 4.1],
            4.0,
            "2026-09-01T10:00:00Z",
            "2026-08-01",
            [100, 0, 95],
        )


def test_validate_input_prev_prices_wrong_count():
    with pytest.raises(ValueError, match="must have 3 values"):
        validate_input(
            "bed4",
            [105, 120, 90],
            [4.2, 4.0, 4.1],
            4.0,
            "2026-09-01T10:00:00Z",
            "2026-08-01",
            [100, 110],
        )


def test_validate_input_bad_last_observed_format():
    with pytest.raises(ValueError, match="Invalid last_observed_mon_yr"):
        validate_input(
            "bed4",
            [105, 120, 90],
            [4.2, 4.0, 4.1],
            4.0,
            "2026-09-01T10:00:00Z",
            "not-a-date",
            [100, 110, 95],
        )


def test_validate_input_bad_our_rating_raises():
    with pytest.raises(ValueError, match="Invalid our_rating"):
        validate_input(
            "bed4",
            [105, 120, 90],
            [4.2, 4.0, 4.1],
            7.5,
            "2026-09-01T10:00:00Z",
            "2026-08-01",
            [100, 110, 95],
        )


# --- observation_fingerprint ---
def test_observation_fingerprint_same_inputs_same_hash():
    a = observation_fingerprint([80, 40, 35], [4.5, 4.0, 3.8], 4.0)
    b = observation_fingerprint([80, 40, 35], [4.5, 4.0, 3.8], 4.0)
    assert a == b


def test_observation_fingerprint_changed_comp_price_different_hash():
    a = observation_fingerprint([80, 40, 35], [4.5, 4.0, 3.8], 4.0)
    b = observation_fingerprint([80, 40.5, 35], [4.5, 4.0, 3.8], 4.0)
    assert a != b


def test_observation_fingerprint_competitor_order_swapped_different_hash():
    a = observation_fingerprint([80, 40, 35], [4.5, 4.0, 3.8], 4.0)
    b = observation_fingerprint([40, 80, 35], [4.0, 4.5, 3.8], 4.0)
    assert a != b


def test_observation_fingerprint_result_is_64_char_str():
    result = observation_fingerprint([80, 40, 35], [4.5, 4.0, 3.8], 4.0)
    assert isinstance(result, str)
    assert len(result) == 64


def test_observation_fingerprint_int_vs_float_same_hash():
    a = observation_fingerprint([80, 40, 35], [4, 4, 4], 4)
    b = observation_fingerprint([80.0, 40.0, 35.0], [4.0, 4.0, 4.0], 4.0)
    assert a == b


def test_observation_fingerprint_float_noise_same_hash():
    a = observation_fingerprint([39.24, 40, 35], [4.5, 4.0, 3.8], 4.0)
    b = observation_fingerprint([39.24000000000001, 40, 35], [4.5, 4.0, 3.8], 4.0)
    assert a == b


# --- pair_competitors ---
def test_pair_competitors():
    result = pair_competitors("bed4", [105, 120, 90], [4.2, 4.0, 4.1])
    assert result == [
        {"comp_price": 105, "comp_rating": 4.2},
        {"comp_price": 120, "comp_rating": 4.0},
        {"comp_price": 90, "comp_rating": 4.1},
    ]


def test_pair_competitors_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        pair_competitors("bed4", [105, 120, 90], [4.2, 4.0])


# --- competitor_move_check ---
def test_competitor_move_check_proceed():
    result = competitor_move_check("g4", [105, 120, 90], [100, 100, 100])
    assert result["action"] == "proceed"


def test_competitor_move_check_none():
    result = competitor_move_check("g4", [101, 100, 99], [100, 100, 100])
    assert result["action"] == "none"


def test_competitor_move_check_trigger():
    result = competitor_move_check("g4", [140, 120, 90], [100, 100, 100])
    assert result["action"] == "trigger"
    assert result["tier"] == "high"
    # percent_diff covers every competitor, not just the one(s) that triggered
    assert len(result["percent_diff"]) == 3


def test_competitor_move_check_boundary_proceed():
    # exactly 2% -> should still count as "moved" (>=2), not fall below the deadband
    result = competitor_move_check("g4", [102, 100, 100], [100, 100, 100])
    assert result["action"] == "proceed"


def test_competitor_move_check_boundary_trigger():
    # exactly 30% -> should still escalate (>=30)
    result = competitor_move_check("g4", [130, 100, 100], [100, 100, 100])
    assert result["action"] == "trigger"
    assert result["tier"] == "high"


def test_competitor_move_check_negative_trigger():
    # a competitor price DROP of >=30% should escalate too, not just increases
    result = competitor_move_check("g4", [60, 100, 100], [100, 100, 100])
    assert result["action"] == "trigger"
    assert result["tier"] == "high"


# --- build_band ---
def test_build_band():
    assert build_band([105, 120, 90]) == {"low": 90, "high": 120, "mid": 105.0}


# --- rating_gap ---
def test_rating_gap():
    result = rating_gap(4.3, [4.2, 4.0, 4.1])
    assert round(result, 6) == 0.2


# --- choose_target ---
def test_choose_target_high():
    assert choose_target(0.25) == "high"


def test_choose_target_mid():
    assert choose_target(0.1) == "mid"


def test_choose_target_low():
    assert choose_target(-0.3) == "low"


# --- resolve_target_price ---
def test_resolve_target_price_tie_pricier_than_better():
    # tied comp (95, rating 4.0) is PRICIER than the better-rated comp (90, rating 4.2)
    # -> should target just under the better-rated one: 90 - epsilon = 89.99
    band = {"low": 85, "mid": 90, "high": 95}
    comps = [
        {"comp_price": 90, "comp_rating": 4.2},
        {"comp_price": 95, "comp_rating": 4.0},
        {"comp_price": 80, "comp_rating": 3.8},
    ]
    target, cleared, source = resolve_target_price("mid", 0.0, band, 4.0, comps)
    assert round(target, 2) == 89.99
    assert source == "tie_undercut"


def test_resolve_target_price_tie_cheaper_than_better():
    # tied comp (85, rating 4.0) is CHEAPER than the better-rated comp (98, rating 4.2)
    # -> should match the tied comp directly: 85
    band = {"low": 80, "mid": 90, "high": 100}
    comps = [
        {"comp_price": 98, "comp_rating": 4.2},
        {"comp_price": 85, "comp_rating": 4.0},
        {"comp_price": 75, "comp_rating": 3.8},
    ]
    target, cleared, source = resolve_target_price("mid", 0.0, band, 4.0, comps)
    assert target == 85
    assert source == "tie_match"


def test_resolve_target_price_top_rated_premium():
    band = {"low": 96, "mid": 100, "high": 104}
    comps = [
        {"comp_price": 102, "comp_rating": 4.0},
        {"comp_price": 104, "comp_rating": 3.9},
        {"comp_price": 96, "comp_rating": 3.8},
    ]
    target, cleared, source = resolve_target_price("high", 0.35, band, 4.5, comps)
    assert round(target, 2) == 113.10
    assert source == "top_premium"


def test_resolve_target_price_plain_no_special_case():
    # no tie, not top-rated (a comp outranks us) -> just band[target_label]
    band = {"low": 80, "mid": 90, "high": 100}
    comps = [
        {"comp_price": 85, "comp_rating": 4.5},
        {"comp_price": 78, "comp_rating": 3.5},
        {"comp_price": 95, "comp_rating": 3.0},
    ]
    target, cleared, source = resolve_target_price("mid", 0.05, band, 4.0, comps)
    assert target == 90
    assert cleared is None
    assert source == "band"


def test_resolve_target_price_multi_tie_undercut():
    comp_details = pair_competitors("g4", [45, 30, 50], [4.5, 4.0, 4.0])
    band = {"low": 30, "mid": 40, "high": 50}
    target, cleared, source = resolve_target_price("mid", 0.0, band, 4.0, comp_details)
    assert round(target, 2) == 44.99
    assert cleared == {"comp_price": 45, "comp_rating": 4.5}
    assert source == "tie_undercut"


def test_resolve_target_price_multi_tie_undercut_order_independent():
    # same prices/ratings as above, reordered -> same result
    comp_details = pair_competitors("g4", [45, 50, 30], [4.5, 4.0, 4.0])
    band = {"low": 30, "mid": 40, "high": 50}
    target, cleared, source = resolve_target_price("mid", 0.0, band, 4.0, comp_details)
    assert round(target, 2) == 44.99
    assert cleared == {"comp_price": 45, "comp_rating": 4.5}
    assert source == "tie_undercut"


def test_resolve_target_price_multi_tie_match():
    comp_details = pair_competitors("g4", [45, 30, 40], [4.5, 4.0, 4.0])
    band = {"low": 30, "mid": 37.5, "high": 45}
    target, cleared, source = resolve_target_price("mid", 0.0, band, 4.0, comp_details)
    assert target == 40
    assert cleared == {"comp_price": 45, "comp_rating": 4.5}
    assert source == "tie_match"


def test_resolve_target_price_top_rated_but_not_high_target():
    # top-rated, but target isn't "high" -> premium branch must not apply
    band = {"low": 90, "mid": 100, "high": 110}
    comps = [
        {"comp_price": 95, "comp_rating": 4.0},
        {"comp_price": 105, "comp_rating": 3.9},
        {"comp_price": 90, "comp_rating": 3.8},
    ]
    target, cleared, source = resolve_target_price("mid", 0.3, band, 4.5, comps)
    assert target == 100
    assert cleared is None
    assert source == "band"


# --- check_move_size ---
def test_check_move_size_escalate():
    path, x = check_move_size(130, 100)
    assert path == "escalate"


def test_check_move_size_llm():
    path, x = check_move_size(104, 100)
    assert path == "llm"


def test_check_move_size_direct():
    path, x = check_move_size(109, 100)
    assert path == "direct"


def test_check_move_size_boundary_llm_upper():
    # exactly 5% -> "llm" (condition is <=5)
    path, x = check_move_size(105, 100)
    assert path == "llm"
    assert x == 5


def test_check_move_size_boundary_direct_upper():
    # exactly 10% -> "direct", not "escalate" (condition is strictly >10)
    path, x = check_move_size(110, 100)
    assert path == "direct"
    assert x == 10


# --- compute_step ---
def test_compute_step_direct():
    assert compute_step(100, 109.2, 100, None) == 109.2


def test_compute_step_llm_partial():
    result = compute_step(99, 95, 100, 0.02)
    assert result == 97


def test_compute_step_llm_overshoot_clamps_to_target():
    # requested step (20) is larger than the remaining distance (10) -> land exactly on target, no overshoot
    result = compute_step(100, 110, 100, 0.2)
    assert result == 110


# --- apply_dominance_clamp ---
def test_apply_dominance_clamp_fires():
    comps = [{"comp_price": 102, "comp_rating": 4.0}]
    price, side = apply_dominance_clamp(104, 3.95, comps, direction=1)
    assert round(price, 2) == 101.99
    assert side == "below_higher_rated"


def test_apply_dominance_clamp_no_skip_for_cleared_comp():
    # cleared_comp is now for logging only -- the clamp checks every
    # higher-rated competitor, so a price at/above it still gets clamped
    comps = [{"comp_price": 98, "comp_rating": 4.2}]
    price, side = apply_dominance_clamp(99, 4.0, comps, direction=1)
    assert round(price, 2) == 97.99
    assert side == "below_higher_rated"


def test_apply_dominance_clamp_downward_past_one_lower_rated():
    comps = [{"comp_price": 90, "comp_rating": 3.8}]
    price, side = apply_dominance_clamp(88, 4.0, comps, direction=-1)
    assert round(price, 2) == 90.01
    assert side == "above_lower_rated"


def test_apply_dominance_clamp_downward_two_lower_rated_order_a():
    comps = [
        {"comp_price": 90, "comp_rating": 3.8},
        {"comp_price": 95, "comp_rating": 3.5},
    ]
    price, side = apply_dominance_clamp(80, 4.0, comps, direction=-1)
    assert round(price, 2) == 95.01
    assert side == "above_lower_rated"


def test_apply_dominance_clamp_downward_two_lower_rated_order_b():
    comps = [
        {"comp_price": 95, "comp_rating": 3.5},
        {"comp_price": 90, "comp_rating": 3.8},
    ]
    price, side = apply_dominance_clamp(80, 4.0, comps, direction=-1)
    assert round(price, 2) == 95.01
    assert side == "above_lower_rated"


def test_apply_dominance_clamp_upward_ignores_lower_rated():
    comps = [{"comp_price": 90, "comp_rating": 3.8}]
    price, side = apply_dominance_clamp(95, 4.0, comps, direction=1)
    assert price == 95
    assert side is None


def test_apply_dominance_clamp_downward_ignores_higher_rated():
    comps = [{"comp_price": 102, "comp_rating": 4.2}]
    price, side = apply_dominance_clamp(99, 4.0, comps, direction=-1)
    assert price == 99
    assert side is None


def test_apply_dominance_clamp_no_move_returns_unchanged():
    comps = [
        {"comp_price": 90, "comp_rating": 3.8},
        {"comp_price": 102, "comp_rating": 4.2},
    ]
    price, side = apply_dominance_clamp(99, 4.0, comps, direction=0)
    assert price == 99
    assert side is None


def test_apply_dominance_clamp_downward_conflict_higher_rated_wins():
    comps = [
        {"comp_price": 99, "comp_rating": 3.5},
        {"comp_price": 98, "comp_rating": 4.5},
    ]
    price, side = apply_dominance_clamp(96, 4.0, comps, direction=-1)
    assert round(price, 2) == 97.99
    assert side == "below_higher_rated"


def test_apply_dominance_clamp_downward_no_conflict():
    comps = [{"comp_price": 99, "comp_rating": 3.5}]
    price, side = apply_dominance_clamp(96, 4.0, comps, direction=-1)
    assert round(price, 2) == 99.01
    assert side == "above_lower_rated"


def test_apply_dominance_clamp_downward_reversal_past_current_price():
    comps = [{"comp_price": 101, "comp_rating": 3.5}]
    price, side = apply_dominance_clamp(98, 4.0, comps, direction=-1)
    assert round(price, 2) == 101.01
    assert side == "above_lower_rated"


# --- enforce_bounds ---
def test_enforce_bounds_below_floor():
    assert enforce_bounds("g4", 85, 90, 130) == 90


def test_enforce_bounds_above_ceiling():
    assert enforce_bounds("g4", 135, 90, 130) == 130


def test_enforce_bounds_within_range():
    assert enforce_bounds("g4", 105, 90, 130) == 105


def test_enforce_bounds_at_floor_boundary():
    # price == min_price exactly: comparison is strict (<), must not be treated as "below floor"
    assert enforce_bounds("g4", 90, 90, 130) == 90


def test_enforce_bounds_at_ceiling_boundary():
    # price == max_price exactly: comparison is strict (>), must not be treated as "above ceiling"
    assert enforce_bounds("g4", 130, 90, 130) == 130


# --- months_to_reach ---
def test_months_to_reach_rising_target():
    assert months_to_reach(100, 125) == 3


def test_months_to_reach_falling_target():
    # linear (distance / cap) would say 3; compounding needs 4
    assert months_to_reach(100, 72) == 4


def test_months_to_reach_rising_target_large():
    # linear would say 5; compounding needs only 4
    assert months_to_reach(100, 146) == 4


def test_months_to_reach_exact_boundary():
    # 0.9**4 == 0.6561 exactly, but the raw ratio is 4.000000000000001 in
    # floating point; without round() ceil would give 5, not 4
    assert months_to_reach(100, 65.61) == 4


def test_months_to_reach_small_move():
    assert months_to_reach(100, 105) == 1


def test_months_to_reach_target_equals_anchor():
    assert months_to_reach(100, 100) == 0


def test_months_to_reach_anchor_not_positive():
    with pytest.raises(ValueError):
        months_to_reach(0, 100)


def test_months_to_reach_target_not_positive():
    with pytest.raises(ValueError, match="Invalid target"):
        months_to_reach(100, 0)
