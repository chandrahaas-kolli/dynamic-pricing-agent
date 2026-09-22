import pytest
from src.pricing import (
    validate_prices, validate_ratings, validate_observed_at, check_continuity,
    validate_input, pair_competitors, competitor_move_check, build_band,
    rating_gap, choose_target, resolve_target_price, check_move_size,
    compute_step, apply_dominance_clamp, enforce_bounds,
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

def test_validate_ratings_above_max():
    with pytest.raises(ValueError):
        validate_ratings("g4", [4.2, 5.1, 4.1])

def test_validate_ratings_exactly_five_valid():
    assert validate_ratings("g4", [4.2, 5.0, 4.1]) == [4.2, 5.0, 4.1]


# --- validate_observed_at ---
def test_validate_observed_at_valid():
    result = validate_observed_at("g4", "01-08-2017")
    assert result.year == 2017 and result.month == 8

def test_validate_observed_at_bad_format():
    with pytest.raises(ValueError):
        validate_observed_at("g4", "not-a-date")


# --- check_continuity ---
def test_check_continuity_same_month():
    result = check_continuity("g4", "01-08-2026", "01-08-2026")
    assert result.month == 8

def test_check_continuity_next_month():
    result = check_continuity("g4", "01-08-2026", "01-09-2026")
    assert result.month == 9

def test_check_continuity_year_rollover():
    result = check_continuity("g4", "01-12-2026", "01-01-2027")
    assert result.year == 2027 and result.month == 1

def test_check_continuity_skip_rejected():
    with pytest.raises(ValueError):
        check_continuity("g4", "01-08-2026", "01-11-2026")

def test_check_continuity_same_month_different_year_rejected():
    with pytest.raises(ValueError):
        check_continuity("g4", "01-08-2025", "01-08-2026")


# --- validate_input ---
def test_validate_input_success():
    result = validate_input("bed4", [105, 120, 90], [4.2, 4.0, 4.1], "01-09-2026", "01-08-2026")
    assert result.month == 9

def test_validate_input_bad_prices_raises():
    with pytest.raises(ValueError):
        validate_input("bed4", [105, 120], [4.2, 4.0, 4.1], "01-09-2026", "01-08-2026")

def test_validate_input_bad_continuity_raises():
    with pytest.raises(ValueError):
        validate_input("bed4", [105, 120, 90], [4.2, 4.0, 4.1], "01-11-2026", "01-08-2026")


# --- pair_competitors ---
def test_pair_competitors():
    result = pair_competitors("bed4", [105, 120, 90], [4.2, 4.0, 4.1])
    assert result == [
        {"comp_price": 105, "comp_rating": 4.2},
        {"comp_price": 120, "comp_rating": 4.0},
        {"comp_price": 90, "comp_rating": 4.1},
    ]


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
def test_resolve_target_price_tie_cheaper_than_better():
    # tied comp (95, rating 4.0) is PRICIER than the better-rated comp (90, rating 4.2)
    # -> should target just under the better-rated one: 90 - epsilon = 89.99
    band = {"low": 85, "mid": 90, "high": 95}
    comps = [{"comp_price": 90, "comp_rating": 4.2}, {"comp_price": 95, "comp_rating": 4.0}, {"comp_price": 80, "comp_rating": 3.8}]
    target, cleared = resolve_target_price("mid", 0.0, band, 4.0, comps)
    assert round(target, 2) == 89.99

def test_resolve_target_price_tie_pricier_than_better():
    # tied comp (85, rating 4.0) is CHEAPER than the better-rated comp (98, rating 4.2)
    # -> should match the tied comp directly: 85
    band = {"low": 80, "mid": 90, "high": 100}
    comps = [{"comp_price": 98, "comp_rating": 4.2}, {"comp_price": 85, "comp_rating": 4.0}, {"comp_price": 75, "comp_rating": 3.8}]
    target, cleared = resolve_target_price("mid", 0.0, band, 4.0, comps)
    assert target == 85

def test_resolve_target_price_top_rated_premium():
    band = {"low": 96, "mid": 100, "high": 104}
    comps = [{"comp_price": 102, "comp_rating": 4.0}, {"comp_price": 104, "comp_rating": 3.9}, {"comp_price": 96, "comp_rating": 3.8}]
    target, cleared = resolve_target_price("high", 0.35, band, 4.5, comps)
    assert round(target, 2) == 113.10

def test_resolve_target_price_plain_no_special_case():
    # no tie, not top-rated (a comp outranks us) -> just band[target_label]
    band = {"low": 80, "mid": 90, "high": 100}
    comps = [{"comp_price": 85, "comp_rating": 4.5}, {"comp_price": 78, "comp_rating": 3.5}, {"comp_price": 95, "comp_rating": 3.0}]
    target, cleared = resolve_target_price("mid", 0.05, band, 4.0, comps)
    assert target == 90
    assert cleared is None

def test_resolve_target_price_top_rated_but_not_high_target():
    # top-rated, but target isn't "high" -> premium branch must not apply
    band = {"low": 90, "mid": 100, "high": 110}
    comps = [{"comp_price": 95, "comp_rating": 4.0}, {"comp_price": 105, "comp_rating": 3.9}, {"comp_price": 90, "comp_rating": 3.8}]
    target, cleared = resolve_target_price("mid", 0.3, band, 4.5, comps)
    assert target == 100
    assert cleared is None


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
    result = apply_dominance_clamp(104, 3.95, comps)
    assert round(result, 2) == 101.99

def test_apply_dominance_clamp_no_skip_for_cleared_comp():
    # cleared_comp is now for logging only -- the clamp checks every
    # higher-rated competitor, so a price at/above it still gets clamped
    comps = [{"comp_price": 98, "comp_rating": 4.2}]
    result = apply_dominance_clamp(99, 4.0, comps)
    assert round(result, 2) == 97.99


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