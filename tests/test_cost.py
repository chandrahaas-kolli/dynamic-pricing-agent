import pytest

from src.cost import estimate_cost_usd


def test_estimate_cost_usd_valid():
    assert estimate_cost_usd(1000, 200, 3.00, 15.00) == pytest.approx(0.006)


def test_estimate_cost_usd_zero_tokens():
    assert estimate_cost_usd(0, 0, 3.00, 15.00) == 0.0


@pytest.mark.parametrize("arg_name", ["input_tokens", "output_tokens"])
def test_estimate_cost_usd_negative_raises_value_error(arg_name):
    kwargs = {"input_tokens": 100, "output_tokens": 50}
    kwargs[arg_name] = -1
    with pytest.raises(ValueError, match=arg_name):
        estimate_cost_usd(
            input_price_per_mtok=3.00, output_price_per_mtok=15.00, **kwargs
        )


@pytest.mark.parametrize("arg_name", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("bad_value", [True, 1.5, "100"])
def test_estimate_cost_usd_wrong_type_raises_type_error(arg_name, bad_value):
    kwargs = {"input_tokens": 100, "output_tokens": 50}
    kwargs[arg_name] = bad_value
    with pytest.raises(TypeError, match=arg_name):
        estimate_cost_usd(
            input_price_per_mtok=3.00, output_price_per_mtok=15.00, **kwargs
        )
