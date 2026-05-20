"""Tests for closed-form analytical scoring in hybrid_strategy_analytics.py."""

import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ssf.hybrid_strategy_spec import HybridStrategySpec
from ssf.hybrid_strategy_analytics import (
    majority_tie_probability,
    majority_success_probability,
    majority_success_probability_given_not_tie,
    pyramid_success_probability,
    analytical_win_rate,
)


# ---------------------------------------------------------------------------
# majority_tie_probability
# ---------------------------------------------------------------------------

def test_majority_tie_probability_n6_b0():
    # C(6,3) / 2**6 = 20 / 64
    expected = math.comb(6, 3) / 64
    assert majority_tie_probability(6, 0) == pytest.approx(expected)
    assert majority_tie_probability(6, 0) == pytest.approx(0.3125)


def test_majority_tie_probability_n8_b0():
    # C(8,4) / 2**8 = 70 / 256
    expected = math.comb(8, 4) / 256
    assert majority_tie_probability(8, 0) == pytest.approx(expected)
    assert majority_tie_probability(8, 0) == pytest.approx(0.2734375)


# ---------------------------------------------------------------------------
# majority_success_probability
# ---------------------------------------------------------------------------

def test_majority_success_probability_n8_b0():
    assert majority_success_probability(8, 0) == pytest.approx(0.63671875)


def test_majority_success_probability_n6_b0():
    assert majority_success_probability(6, 0) == pytest.approx(0.65625)


def test_majority_success_probability_given_not_tie_n2_b0():
    assert majority_success_probability_given_not_tie(2, 0) == pytest.approx(1.0)


def test_majority_success_probability_given_not_tie_n4_b0():
    assert majority_success_probability_given_not_tie(4, 0) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# pyramid_success_probability
# ---------------------------------------------------------------------------

def test_pyramid_success_probability_depth3_p09():
    result = pyramid_success_probability(3, 0.9)
    assert result == pytest.approx((1 + (0.8) ** 3) / 2)
    assert result == pytest.approx(0.756)


def test_pyramid_success_probability_depth3_p05():
    assert pyramid_success_probability(3, 0.5) == pytest.approx(0.5)


def test_pyramid_success_probability_depth3_p10():
    assert pyramid_success_probability(3, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# analytical_win_rate — pure majority
# ---------------------------------------------------------------------------

def test_analytical_pure_majority_n8():
    spec = HybridStrategySpec(family="majority", n2=8, k_box=7, tie_bandwidth=0)
    result = analytical_win_rate(spec, p_rule=0.9)
    assert result.success == pytest.approx(0.63671875)
    assert result.exact is True
    assert result.method == "closed_form"


# ---------------------------------------------------------------------------
# analytical_win_rate — pure pyramid
# ---------------------------------------------------------------------------

def test_analytical_pure_pyramid_n8_p09():
    spec = HybridStrategySpec(family="pyramid", n2=8, k_box=7)
    result = analytical_win_rate(spec, p_rule=0.9)
    assert result.success == pytest.approx(0.756)
    assert result.exact is True
    assert result.method == "closed_form"


# ---------------------------------------------------------------------------
# analytical_win_rate — horizontal
# ---------------------------------------------------------------------------

def test_analytical_horizontal_n4_split2_2_p08():
    spec = HybridStrategySpec(
        family="horizontal", n2=4, k_box=3,
        split=2, order="maj_pyr", tie_bandwidth=0,
    )
    result = analytical_win_rate(spec, p_rule=0.8)
    assert result.success == pytest.approx(0.70, abs=1e-12)


def test_analytical_horizontal_n8_split6_2_p07():
    spec = HybridStrategySpec(
        family="horizontal", n2=8, k_box=7,
        split=6, order="maj_pyr", tie_bandwidth=0,
    )
    result = analytical_win_rate(spec, p_rule=0.7)
    assert result.success == pytest.approx(0.6328125, abs=1e-12)


def test_analytical_horizontal_n8_split6_2_p08():
    spec = HybridStrategySpec(
        family="horizontal", n2=8, k_box=7,
        split=6, order="maj_pyr", tie_bandwidth=0,
    )
    result = analytical_win_rate(spec, p_rule=0.8)
    assert result.success == pytest.approx(0.640625, abs=1e-12)
    assert result.exact is True


def test_analytical_horizontal_n8_split6_2_p09():
    spec = HybridStrategySpec(
        family="horizontal", n2=8, k_box=7,
        split=6, order="maj_pyr", tie_bandwidth=0,
    )
    result = analytical_win_rate(spec, p_rule=0.9)
    assert result.success == pytest.approx(0.6484375, abs=1e-12)
    assert result.exact is True


# ---------------------------------------------------------------------------
# analytical_win_rate — vertical
# ---------------------------------------------------------------------------

def test_analytical_vertical_maj_pyr_smoke():
    spec = HybridStrategySpec(
        family="vertical", n2=8, k_box=7,
        depth_s=2, order="maj_pyr", tie_bandwidth=0,
    )
    result = analytical_win_rate(spec, p_rule=0.9)
    assert 0.0 <= result.success <= 1.0
    assert result.method == "closed_form"
    assert result.exact is True
    assert "maj_pyr" in result.notes
    assert "XOR double-error cancellation" in result.notes


def test_analytical_vertical_pyr_maj_smoke():
    spec = HybridStrategySpec(
        family="vertical", n2=8, k_box=7,
        depth_s=2, order="pyr_maj", tie_bandwidth=0,
    )
    result = analytical_win_rate(spec, p_rule=0.9)
    assert 0.0 <= result.success <= 1.0
    assert result.method == "closed_form"
    assert result.exact is True
    assert "pyr_maj" in result.notes
    assert "XOR double-error cancellation" in result.notes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_p_rule_below_zero_raises():
    spec = HybridStrategySpec(family="majority", n2=8, k_box=7, tie_bandwidth=0)
    with pytest.raises(ValueError, match="p_rule"):
        analytical_win_rate(spec, p_rule=-0.01)


def test_p_rule_above_one_raises():
    spec = HybridStrategySpec(family="majority", n2=8, k_box=7, tie_bandwidth=0)
    with pytest.raises(ValueError, match="p_rule"):
        analytical_win_rate(spec, p_rule=1.01)


def test_invalid_spec_raises():
    # k_box too small for pyramid (requires n2-1=7, given 3)
    spec = HybridStrategySpec(family="pyramid", n2=8, k_box=3)
    with pytest.raises(ValueError):
        analytical_win_rate(spec, p_rule=0.9)
