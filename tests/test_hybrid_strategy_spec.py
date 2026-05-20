from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_spec import (
    HybridStrategySpec,
    candidate_name,
    even_tie_values,
    is_power_of_two,
    required_boxes,
    stable_key,
    validate_spec,
    with_required_boxes,
)


def test_is_power_of_two_cases() -> None:
    assert is_power_of_two(1)
    assert is_power_of_two(2)
    assert is_power_of_two(4)
    assert is_power_of_two(8)
    assert not is_power_of_two(0)
    assert not is_power_of_two(3)
    assert not is_power_of_two(6)


def test_majority_valid() -> None:
    spec = HybridStrategySpec("majority", n2=8, k_box=7, tie_bandwidth=0)
    validate_spec(spec)
    assert required_boxes(spec) == 0
    payload = with_required_boxes(spec)
    assert payload["unused_boxes"] == 7
    assert candidate_name(spec) == "majority_n2_8_k_7_b_0"


def test_majority_candidate_name_differs_by_tie_bandwidth() -> None:
    spec_b0 = HybridStrategySpec("majority", n2=8, k_box=7, tie_bandwidth=0)
    spec_b2 = HybridStrategySpec("majority", n2=8, k_box=7, tie_bandwidth=2)

    assert candidate_name(spec_b0) == "majority_n2_8_k_7_b_0"
    assert candidate_name(spec_b2) == "majority_n2_8_k_7_b_2"
    assert candidate_name(spec_b0) != candidate_name(spec_b2)


def test_majority_invalid_cases() -> None:
    with pytest.raises(ValueError, match="even n2"):
        validate_spec(HybridStrategySpec("majority", n2=7, k_box=7, tie_bandwidth=0))
    with pytest.raises(ValueError, match="tie_bandwidth out of range"):
        validate_spec(HybridStrategySpec("majority", n2=8, k_box=7, tie_bandwidth=4))
    with pytest.raises(ValueError, match="does not allow"):
        validate_spec(HybridStrategySpec("majority", n2=8, k_box=7, tie_bandwidth=0, order="maj_pyr"))
    with pytest.raises(ValueError, match="tie_bandwidth must be even"):
        validate_spec(HybridStrategySpec("majority", n2=8, k_box=7, tie_bandwidth=1))
    # n2=8, tie_bandwidth=2 is valid
    validate_spec(HybridStrategySpec("majority", n2=8, k_box=7, tie_bandwidth=2))


def test_pyramid_valid_and_name() -> None:
    spec = HybridStrategySpec("pyramid", n2=8, k_box=7)
    validate_spec(spec)
    assert required_boxes(spec) == 7
    payload = with_required_boxes(spec)
    assert payload["unused_boxes"] == 0
    assert candidate_name(spec) == "pyramid_n2_8_k_7"


def test_pyramid_extra_boxes_valid() -> None:
    spec = HybridStrategySpec("pyramid", n2=8, k_box=10)
    validate_spec(spec)
    assert required_boxes(spec) == 7
    assert with_required_boxes(spec)["unused_boxes"] == 3


def test_pyramid_invalid_cases() -> None:
    with pytest.raises(ValueError, match="power of two"):
        validate_spec(HybridStrategySpec("pyramid", n2=6, k_box=10))
    with pytest.raises(ValueError, match="k_box too small"):
        validate_spec(HybridStrategySpec("pyramid", n2=8, k_box=6))
    with pytest.raises(ValueError, match="does not allow"):
        validate_spec(HybridStrategySpec("pyramid", n2=8, k_box=7, tie_bandwidth=0))


def test_horizontal_maj_pyr_valid() -> None:
    spec = HybridStrategySpec("horizontal", n2=8, k_box=7, split=6, order="maj_pyr", tie_bandwidth=0)
    validate_spec(spec)
    assert required_boxes(spec) == 1
    payload = with_required_boxes(spec)
    assert payload["unused_boxes"] == 6
    assert candidate_name(spec) == "hybrid_horiz_maj_pyr_n2_8_k_7_split_6_2_b_0"


def test_horizontal_pyr_maj_valid() -> None:
    spec = HybridStrategySpec("horizontal", n2=8, k_box=7, split=2, order="pyr_maj", tie_bandwidth=0)
    validate_spec(spec)
    assert required_boxes(spec) == 1
    assert candidate_name(spec) == "hybrid_horiz_pyr_maj_n2_8_k_7_split_2_6_b_0"


def test_horizontal_invalid_cases() -> None:
    with pytest.raises(ValueError, match="split must satisfy"):
        validate_spec(HybridStrategySpec("horizontal", n2=8, k_box=7, split=0, order="maj_pyr", tie_bandwidth=0))
    with pytest.raises(ValueError, match="majority side must have even"):
        validate_spec(HybridStrategySpec("horizontal", n2=8, k_box=7, split=5, order="maj_pyr", tie_bandwidth=0))
    with pytest.raises(ValueError, match="pyramid side length must be a power of two"):
        validate_spec(HybridStrategySpec("horizontal", n2=12, k_box=11, split=6, order="maj_pyr", tie_bandwidth=0))
    with pytest.raises(ValueError, match="tie_bandwidth out of range"):
        validate_spec(HybridStrategySpec("horizontal", n2=8, k_box=7, split=6, order="maj_pyr", tie_bandwidth=3))
    with pytest.raises(ValueError, match="k_box too small"):
        validate_spec(HybridStrategySpec("horizontal", n2=8, k_box=0, split=6, order="maj_pyr", tie_bandwidth=0))
    # n2=8, split=6, order=maj_pyr, tie_bandwidth=1 is invalid
    with pytest.raises(ValueError, match="tie_bandwidth must be even"):
        validate_spec(HybridStrategySpec("horizontal", n2=8, k_box=7, split=6, order="maj_pyr", tie_bandwidth=1))
    # tie_bandwidth=0 valid
    validate_spec(HybridStrategySpec("horizontal", n2=8, k_box=7, split=6, order="maj_pyr", tie_bandwidth=0))


def test_vertical_maj_pyr_tie_bandwidth_even() -> None:
    # n2=8, depth_s=2, order=maj_pyr, tie_bandwidth=1 is invalid
    with pytest.raises(ValueError, match="tie_bandwidth"):
        validate_spec(HybridStrategySpec("vertical", n2=8, k_box=7, depth_s=2, order="maj_pyr", tie_bandwidth=1))


def test_vertical_pyr_maj_tie_bandwidth_even() -> None:
    # n2=8, depth_s=2, order=pyr_maj, tie_bandwidth=1 is invalid
    with pytest.raises(ValueError, match="tie_bandwidth"):
        validate_spec(HybridStrategySpec("vertical", n2=8, k_box=7, depth_s=2, order="pyr_maj", tie_bandwidth=1))


def test_vertical_maj_pyr_tie_bandwidth_out_of_range() -> None:
    # n2=8, depth_s=2 => block_size=2, so valid tie_bandwidth is only 0
    with pytest.raises(ValueError, match="tie_bandwidth out of range"):
        validate_spec(HybridStrategySpec("vertical", n2=8, k_box=7, depth_s=2, order="maj_pyr", tie_bandwidth=2))


def test_vertical_pyr_maj_tie_bandwidth_out_of_range() -> None:
    # n2=8, depth_s=2 => apex_len=2, so valid tie_bandwidth is only 0
    with pytest.raises(ValueError, match="tie_bandwidth out of range"):
        validate_spec(HybridStrategySpec("vertical", n2=8, k_box=7, depth_s=2, order="pyr_maj", tie_bandwidth=2))


def test_vertical_maj_pyr_valid() -> None:
    spec = HybridStrategySpec("vertical", n2=8, k_box=7, depth_s=2, order="maj_pyr", tie_bandwidth=0)
    validate_spec(spec)
    assert required_boxes(spec) == 3
    payload = with_required_boxes(spec)
    assert payload["unused_boxes"] == 4
    assert candidate_name(spec) == "hybrid_vert_maj_pyr_n2_8_k_7_s_2_b_0"


def test_vertical_pyr_maj_valid() -> None:
    spec = HybridStrategySpec("vertical", n2=8, k_box=7, depth_s=2, order="pyr_maj", tie_bandwidth=0)
    validate_spec(spec)
    assert required_boxes(spec) == 6
    payload = with_required_boxes(spec)
    assert payload["unused_boxes"] == 1
    assert candidate_name(spec) == "hybrid_vert_pyr_maj_n2_8_k_7_s_2_b_0"


def test_vertical_invalid_cases() -> None:
    with pytest.raises(ValueError, match="nontrivial depth_s"):
        validate_spec(HybridStrategySpec("vertical", n2=8, k_box=7, depth_s=0, order="maj_pyr", tie_bandwidth=0))
    with pytest.raises(ValueError, match="nontrivial depth_s"):
        validate_spec(HybridStrategySpec("vertical", n2=8, k_box=7, depth_s=3, order="maj_pyr", tie_bandwidth=0))
    with pytest.raises(ValueError, match="power of two"):
        validate_spec(HybridStrategySpec("vertical", n2=6, k_box=7, depth_s=1, order="maj_pyr", tie_bandwidth=0))
    with pytest.raises(ValueError, match="k_box too small"):
        validate_spec(HybridStrategySpec("vertical", n2=8, k_box=2, depth_s=2, order="maj_pyr", tie_bandwidth=0))
    with pytest.raises(ValueError, match="k_box too small"):
        validate_spec(HybridStrategySpec("vertical", n2=8, k_box=5, depth_s=2, order="pyr_maj", tie_bandwidth=0))


def test_stable_key_equivalence_and_difference() -> None:
    spec1 = HybridStrategySpec("horizontal", n2=8, k_box=7, split=6, order="maj_pyr", tie_bandwidth=0)
    spec2 = HybridStrategySpec("horizontal", n2=8, k_box=7, split=6, order="maj_pyr", tie_bandwidth=0)
    spec4 = HybridStrategySpec("horizontal", n2=8, k_box=8, split=6, order="maj_pyr", tie_bandwidth=0)

    assert stable_key(spec1) == stable_key(spec2)
    assert stable_key(spec1) != stable_key(spec4)


def test_with_required_boxes_serializable() -> None:
    spec = HybridStrategySpec("vertical", n2=8, k_box=7, depth_s=2, order="pyr_maj", tie_bandwidth=0)
    payload = with_required_boxes(spec)
    assert payload["required_boxes"] == 6
    assert payload["unused_boxes"] == 1
    json.dumps(payload)


def test_even_tie_values_valid() -> None:
    assert even_tie_values(2) == [0]
    assert even_tie_values(6) == [0, 2]
    assert even_tie_values(8) == [0, 2]


def test_even_tie_values_invalid() -> None:
    with pytest.raises(ValueError):
        even_tie_values(0)
    with pytest.raises(ValueError):
        even_tie_values(-2)
    with pytest.raises(ValueError):
        even_tie_values(3)
