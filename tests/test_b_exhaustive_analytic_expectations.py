from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.bob_exhaustive import evaluate_strategy_exhaustive
from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name
from ssf.strategy_io import load_manifest, load_strategy


FIXTURE_DIR = PROJECT_ROOT / "fixtures" / "persistent" / "a_strategies"
ANALYTIC_MAJORITY_NO_BOX_SCORE = 11 / 16


def _spec_from_manifest_entry(entry: dict) -> HybridStrategySpec:
    return HybridStrategySpec(
        family=entry["family"],
        n2=entry["n2"],
        k_box=entry["k_box"],
        split=entry.get("split"),
        depth_s=entry.get("depth_s"),
        order=entry.get("order"),
        tie_bandwidth=entry.get("tie_bandwidth"),
        tie_value=entry.get("tie_value", 0),
    )


def _entry_by_spec(spec: HybridStrategySpec) -> dict:
    expected_name = candidate_name(spec)
    manifest = load_manifest(FIXTURE_DIR / "manifest.json")
    entry = next((e for e in manifest if e["name"] == expected_name), None)
    if entry is None:
        raise AssertionError(f"Missing fixture for spec name: {expected_name}")
    assert _spec_from_manifest_entry(entry) == spec
    assert entry["file"] == expected_name + ".npz"
    return entry


def _load_strategy_by_spec(spec: HybridStrategySpec):
    entry = _entry_by_spec(spec)
    strategy = load_strategy(FIXTURE_DIR / entry["file"])
    assert int(strategy["n2"]) == int(entry["n2"])
    assert int(strategy["k_box"]) == int(entry["k_box"])
    return strategy


@pytest.mark.parametrize("p_rule", [0.5, 0.65, 0.77, 1.0])
def test_majority_no_box_matches_analytic_expectation(p_rule):
    strategy = _load_strategy_by_spec(HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0))

    score = evaluate_strategy_exhaustive(
        strategy,
        p_rule=p_rule,
        box_limit=0,
        subset_policy="up_to",
    )

    assert score == pytest.approx(ANALYTIC_MAJORITY_NO_BOX_SCORE)


def test_majority_no_box_is_independent_of_p_rule():
    strategy = _load_strategy_by_spec(HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0))

    scores = [
        evaluate_strategy_exhaustive(
            strategy,
            p_rule=p_rule,
            box_limit=0,
            subset_policy="up_to",
        )
        for p_rule in [0.5, 0.65, 0.77, 1.0]
    ]

    for score in scores:
        assert score == pytest.approx(ANALYTIC_MAJORITY_NO_BOX_SCORE)

    assert scores[0] == pytest.approx(scores[1])
    assert scores[1] == pytest.approx(scores[2])
    assert scores[2] == pytest.approx(scores[3])


def test_majority_no_box_ignores_expected_scores_payload():
    strategy = _load_strategy_by_spec(HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0))
    strategy["expected_scores"] = {"0.5": 999.0, "1.0": -123.0}

    score = evaluate_strategy_exhaustive(
        strategy,
        p_rule=0.5,
        box_limit=0,
        subset_policy="up_to",
    )

    assert score == pytest.approx(ANALYTIC_MAJORITY_NO_BOX_SCORE)


@pytest.mark.parametrize("p_rule", [0.5, 0.65, 0.77, 1.0])
def test_majority_k3_no_box_matches_analytic_expectation(p_rule):
    strategy = _load_strategy_by_spec(HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0))

    score = evaluate_strategy_exhaustive(
        strategy,
        p_rule=p_rule,
        box_limit=0,
        subset_policy="up_to",
    )

    assert score == pytest.approx(11 / 16)


@pytest.mark.parametrize("p_rule", [0.5, 0.65, 0.77, 1.0])
def test_majority_k3_full_box_limit_still_matches_analytic_expectation(p_rule):
    strategy = _load_strategy_by_spec(HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0))

    score = evaluate_strategy_exhaustive(
        strategy,
        p_rule=p_rule,
        box_limit=3,
        subset_policy="up_to",
    )

    assert score == pytest.approx(11 / 16)


def test_hybrid_perfect_correlation_n2_4_k1():
    strategy = _load_strategy_by_spec(HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0))

    score = evaluate_strategy_exhaustive(
        strategy,
        p_rule=1.0,
        box_limit=1,
        subset_policy="up_to",
    )

    assert score == pytest.approx(0.75)


def test_pyramid_perfect_correlation_n2_4_k3():
    strategy = _load_strategy_by_spec(HybridStrategySpec("pyramid", n2=4, k_box=3))

    score = evaluate_strategy_exhaustive(
        strategy,
        p_rule=1.0,
        box_limit=3,
        subset_policy="up_to",
    )

    assert score == pytest.approx(1.0)


def test_hybrid_uses_box_only_where_useful():
    strategy = _load_strategy_by_spec(HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0))

    s0 = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=0)
    s1 = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=1)

    assert s0 < s1
    assert s1 == pytest.approx(0.75)


def test_pyramid_degrades_without_boxes():
    strategy = _load_strategy_by_spec(HybridStrategySpec("pyramid", n2=4, k_box=3))

    s0 = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=0)
    s3 = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=3)

    assert s0 < s3
    assert s3 == pytest.approx(1.0)


@pytest.mark.parametrize(
    "p_rule,expected",
    [
        (1.0, 0.75),
        (0.75, 0.6875),
        (0.5, 0.625),
    ],
)
def test_hybrid_horiz_maj_pyr_n2_4_k3_prefix_tie_logic(p_rule, expected):
    strategy = _load_strategy_by_spec(HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0))
    score = evaluate_strategy_exhaustive(
        strategy,
        p_rule=p_rule,
        box_limit=3,
        subset_policy="up_to",
    )
    assert score == pytest.approx(expected)


def test_hybrid_horiz_maj_pyr_n2_4_k3_boxlimit_three_equals_one():
    strategy = _load_strategy_by_spec(HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0))
    s1 = evaluate_strategy_exhaustive(strategy, p_rule=0.77, box_limit=1)
    s3 = evaluate_strategy_exhaustive(strategy, p_rule=0.77, box_limit=3)
    assert s3 == pytest.approx(s1)

