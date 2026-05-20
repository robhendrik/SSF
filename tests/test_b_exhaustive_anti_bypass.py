from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.bob_exhaustive import evaluate_strategy_exhaustive
from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name
from ssf.strategy_io import load_manifest, load_strategy


FIXTURE_DIR = PROJECT_ROOT / "fixtures" / "persistent" / "a_strategies"


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


def test_evaluator_ignores_strategy_name():
    strategy = _load_strategy_by_spec(HybridStrategySpec("pyramid", n2=4, k_box=3))
    score_original = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=3)

    strategy_renamed = copy.deepcopy(strategy)
    strategy_renamed["name"] = "majority"
    strategy_renamed["expected_scores"] = {"1.0": -999.0}
    score_renamed = evaluate_strategy_exhaustive(strategy_renamed, p_rule=1.0, box_limit=3)

    assert score_renamed == pytest.approx(score_original)
    assert score_renamed == pytest.approx(1.0)


def test_evaluator_depends_on_comm_table():
    strategy = _load_strategy_by_spec(HybridStrategySpec("pyramid", n2=4, k_box=3))
    baseline = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=3)

    broken = copy.deepcopy(strategy)
    broken["comm"] = np.zeros_like(broken["comm"], dtype=np.uint8)
    broken_score = evaluate_strategy_exhaustive(broken, p_rule=1.0, box_limit=3)

    assert baseline == pytest.approx(1.0)
    assert broken_score < baseline - 1e-9
    assert broken_score < 1.0


def test_evaluator_depends_on_f_tables():
    strategy = _load_strategy_by_spec(HybridStrategySpec("pyramid", n2=4, k_box=3))
    baseline = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=3)

    broken = copy.deepcopy(strategy)
    broken["f_tables"] = [
        np.zeros_like(table, dtype=np.uint8)
        for table in broken["f_tables"]
    ]
    broken_score = evaluate_strategy_exhaustive(broken, p_rule=1.0, box_limit=3)

    assert baseline == pytest.approx(1.0)
    assert broken_score < baseline - 1e-9
    assert broken_score < 1.0


def test_p_rule_affects_box_dependent_strategy():
    strategy = _load_strategy_by_spec(HybridStrategySpec("pyramid", n2=4, k_box=3))

    s_random = evaluate_strategy_exhaustive(strategy, p_rule=0.5, box_limit=3)
    s_perfect = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=3)

    assert s_perfect == pytest.approx(1.0)
    assert s_random < s_perfect - 1e-9


def test_intermediate_p_rule_ordering():
    majority = _load_strategy_by_spec(HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0))
    hybrid = _load_strategy_by_spec(HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0))
    pyramid = _load_strategy_by_spec(HybridStrategySpec("pyramid", n2=4, k_box=3))

    majority_score = evaluate_strategy_exhaustive(majority, p_rule=0.77, box_limit=3)
    hybrid_score = evaluate_strategy_exhaustive(hybrid, p_rule=0.77, box_limit=3)
    pyramid_score = evaluate_strategy_exhaustive(pyramid, p_rule=0.77, box_limit=3)

    assert 0.0 <= majority_score <= 1.0
    assert 0.0 <= hybrid_score <= 1.0
    assert 0.0 <= pyramid_score <= 1.0
    
    scores = {
        round(majority_score, 12),
        round(hybrid_score, 12),
        round(pyramid_score, 12),
    }
    assert len(scores) > 1


def test_box_limit_actually_restricts_available_information():
    strategy = _load_strategy_by_spec(HybridStrategySpec("pyramid", n2=4, k_box=3))

    s0 = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=0)
    s1 = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=1)
    s2 = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=2)
    s3 = evaluate_strategy_exhaustive(strategy, p_rule=1.0, box_limit=3)

    assert s0 < s3
    assert s1 < s3
    assert s2 <= s3 # pyramid uses log2(n2) boxes, in this case 2, so s2 may be equal to s3 if the box limit is not actually binding for this strategy, but it should never be better than s3
    assert s3 == pytest.approx(1.0)

