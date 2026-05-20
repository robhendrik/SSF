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


@pytest.mark.parametrize(
    "spec,p_rule",
    [
        (HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0), 0.65),
        (HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0), 0.77),
        (HybridStrategySpec("pyramid", n2=4, k_box=3), 1.0),
    ],
    ids=lambda v: candidate_name(v) if isinstance(v, HybridStrategySpec) else str(v),
)
def test_exhaustive_k3_valid_probability_repeatable_and_monotonic(spec: HybridStrategySpec, p_rule):
    strategy = _load_strategy_by_spec(spec)

    s0 = evaluate_strategy_exhaustive(strategy, p_rule=p_rule, box_limit=0, subset_policy="up_to")
    s1 = evaluate_strategy_exhaustive(strategy, p_rule=p_rule, box_limit=1, subset_policy="up_to")
    s2 = evaluate_strategy_exhaustive(strategy, p_rule=p_rule, box_limit=2, subset_policy="up_to")
    s3 = evaluate_strategy_exhaustive(strategy, p_rule=p_rule, box_limit=3, subset_policy="up_to")
    s3_repeat = evaluate_strategy_exhaustive(strategy, p_rule=p_rule, box_limit=3, subset_policy="up_to")

    assert 0.0 <= s0 <= 1.0
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0
    assert 0.0 <= s3 <= 1.0

    assert s0 <= s1 + 1e-12
    assert s1 <= s2 + 1e-12
    assert s2 <= s3 + 1e-12

    assert s3 == pytest.approx(s3_repeat)


