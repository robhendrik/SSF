from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.bob_mc import evaluate_strategy_mc
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


def _load_strategy_by_spec(spec: HybridStrategySpec) -> dict:
    entry = _entry_by_spec(spec)
    strategy = load_strategy(FIXTURE_DIR / entry["file"])
    assert int(strategy["n2"]) == int(entry["n2"])
    assert int(strategy["k_box"]) == int(entry["k_box"])
    return strategy


def test_mc_reproducible_for_same_seed():
    strategy = _load_strategy_by_spec(
        HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0)
    )

    r1 = evaluate_strategy_mc(
        strategy,
        p_rule=0.8,
        box_limit=1,
        seed=42,
        n_train_samples=10_000,
        n_eval_samples=20_000,
    )
    r2 = evaluate_strategy_mc(
        strategy,
        p_rule=0.8,
        box_limit=1,
        seed=42,
        n_train_samples=10_000,
        n_eval_samples=20_000,
    )

    assert r1 == r2


def test_mc_different_seed_stays_valid():
    strategy = _load_strategy_by_spec(
        HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0)
    )

    r3 = evaluate_strategy_mc(
        strategy,
        p_rule=0.8,
        box_limit=1,
        seed=43,
        n_train_samples=10_000,
        n_eval_samples=20_000,
    )

    assert 0.0 <= r3.success <= 1.0
    assert 0.0 <= r3.ci_low <= r3.success <= r3.ci_high <= 1.0
    assert r3.stderr >= 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_rule": -0.1},
        {"p_rule": 1.1},
        {"box_limit": -1},
        {"box_limit": 99},
        {"subset_policy": "bad"},
        {"n_train_samples": 0},
        {"n_eval_samples": 0},
    ],
)
def test_mc_validation_errors(kwargs):
    strategy = _load_strategy_by_spec(
        HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0)
    )

    base = {
        "strategy": strategy,
        "p_rule": 0.8,
        "box_limit": 1,
        "subset_policy": "up_to",
        "n_train_samples": 1000,
        "n_eval_samples": 2000,
        "seed": 7,
    }
    base.update(kwargs)

    with pytest.raises(ValueError):
        evaluate_strategy_mc(**base)


def test_mc_source_has_no_shortcuts():
    source_path = PROJECT_ROOT / "src" / "ssf" / "bob_mc.py"
    text = source_path.read_text(encoding="utf-8")

    forbidden = [
        "evaluate_strategy_exhaustive",
        "bob_exhaustive",
        "expected_scores",
        'strategy.get("name")',
        "majority",
        "hybrid",
        "pyramid",
        "return 0.625",
        "return 0.75",
        "return 1.0",
    ]

    for token in forbidden:
        assert token not in text

