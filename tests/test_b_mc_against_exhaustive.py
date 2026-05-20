from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.bob_exhaustive import evaluate_strategy_exhaustive
from ssf.bob_mc import evaluate_strategy_mc
from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name
from ssf.strategy_io import load_manifest, load_strategy

pytestmark = pytest.mark.slow

FIXTURE_DIR = PROJECT_ROOT / "fixtures" / "persistent" / "a_strategies"

SELECTED_SPECS = [
    HybridStrategySpec("majority", n2=8, k_box=1, tie_bandwidth=0),
    HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0),
    HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0),
    HybridStrategySpec("pyramid", n2=4, k_box=3),
]


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
    assert entry is not None, f"Missing fixture for spec name: {expected_name}"
    assert _spec_from_manifest_entry(entry) == spec
    assert entry["name"] == expected_name
    assert entry["file"] == expected_name + ".npz"
    return entry


def _load_strategy_by_spec(spec: HybridStrategySpec) -> dict:
    entry = _entry_by_spec(spec)
    strategy = load_strategy(FIXTURE_DIR / entry["file"])
    assert int(strategy["n2"]) == int(entry["n2"])
    assert int(strategy["k_box"]) == int(entry["k_box"])
    return strategy


def test_no_stale_split_literal_in_source() -> None:
    text = Path(__file__).read_text(encoding="utf-8")
    stale = "split_2_" + "b_0"
    assert stale not in text


@pytest.mark.parametrize("spec", SELECTED_SPECS, ids=lambda s: candidate_name(s))
@pytest.mark.parametrize("p_rule", [1.0, 0.8, 0.5])
@pytest.mark.parametrize("box_limit", [0, 1, 3])
def test_b_mc_tracks_exhaustive(spec: HybridStrategySpec, p_rule: float, box_limit: int):
    strategy = _load_strategy_by_spec(spec)

    if box_limit > strategy["k_box"]:
        pytest.skip("box_limit exceeds k_box")

    exact = evaluate_strategy_exhaustive(
        strategy,
        p_rule=p_rule,
        box_limit=box_limit,
        subset_policy="up_to",
    )

    mc = evaluate_strategy_mc(
        strategy,
        p_rule=p_rule,
        box_limit=box_limit,
        subset_policy="up_to",
        n_train_samples=30_000,
        n_eval_samples=50_000,
        seed=123,
    )

    assert mc.ci_low <= mc.success <= mc.ci_high
    assert abs(mc.success - exact) < 0.02

