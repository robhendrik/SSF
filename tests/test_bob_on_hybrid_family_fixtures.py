"""
Tests validating bob_exhaustive and bob_mc against A-side hybrid-family fixtures.

These tests load fixtures from fixtures/persistent/a_strategies/ and verify that
Bob's optimal response success rates match the analytical_success values stored in
manifest.json.

Note: If optimal Bob outperforms the named analytical decoder, the test failure is
useful and reflects an opportunity to improve the strategy design, not a test bug.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



# Import bob evaluators from ssf package
try:
    from ssf.bob_exhaustive import evaluate_strategy_exhaustive
except ImportError as e:
    pytest.fail(f"Cannot import ssf.bob_exhaustive: {e}")

try:
    from ssf.bob_mc import evaluate_strategy_mc
except ImportError as e:
    pytest.fail(f"Cannot import ssf.bob_mc: {e}")

from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name
from ssf.strategy_io import load_strategy

FIXTURE_DIR = PROJECT_ROOT / "fixtures" / "persistent" / "a_strategies"
MANIFEST_FILE = FIXTURE_DIR / "manifest.json"

# Selected fixtures for exhaustive testing (subset of available)
SELECTED_FIXTURE_SPECS_EXHAUSTIVE = [
    HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0),
    HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="pyr_maj", tie_bandwidth=0),
    HybridStrategySpec("vertical", n2=4, k_box=3, depth_s=1, order="maj_pyr", tie_bandwidth=0),
    HybridStrategySpec("majority", n2=8, k_box=1, tie_bandwidth=0),
]

# Test p_rule values
P_RULE_VALUES = [0.5, 0.77, 1.0]


def _load_manifest() -> list[dict]:
    """Load manifest.json, fail if missing."""
    if not MANIFEST_FILE.exists():
        pytest.fail(f"Manifest file missing: {MANIFEST_FILE}")
    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    if not isinstance(manifest, list):
        pytest.fail(f"Manifest is not a list: {type(manifest)}")
    return manifest


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


def _entry_by_spec(manifest: list[dict], spec: HybridStrategySpec) -> dict:
    expected_name = candidate_name(spec)
    entry = next((e for e in manifest if e.get("name") == expected_name), None)
    if entry is None:
        pytest.fail(f"Fixture {expected_name} not found in manifest")
    assert _spec_from_manifest_entry(entry) == spec
    assert entry["file"] == expected_name + ".npz"
    return entry


def _load_strategy(entry: dict) -> dict:
    fixture_path = FIXTURE_DIR / entry["file"]
    if not fixture_path.exists():
        pytest.fail(f"Fixture file missing: {fixture_path}")
    strategy = load_strategy(fixture_path)
    assert int(strategy["n2"]) == int(entry["n2"])
    assert int(strategy["k_box"]) == int(entry["k_box"])
    return strategy


def test_manifest_exists():
    """Verify manifest.json exists and is readable."""
    assert MANIFEST_FILE.exists(), f"Manifest file missing: {MANIFEST_FILE}"
    manifest = _load_manifest()
    assert len(manifest) > 0, "Manifest is empty"


def test_all_selected_fixtures_exist():
    """Verify all selected test fixtures are present in manifest."""
    manifest = _load_manifest()
    manifest_names = {e["name"] for e in manifest}
    for fixture_name in [candidate_name(spec) for spec in SELECTED_FIXTURE_SPECS_EXHAUSTIVE]:
        assert fixture_name in manifest_names, (
            f"Fixture {fixture_name} not found in manifest. "
            f"Available: {sorted(manifest_names)}"
        )


def test_all_fixture_files_exist():
    """Verify all selected fixture files exist on disk."""
    for fixture_name in [candidate_name(spec) for spec in SELECTED_FIXTURE_SPECS_EXHAUSTIVE]:
        fixture_path = FIXTURE_DIR / f"{fixture_name}.npz"
        assert fixture_path.exists(), f"Fixture file missing: {fixture_path}"


@pytest.mark.parametrize("spec", SELECTED_FIXTURE_SPECS_EXHAUSTIVE, ids=lambda s: candidate_name(s))
@pytest.mark.parametrize("p_rule", P_RULE_VALUES)
def test_bob_exhaustive_matches_analytical(spec: HybridStrategySpec, p_rule: float):
    """
    Test that bob_exhaustive result matches analytical_success from manifest.

    Loads the A-side fixture and runs evaluate_strategy_exhaustive to find optimal
    Bob response. Compares to the analytical_success value stored in manifest.json.

    If this fails: either the fixture was generated with a wrong formula, or the
    optimal Bob can outperform the named analytical decoder (which is useful feedback).
    """
    manifest = _load_manifest()
    fixture_name = candidate_name(spec)
    entry = _entry_by_spec(manifest, spec)

    # For approximate formulas, only test at exact boundary values (0.5, 1.0)
    formula_notes = entry.get("analytical_notes", "")
    if "Approximate" in formula_notes:
        if p_rule not in [0.5, 1.0]:
            pytest.skip(
                f"Analytical formula marked Approximate; "
                f"skipping p_rule={p_rule} (test only at 0.5 and 1.0)"
            )

    # Get expected value from manifest
    p_rule_key = f"{p_rule:.2f}"
    if p_rule_key not in entry.get("analytical_success", {}):
        pytest.skip(f"p_rule={p_rule} not in analytical_success for {fixture_name}")

    expected = entry["analytical_success"][p_rule_key]

    # Load and evaluate strategy
    strategy = _load_strategy(entry)
    actual = evaluate_strategy_exhaustive(
        strategy,
        p_rule=p_rule,
        box_limit=entry["required_boxes"],
        subset_policy="up_to",
    )

    # Verify match
    assert actual == pytest.approx(expected, abs=1e-9), (
        f"Success rate mismatch for {fixture_name} at p_rule={p_rule}: "
        f"expected {expected}, got {actual} (diff: {abs(expected - actual):.2e})"
    )


def test_bob_mc_within_confidence_interval():
    """
    Test that MC evaluation returns expected value within reported confidence interval.

    Runs bob_mc.evaluate_strategy_mc and verifies that the analytical_success value
    falls within the returned confidence interval at the specified confidence level.
    """
    manifest = _load_manifest()

    spec = HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0)
    fixture_name = candidate_name(spec)
    p_rule = 0.77

    entry = _entry_by_spec(manifest, spec)

    # Get expected value
    p_rule_key = f"{p_rule:.2f}"
    expected = entry["analytical_success"][p_rule_key]

    # Load and evaluate strategy
    strategy = _load_strategy(entry)
    result = evaluate_strategy_mc(
        strategy,
        p_rule=p_rule,
        box_limit=entry["required_boxes"],
        subset_policy="up_to",
        n_train_samples=50000,
        n_eval_samples=100000,
        confidence=0.99,
        seed=123,
    )

    # Verify expected is within CI
    assert result.ci_low <= expected <= result.ci_high, (
        f"Expected value {expected} outside 99% confidence interval "
        f"[{result.ci_low:.6f}, {result.ci_high:.6f}] "
        f"(point estimate: {result.success:.6f})"
    )


def test_bob_mc_result_sanity():
    """
    Smoke test for MC evaluation result structure and consistency.

    Verifies that:
    - Success rate is in [0, 1]
    - Point estimate is within its own confidence interval
    - Standard error is positive
    """
    manifest = _load_manifest()

    spec = HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0)
    fixture_name = candidate_name(spec)
    p_rule = 0.77

    entry = _entry_by_spec(manifest, spec)

    # Load and evaluate strategy
    strategy = _load_strategy(entry)
    result = evaluate_strategy_mc(
        strategy,
        p_rule=p_rule,
        box_limit=entry["required_boxes"],
        subset_policy="up_to",
        n_train_samples=50000,
        n_eval_samples=100000,
        confidence=0.99,
        seed=123,
    )

    # Sanity checks
    assert 0 <= result.success <= 1, (
        f"Success rate outside [0, 1]: {result.success}"
    )
    assert result.ci_low <= result.success <= result.ci_high, (
        f"Point estimate {result.success:.6f} outside its own CI "
        f"[{result.ci_low:.6f}, {result.ci_high:.6f}]"
    )
    assert result.stderr > 0, (
        f"Standard error should be positive, got {result.stderr}"
    )
    assert result.ci_high > result.ci_low, (
        f"Invalid interval: ci_high={result.ci_high} <= ci_low={result.ci_low}"
    )

