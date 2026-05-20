from __future__ import annotations

from collections import defaultdict
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


def slow_oracle(
    strategy: dict,
    p_rule: float,
    box_limit: int | None,
    subset_policy: str = "up_to",
) -> float:
    n2 = int(strategy["n2"])
    k_box = int(strategy["k_box"])
    f_tables = strategy["f_tables"]
    comm = strategy["comm"]

    eff_limit = k_box if box_limit is None else int(box_limit)

    if subset_policy == "up_to":
        allowed_subsets = [
            mask for mask in range(1 << k_box) if mask.bit_count() <= eff_limit
        ]
    elif subset_policy == "exact":
        allowed_subsets = [
            mask for mask in range(1 << k_box) if mask.bit_count() == eff_limit
        ]
    else:
        raise ValueError("subset_policy must be 'up_to' or 'exact'.")

    total = 0.0

    for gun in range(n2):
        best_for_gun = 0.0

        for subset_mask in allowed_subsets:
            active = [j for j in range(k_box) if subset_mask & (1 << j)]
            r = len(active)

            for meas_mask in range(1 << r):
                mass = defaultdict(lambda: [0.0, 0.0])

                for field in range(1 << n2):
                    target = (field >> gun) & 1

                    for out_a in range(1 << k_box):
                        comm_bit = int(comm[field, out_a])
                        base_weight = 1.0 / ((1 << n2) * (1 << k_box))

                        m_a = []
                        for j in range(k_box):
                            prefix = 0 if j == 0 else out_a & ((1 << j) - 1)
                            m_a.append(int(f_tables[j][field, prefix]))

                        for out_b_mask in range(1 << r):
                            weight = base_weight
                            visible_bits = []

                            for t, box_idx in enumerate(active):
                                out_a_j = (out_a >> box_idx) & 1
                                out_b_j = (out_b_mask >> t) & 1
                                m_b_j = (meas_mask >> t) & 1
                                correct = out_a_j ^ (m_a[box_idx] & m_b_j)

                                if out_b_j == correct:
                                    weight *= p_rule
                                else:
                                    weight *= (1.0 - p_rule)
                                visible_bits.append(out_b_j)

                            visible = (comm_bit, tuple(visible_bits))
                            mass[visible][target] += weight

                success = sum(max(v[0], v[1]) for v in mass.values())
                best_for_gun = max(best_for_gun, success)

        total += best_for_gun

    return total / n2


@pytest.mark.parametrize(
    "spec",
    [
        HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0),
        HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0),
        HybridStrategySpec("pyramid", n2=4, k_box=3),
    ],
    ids=lambda s: candidate_name(s),
)
@pytest.mark.parametrize("p_rule", [0.5, 0.75, 1.0])
@pytest.mark.parametrize("box_limit", [0, 1, 2, 3])
def test_exhaustive_matches_independent_slow_oracle(spec: HybridStrategySpec, p_rule, box_limit):
    strategy = _load_strategy_by_spec(spec)

    fast = evaluate_strategy_exhaustive(strategy, p_rule=p_rule, box_limit=box_limit)
    slow = slow_oracle(strategy, p_rule=p_rule, box_limit=box_limit)

    assert fast == pytest.approx(slow)


