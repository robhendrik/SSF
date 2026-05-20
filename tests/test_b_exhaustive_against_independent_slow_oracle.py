from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.bob_exhaustive import evaluate_strategy_exhaustive
from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name
from ssf.strategy_io import load_manifest, load_strategy

FIXTURE_DIR = PROJECT_ROOT / "fixtures" / "persistent" / "a_strategies"

SELECTED_SPECS = [
    HybridStrategySpec("majority", n2=8, k_box=1, tie_bandwidth=0),
    HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0),
    HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0),
    HybridStrategySpec("pyramid", n2=4, k_box=3),
]

P_RULES = [1.0, 0.8, 0.5]
BOX_LIMITS = [0, 1, 2, 3]

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


def slow_oracle(strategy: dict, p_rule: float, box_limit: int | None, subset_policy: str = "up_to") -> float:
    n2 = int(strategy["n2"])
    k_box = int(strategy["k_box"])
    f_tables = strategy["f_tables"]
    comm = strategy["comm"]

    if box_limit is None:
        eff_limit = k_box
    else:
        eff_limit = int(box_limit)

    subset_masks = []
    for mask in range(1 << k_box):
        bits = mask.bit_count()
        if subset_policy == "up_to":
            if bits <= eff_limit:
                subset_masks.append(mask)
        elif subset_policy == "exact":
            if bits == eff_limit:
                subset_masks.append(mask)
        else:
            raise ValueError("bad subset_policy")

    n_fields = 1 << n2
    outa_count = 1 << k_box
    base_weight = 1.0 / (n_fields * outa_count)

    total = 0.0

    for gun in range(n2):
        best_for_gun = 0.0

        for subset_mask in subset_masks:
            active = [j for j in range(k_box) if (subset_mask >> j) & 1]
            r = len(active)

            for meas_mask in range(1 << r):
                masses = {}

                for field in range(n_fields):
                    target = (field >> gun) & 1

                    for out_a in range(outa_count):
                        comm_bit = int(comm[field, out_a])

                        m_a_vals = []
                        for j in range(k_box):
                            prefix = 0 if j == 0 else (out_a & ((1 << j) - 1))
                            m_a_vals.append(int(f_tables[j][field, prefix]))

                        for out_b_mask in range(1 << r):
                            weight = base_weight
                            visible_bits = []

                            for t, box_idx in enumerate(active):
                                out_a_j = (out_a >> box_idx) & 1
                                out_b_j = (out_b_mask >> t) & 1
                                m_b_j = (meas_mask >> t) & 1

                                correct_b = out_a_j ^ (m_a_vals[box_idx] & m_b_j)

                                if out_b_j == correct_b:
                                    weight *= p_rule
                                else:
                                    weight *= 1.0 - p_rule

                                visible_bits.append(out_b_j)

                            visible = (comm_bit, tuple(visible_bits))
                            if visible not in masses:
                                masses[visible] = [0.0, 0.0]

                            masses[visible][target] += weight

                success = sum(max(v[0], v[1]) for v in masses.values())
                if success > best_for_gun:
                    best_for_gun = success

        total += best_for_gun

    return total / n2


@pytest.mark.parametrize("spec", SELECTED_SPECS, ids=lambda s: candidate_name(s))
@pytest.mark.parametrize("p_rule", P_RULES)
@pytest.mark.parametrize("box_limit", BOX_LIMITS)
def test_b_exhaustive_matches_independent_slow_oracle(spec: HybridStrategySpec, p_rule: float, box_limit: int):
    strategy = _load_strategy_by_spec(spec)

    if box_limit > strategy["k_box"]:
        pytest.skip("box_limit exceeds k_box")

    fast = evaluate_strategy_exhaustive(
        strategy,
        p_rule=p_rule,
        box_limit=box_limit,
        subset_policy="up_to",
    )

    slow = slow_oracle(
        strategy,
        p_rule=p_rule,
        box_limit=box_limit,
        subset_policy="up_to",
    )

    assert fast == pytest.approx(slow, abs=1e-12)

