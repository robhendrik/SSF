"""
Tests for bob_mc_procedural.evaluate_strategy_mc_procedural.

Test strategy classes are defined in this file:
  - DenseAdapterAliceStrategy: wraps a dense strategy dict for protocol compliance
  - PurePyramidProceduralAliceStrategy: computes pyramid A-side on demand

Tests:
  A. Small dense equivalence (DenseAdapter vs bob_mc)
  B. Procedural pure pyramid known-answer test, n2=4
  C. Procedural pure pyramid medium test, n2=8
  D. Large-scale smoke/benchmark, n2=16 (marked slow)
  E. Input validation
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# SSF/tests is one level below SSF root
SSF_DIR = Path(__file__).resolve().parents[1]
SSF_SRC_DIR = SSF_DIR / "src"
if str(SSF_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SSF_SRC_DIR))

from ssf.bob_mc import evaluate_strategy_mc
from ssf.bob_mc_procedural import (
    ProceduralBobResult,
    evaluate_strategy_mc_procedural,
)

FIXTURE_DIR = SSF_DIR / "fixtures" / "persistent" / "a_strategies"
MANIFEST_FILE = FIXTURE_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# Test strategy helpers
# ---------------------------------------------------------------------------

class DenseAdapterAliceStrategy:
    """Wraps a dense strategy dict to implement ProceduralAliceStrategy."""

    def __init__(self, strategy_dict: dict):
        self.n2: int = int(strategy_dict["n2"])
        self.k_box: int = int(strategy_dict["k_box"])
        self._f_tables: list[np.ndarray] = list(strategy_dict["f_tables"])
        comm = strategy_dict.get("comm")
        if comm is None:
            comm = strategy_dict.get("comm_table")
        self._comm: np.ndarray = np.asarray(comm, dtype=np.uint8)

    def measure_a(self, box_idx: int, field: int, out_a_prefix: int) -> int:
        return int(self._f_tables[box_idx][field, out_a_prefix])

    def comm(self, field: int, out_a: int) -> int:
        return int(self._comm[field, out_a])


class PurePyramidProceduralAliceStrategy:
    """
    Full dense-free pure XOR pyramid A-side strategy.

    Box ordering (layer 0 first, left to right):
        n2=4, k_box=3:
            box0: inputs (field[0], field[1])   layer 0, pos 0
            box1: inputs (field[2], field[3])   layer 0, pos 1
            box2: top box                       layer 1, pos 0

        n2=8, k_box=7:
            box0..box3: layer 0 (leaf pairs)
            box4: layer 1, pos 0 (combines boxes 0, 1)
            box5: layer 1, pos 1 (combines boxes 2, 3)
            box6: layer 2, pos 0 (apex)

    Key formula — left_output(layer, pos):
        The value propagated UP the LEFT branch out of box (layer, pos).
        left_output(0, p)   = field_bit(2p) XOR a_{box(0,p)}
        left_output(l, p)   = left_output(l-1, 2p) XOR a_{box(l,p)}

    measure_a(box_idx, field, prefix):
        At layer 0, pos p: field_bit(2p) XOR field_bit(2p+1)  [raw leaf XOR]
        At layer l, pos p: left_output(l-1, 2p) XOR left_output(l-1, 2p+1)

    comm(field, out_a):
        left_output(depth-1, 0, field, out_a)
        = field_bit(0) XOR a_{box on left spine of layer 0}
                        XOR a_{box on left spine of layer 1}
                        XOR ... XOR a_{apex}

    For n2=4:  comm = (field[0] XOR a0) XOR a2
    For n2=8:  comm = (field[0] XOR a0) XOR a4 XOR a6
    """

    def __init__(self, n2: int, k_box: int):
        if n2 <= 0 or (n2 & (n2 - 1)) != 0:
            raise ValueError("n2 must be a positive power of 2.")
        depth = int(math.log2(n2))
        if k_box != n2 - 1:
            raise ValueError(f"Full pyramid requires k_box == n2-1, got k_box={k_box}")
        self.n2: int = n2
        self.k_box: int = k_box
        self._depth: int = depth

        # Precompute: box_idx -> (layer, pos)
        self._box_layer: list[int] = []
        self._box_pos: list[int] = []
        for layer in range(depth):
            n_boxes_layer = n2 >> (layer + 1)
            for pos in range(n_boxes_layer):
                self._box_layer.append(layer)
                self._box_pos.append(pos)

        # Precompute: (layer, pos) -> box_idx
        self._box_idx_map: dict[tuple[int, int], int] = {}
        for b, (lay, pos) in enumerate(zip(self._box_layer, self._box_pos)):
            self._box_idx_map[(lay, pos)] = b

    def _left_output(self, layer: int, pos: int, field: int, out_a: int) -> int:
        """Compute the left-branch propagated output of box (layer, pos).

        This is the value passed as LEFT input to the parent box at (layer+1, pos//2).

            left_output(0, p) = field_bit(2p) XOR a_{box(0,p)}
            left_output(l, p) = left_output(l-1, 2p) XOR a_{box(l,p)}
        """
        box = self._box_idx_map[(layer, pos)]
        a_box = (out_a >> box) & 1
        if layer == 0:
            left_leaf = (field >> (2 * pos)) & 1
            return left_leaf ^ a_box
        else:
            return self._left_output(layer - 1, 2 * pos, field, out_a) ^ a_box

    def measure_a(self, box_idx: int, field: int, out_a_prefix: int) -> int:
        """Return the measurement bit Alice uses for this PR box.

        At layer 0 (leaf boxes): XOR of the two adjacent field bits.
        At layer l (internal boxes): XOR of the two left-branch outputs from
        the child boxes (which are computed from field bits and prior out_a).
        """
        layer = self._box_layer[box_idx]
        pos = self._box_pos[box_idx]

        if layer == 0:
            return ((field >> (2 * pos)) & 1) ^ ((field >> (2 * pos + 1)) & 1)
        else:
            left_val = self._left_output(layer - 1, 2 * pos, field, out_a_prefix)
            right_val = self._left_output(layer - 1, 2 * pos + 1, field, out_a_prefix)
            return left_val ^ right_val

    def comm(self, field: int, out_a: int) -> int:
        """Return the communication bit Alice sends to Bob.

        This is the left-branch output of the apex box, which equals:
            field_bit(0) XOR a0 XOR a_{left_box_layer_1} XOR ... XOR a_apex
        """
        apex_layer = self._depth - 1
        return self._left_output(apex_layer, 0, field, out_a)


# ---------------------------------------------------------------------------
# Fixture loading helpers
# ---------------------------------------------------------------------------

def _load_manifest() -> list[dict]:
    if not MANIFEST_FILE.exists():
        pytest.fail(f"Manifest missing: {MANIFEST_FILE}")
    with open(MANIFEST_FILE) as f:
        return json.load(f)


def _load_strategy_dict(fixture_name: str) -> dict:
    path = FIXTURE_DIR / f"{fixture_name}.npz"
    if not path.exists():
        pytest.fail(f"Fixture file missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        n2 = int(data["n2"])
        k_box = int(data["k_box"])
        f_tables = [data[f"f_table_{d}"] for d in range(k_box)]
        comm = data["comm_table"]
    return {"n2": n2, "k_box": k_box, "f_tables": f_tables, "comm": comm}


def _find_entry(manifest: list[dict], name: str) -> dict:
    for e in manifest:
        if e.get("name") == name:
            return e
    pytest.fail(f"Entry {name!r} not found in manifest")


# ---------------------------------------------------------------------------
# A. Dense adapter equivalence tests
# ---------------------------------------------------------------------------

_DENSE_EQUIV_FIXTURES = [
    "hybrid_horiz_maj_pyr_n2_4_k_3_split_2_2_b_0",
    "hybrid_vert_maj_pyr_n2_4_k_3_s_1_b_0",
]

@pytest.mark.parametrize("fixture_name", _DENSE_EQUIV_FIXTURES)
def test_dense_adapter_matches_bob_mc(fixture_name: str):
    """
    Procedural evaluator via DenseAdapter should produce CIs that overlap with
    bob_mc.evaluate_strategy_mc, and point estimates within 3 combined stderr + 0.01.
    """
    manifest = _load_manifest()
    entry = _find_entry(manifest, fixture_name)
    strategy_dict = _load_strategy_dict(fixture_name)
    adapter = DenseAdapterAliceStrategy(strategy_dict)

    kwargs = dict(
        p_rule=0.77,
        box_limit=entry.get("required_boxes", entry.get("boxes_used")),
        subset_policy="up_to",
        n_train_samples=10_000,
        n_eval_samples=20_000,
        confidence=0.99,
        seed=123,
    )

    mc_result = evaluate_strategy_mc(strategy_dict, **kwargs)
    proc_result = evaluate_strategy_mc_procedural(adapter, **kwargs)

    # CIs must overlap
    ci_overlap = proc_result.ci_low <= mc_result.ci_high and mc_result.ci_low <= proc_result.ci_high
    assert ci_overlap, (
        f"[{fixture_name}] CIs do not overlap: "
        f"mc=[{mc_result.ci_low:.5f},{mc_result.ci_high:.5f}] "
        f"proc=[{proc_result.ci_low:.5f},{proc_result.ci_high:.5f}]"
    )

    # Point estimates within 3 combined stderr + 0.01
    combined_stderr = mc_result.stderr + proc_result.stderr
    diff = abs(mc_result.success - proc_result.success)
    assert diff <= 3 * combined_stderr + 0.01, (
        f"[{fixture_name}] Point estimates differ too much: "
        f"mc={mc_result.success:.5f}, proc={proc_result.success:.5f}, "
        f"diff={diff:.5f}, 3*combined_stderr+0.01={3*combined_stderr+0.01:.5f}"
    )


# ---------------------------------------------------------------------------
# B. Pure pyramid n2=4 known-answer test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_rule", [0.5, 0.77, 1.0])
def test_pure_pyramid_n2_4_known_answer(p_rule: float):
    """
    Full pyramid n2=4 should match analytical formula:
        expected = (1 + (2*p_rule - 1)**log2(n2)) / 2
    """
    n2 = 4
    depth = int(math.log2(n2))
    expected = (1.0 + (2.0 * p_rule - 1.0) ** depth) / 2.0

    strategy = PurePyramidProceduralAliceStrategy(n2=n2, k_box=n2 - 1)
    # Full pyramid has k_box=n2-1 boxes, but Bob's optimal named decoder uses
    # log2(n2) boxes on the active path. Searching all subsets up to k_box is
    # not intended for this scalability test.
    box_limit = depth

    result = evaluate_strategy_mc_procedural(
        strategy,
        p_rule=p_rule,
        box_limit=box_limit,
        subset_policy="up_to",
        n_train_samples=10_000,
        n_eval_samples=20_000,
        confidence=0.99,
        seed=123,
    )

    if p_rule == 0.5:
        # At p_rule=0.5 every strategy has true performance exactly 0.5.
        # The empirical majority-vote estimator is biased upward by
        # ~O(1/sqrt(n_samples_per_visible_state)), so the MC always reports
        # slightly above 0.5 and the CI will not cover 0.5. Accept any
        # result in [expected, expected + 0.03] instead of a CI coverage check.
        assert expected <= result.success <= expected + 0.03, (
            f"n2=4, p_rule=0.5: expected success in [{expected:.4f}, "
            f"{expected + 0.03:.4f}], got {result.success:.6f}"
        )
    else:
        assert result.ci_low <= expected <= result.ci_high, (
            f"n2=4, p_rule={p_rule}: expected={expected:.6f} outside "
            f"99% CI [{result.ci_low:.6f}, {result.ci_high:.6f}] "
            f"(success={result.success:.6f})"
        )

    if p_rule == 1.0:
        assert result.success > 0.99, (
            f"n2=4, p_rule=1.0: expected near-perfect success, got {result.success:.6f}"
        )


# ---------------------------------------------------------------------------
# C. Pure pyramid n2=8 medium test
# ---------------------------------------------------------------------------

def test_pure_pyramid_n2_8_known_answer():
    """
    Full pyramid n2=8 should match analytical formula at p_rule=0.77.
        depth = 3
        expected = (1 + (2*0.77 - 1)**3) / 2
    """
    n2 = 8
    p_rule = 0.77
    depth = int(math.log2(n2))
    expected = (1.0 + (2.0 * p_rule - 1.0) ** depth) / 2.0

    strategy = PurePyramidProceduralAliceStrategy(n2=n2, k_box=n2 - 1)
    box_limit = depth

    result = evaluate_strategy_mc_procedural(
        strategy,
        p_rule=p_rule,
        box_limit=box_limit,
        subset_policy="up_to",
        n_train_samples=10_000,
        n_eval_samples=20_000,
        confidence=0.99,
        seed=123,
    )

    assert result.ci_low <= expected <= result.ci_high, (
        f"n2=8, p_rule={p_rule}: expected={expected:.6f} outside "
        f"99% CI [{result.ci_low:.6f}, {result.ci_high:.6f}] "
        f"(success={result.success:.6f})"
    )


# # ---------------------------------------------------------------------------
# # D. Large-scale smoke/benchmark, n2=16 (slow)
# # ---------------------------------------------------------------------------

# def test_pure_pyramid_n2_16_benchmark(capsys):
#     """
#     Large-scale smoke test for n2=16, k_box=15.
#     Verifies basic sanity and expected CI coverage.
#     Prints benchmark info via capsys.
#     """
#     n2 = 16
#     p_rule = 0.77
#     depth = int(math.log2(n2))
#     expected = (1.0 + (2.0 * p_rule - 1.0) ** depth) / 2.0

#     strategy = PurePyramidProceduralAliceStrategy(n2=n2, k_box=n2 - 1)
#     box_limit = depth

#     result = evaluate_strategy_mc_procedural(
#         strategy,
#         p_rule=p_rule,
#         box_limit=box_limit,
#         subset_policy="up_to",
#         n_train_samples=20_000,
#         n_eval_samples=50_000,
#         confidence=0.99,
#         seed=123,
#         record_time=True,
#     )

#     with capsys.disabled():
#         print(
#             f"\n[benchmark n2=16, k_box=15, p_rule={p_rule}] "
#             f"success={result.success:.6f} "
#             f"CI=[{result.ci_low:.6f},{result.ci_high:.6f}] "
#             f"expected={expected:.6f} "
#             f"elapsed={result.elapsed_seconds:.2f}s"
#         )

#     # Sanity checks
#     assert 0 <= result.success <= 1, f"success out of range: {result.success}"
#     assert result.ci_low <= result.success <= result.ci_high, (
#         f"Point estimate {result.success:.6f} outside its own CI "
#         f"[{result.ci_low:.6f}, {result.ci_high:.6f}]"
#     )
#     assert result.stderr > 0, f"stderr should be positive, got {result.stderr}"

#     # CI coverage of analytical expected
#     assert result.ci_low <= expected <= result.ci_high, (
#         f"n2=16, p_rule={p_rule}: expected={expected:.6f} outside "
#         f"99% CI [{result.ci_low:.6f}, {result.ci_high:.6f}]"
#     )

#     # Timing recorded
#     assert result.elapsed_seconds is not None, "elapsed_seconds should be set when record_time=True"
#     assert result.elapsed_seconds > 0, f"elapsed_seconds should be positive, got {result.elapsed_seconds}"


# ---------------------------------------------------------------------------
# E. Input validation tests
# ---------------------------------------------------------------------------

class _MinimalStrategy:
    """Minimal valid strategy for validation error testing."""
    n2 = 4
    k_box = 1

    def measure_a(self, box_idx: int, field: int, out_a_prefix: int) -> int:
        return 0

    def comm(self, field: int, out_a: int) -> int:
        return 0


def test_validation_invalid_p_rule_low():
    strat = _MinimalStrategy()
    with pytest.raises(ValueError, match="p_rule"):
        evaluate_strategy_mc_procedural(strat, p_rule=-0.1)


def test_validation_invalid_p_rule_high():
    strat = _MinimalStrategy()
    with pytest.raises(ValueError, match="p_rule"):
        evaluate_strategy_mc_procedural(strat, p_rule=1.1)


def test_validation_invalid_subset_policy():
    strat = _MinimalStrategy()
    with pytest.raises(ValueError, match="subset_policy"):
        evaluate_strategy_mc_procedural(strat, p_rule=0.5, subset_policy="bad")


def test_validation_invalid_box_limit_negative():
    strat = _MinimalStrategy()
    with pytest.raises(ValueError, match="box_limit"):
        evaluate_strategy_mc_procedural(strat, p_rule=0.5, box_limit=-1)


def test_validation_invalid_box_limit_too_large():
    strat = _MinimalStrategy()
    with pytest.raises(ValueError, match="box_limit"):
        evaluate_strategy_mc_procedural(strat, p_rule=0.5, box_limit=999)


def test_validation_non_binary_measure_a():
    class BadMeasureStrategy:
        n2 = 2
        k_box = 1

        def measure_a(self, box_idx: int, field: int, out_a_prefix: int) -> int:
            return 5  # non-binary

        def comm(self, field: int, out_a: int) -> int:
            return 0

    with pytest.raises(ValueError, match="non-binary"):
        evaluate_strategy_mc_procedural(
            BadMeasureStrategy(),
            p_rule=0.5,
            n_train_samples=10,
            n_eval_samples=10,
        )


def test_validation_non_binary_comm():
    class BadCommStrategy:
        n2 = 2
        k_box = 1

        def measure_a(self, box_idx: int, field: int, out_a_prefix: int) -> int:
            return 0

        def comm(self, field: int, out_a: int) -> int:
            return 7  # non-binary

    with pytest.raises(ValueError, match="non-binary"):
        evaluate_strategy_mc_procedural(
            BadCommStrategy(),
            p_rule=0.5,
            n_train_samples=10,
            n_eval_samples=10,
        )


def test_validation_zero_n_train_samples():
    strat = _MinimalStrategy()
    with pytest.raises(ValueError, match="n_train_samples"):
        evaluate_strategy_mc_procedural(strat, p_rule=0.5, n_train_samples=0)


def test_validation_zero_n_eval_samples():
    strat = _MinimalStrategy()
    with pytest.raises(ValueError, match="n_eval_samples"):
        evaluate_strategy_mc_procedural(strat, p_rule=0.5, n_eval_samples=0)


def test_validation_confidence_out_of_range():
    strat = _MinimalStrategy()
    with pytest.raises(ValueError, match="confidence"):
        evaluate_strategy_mc_procedural(strat, p_rule=0.5, confidence=1.5)


# ---------------------------------------------------------------------------
# Pyramid helper unit tests
# ---------------------------------------------------------------------------

def test_pyramid_n2_4_measure_a_layer0():
    """Layer 0 boxes should XOR adjacent field bits."""
    s = PurePyramidProceduralAliceStrategy(n2=4, k_box=3)
    # field = 0b1010 = 10  -> bits: 0=0, 1=1, 2=0, 3=1
    field = 0b1010
    # box0: layer=0, pos=0 -> bits 0,1 -> 0 XOR 1 = 1
    assert s.measure_a(0, field, 0) == 1
    # box1: layer=0, pos=1 -> bits 2,3 -> 0 XOR 1 = 1
    assert s.measure_a(1, field, 0) == 1


def test_pyramid_n2_4_comm_perfect():
    """At p_rule=1.0, success should be essentially perfect for n2=4."""
    s = PurePyramidProceduralAliceStrategy(n2=4, k_box=3)
    result = evaluate_strategy_mc_procedural(
        s,
        p_rule=1.0,
        box_limit=2,
        subset_policy="up_to",
        n_train_samples=10_000,
        n_eval_samples=20_000,
        confidence=0.99,
        seed=42,
    )
    assert result.success > 0.99, f"Expected >0.99, got {result.success:.4f}"


def test_procedural_bob_result_is_dataclass():
    """ProceduralBobResult should be a frozen dataclass with all required fields."""
    r = ProceduralBobResult(
        success=0.75,
        ci_low=0.70,
        ci_high=0.80,
        stderr=0.01,
        n_train_samples=1000,
        n_eval_samples=2000,
        confidence=0.95,
        seed=0,
        p_rule=0.77,
        box_limit=3,
        subset_policy="up_to",
        elapsed_seconds=1.5,
    )
    assert r.success == 0.75
    assert r.elapsed_seconds == 1.5
    with pytest.raises(Exception):
        r.success = 0.0  # type: ignore  # frozen

