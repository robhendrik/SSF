from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.bob_mc_procedural import ProceduralAliceStrategy
from ssf.hybrid_strategy_fixtures import build_strategy_fixture
from ssf.hybrid_strategy_procedural import build_procedural_strategy
from ssf.hybrid_strategy_spec import HybridStrategySpec


def _compare_against_dense(spec: HybridStrategySpec, comm_sample: bool = False) -> None:
    dense = build_strategy_fixture(spec)
    proc = build_procedural_strategy(spec)

    assert proc.n2 == dense.n2
    assert proc.k_box == dense.k_box
    assert proc.boxes_used == dense.boxes_used

    for d in range(proc.k_box):
        for field in range(1 << proc.n2):
            for prefix in range(1 << d):
                got = proc.measure_a(d, field, prefix)
                exp = int(dense.f_tables[d][field, prefix])
                assert got == exp

    if comm_sample:
        fields = [0, 1, 2, 3, 7, 15, 31, 63, 85, 127, 170, 255]
        out_as = [0, 1, 2, 3, 7, 15, 31, 63, 64, 95, 127]
        for field in fields:
            for out_a in out_as:
                got = proc.comm(field, out_a)
                exp = int(dense.comm_table[field, out_a])
                assert got == exp
    else:
        for field in range(1 << proc.n2):
            for out_a in range(1 << proc.k_box):
                got = proc.comm(field, out_a)
                exp = int(dense.comm_table[field, out_a])
                assert got == exp


def test_procedural_is_runtime_protocol() -> None:
    spec = HybridStrategySpec(family="pyramid", n2=4, k_box=3)
    proc = build_procedural_strategy(spec)
    assert isinstance(proc, ProceduralAliceStrategy)


def test_majority_matches_dense() -> None:
    spec = HybridStrategySpec(family="majority", n2=4, k_box=3, tie_bandwidth=0)
    _compare_against_dense(spec)


def test_pyramid_matches_dense() -> None:
    spec = HybridStrategySpec(family="pyramid", n2=4, k_box=3)
    _compare_against_dense(spec)


def test_pyramid_with_dummy_boxes_matches_dense() -> None:
    spec = HybridStrategySpec(family="pyramid", n2=4, k_box=5)
    _compare_against_dense(spec)


def test_horizontal_maj_pyr_matches_dense() -> None:
    spec = HybridStrategySpec(
        family="horizontal",
        n2=4,
        k_box=3,
        split=2,
        order="maj_pyr",
        tie_bandwidth=0,
    )
    _compare_against_dense(spec)


def test_horizontal_pyr_maj_matches_dense() -> None:
    spec = HybridStrategySpec(
        family="horizontal",
        n2=4,
        k_box=3,
        split=2,
        order="pyr_maj",
        tie_bandwidth=0,
    )
    _compare_against_dense(spec)


def test_vertical_maj_pyr_matches_dense() -> None:
    spec = HybridStrategySpec(
        family="vertical",
        n2=4,
        k_box=3,
        depth_s=1,
        order="maj_pyr",
        tie_bandwidth=0,
    )
    _compare_against_dense(spec)


def test_vertical_pyr_maj_matches_dense() -> None:
    spec = HybridStrategySpec(
        family="vertical",
        n2=4,
        k_box=3,
        depth_s=1,
        order="pyr_maj",
        tie_bandwidth=0,
    )
    _compare_against_dense(spec)


def test_vertical_pyr_maj_n8k7_matches_dense_sampled_comm() -> None:
    spec = HybridStrategySpec(
        family="vertical",
        n2=8,
        k_box=7,
        depth_s=2,
        order="pyr_maj",
        tie_bandwidth=0,
    )
    _compare_against_dense(spec, comm_sample=True)


def test_invalid_measure_and_comm_inputs_raise() -> None:
    spec = HybridStrategySpec(family="pyramid", n2=4, k_box=3)
    proc = build_procedural_strategy(spec)

    with pytest.raises(ValueError, match="box_idx out of range"):
        proc.measure_a(-1, 0, 0)
    with pytest.raises(ValueError, match="box_idx out of range"):
        proc.measure_a(3, 0, 0)

    with pytest.raises(ValueError, match="field out of range"):
        proc.measure_a(0, -1, 0)
    with pytest.raises(ValueError, match="field out of range"):
        proc.measure_a(0, 16, 0)

    with pytest.raises(ValueError, match="out_a_prefix out of range"):
        proc.measure_a(2, 0, -1)
    with pytest.raises(ValueError, match="out_a_prefix out of range"):
        proc.measure_a(2, 0, 4)

    with pytest.raises(ValueError, match="out_a out of range"):
        proc.comm(0, -1)
    with pytest.raises(ValueError, match="out_a out of range"):
        proc.comm(0, 8)


def test_unused_boxes_zero_and_comm_independent_of_unused_bits() -> None:
    spec = HybridStrategySpec(family="pyramid", n2=4, k_box=5)
    proc = build_procedural_strategy(spec)

    for field in range(1 << proc.n2):
        for prefix in range(1 << 3):
            assert proc.measure_a(3, field, prefix) == 0
        for prefix in range(1 << 4):
            assert proc.measure_a(4, field, prefix) == 0

    for field in range(1 << proc.n2):
        for low in range(1 << proc.boxes_used):
            baseline = proc.comm(field, low)
            for extra in range(1 << (proc.k_box - proc.boxes_used)):
                out_a = (extra << proc.boxes_used) | low
                assert proc.comm(field, out_a) == baseline
