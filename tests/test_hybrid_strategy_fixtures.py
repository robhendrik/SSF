import numpy as np
import pytest
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name
from ssf.hybrid_strategy_fixtures import build_strategy_fixture, write_strategy_fixture, _majority_or_tie


def test_pure_pyramid_shapes():
    spec = HybridStrategySpec(family="pyramid", n2=4, k_box=3)
    fixture = build_strategy_fixture(spec)
    assert len(fixture.f_tables) == 3
    assert fixture.f_tables[0].shape == (16, 1)
    assert fixture.f_tables[1].shape == (16, 2)
    assert fixture.f_tables[2].shape == (16, 4)
    assert fixture.comm_table.shape == (16, 8)
    for arr in fixture.f_tables:
        assert arr.dtype == np.uint8
    assert fixture.comm_table.dtype == np.uint8


def test_pure_majority_comm_table():
    spec = HybridStrategySpec(family="majority", n2=4, k_box=3, tie_bandwidth=0)
    fixture = build_strategy_fixture(spec)
    # comm_table is independent of out_a
    for field in range(16):
        row = fixture.comm_table[field, :]
        assert np.all(row == row[0])
    # field with majority 1
    assert fixture.comm_table[0b1111, 0] == 1
    # field with majority 0
    assert fixture.comm_table[0b0000, 0] == 0
    # tie (2 ones, 2 zeros), tie_value=0
    assert fixture.comm_table[0b0011, 0] == 0


def test_pure_pyramid_entries():
    spec = HybridStrategySpec(family="pyramid", n2=4, k_box=3)
    fixture = build_strategy_fixture(spec)
    # box0: x0 XOR x1
    for field in range(16):
        x0 = (field >> 0) & 1
        x1 = (field >> 1) & 1
        a0 = x0 ^ x1
        assert fixture.f_tables[0][field, 0] == a0
    # box1: x2 XOR x3
    for field in range(16):
        x2 = (field >> 2) & 1
        x3 = (field >> 3) & 1
        a1 = x2 ^ x3
        assert fixture.f_tables[1][field, 0] == a1
    # box2: output(box0) XOR output(box1)
    for field in range(16):
        for out_a in range(8):
            x0 = (field >> 0) & 1
            x1 = (field >> 1) & 1
            x2 = (field >> 2) & 1
            x3 = (field >> 3) & 1
            oA0 = (out_a >> 0) & 1
            oA1 = (out_a >> 1) & 1
            unit0 = x0 ^ oA0
            unit1 = x2 ^ oA1
            a2 = unit0 ^ unit1
            out_a_prefix = out_a & 0b11
            assert fixture.f_tables[2][field, out_a_prefix] == a2


def test_extra_dummy_boxes():
    spec = HybridStrategySpec(family="pyramid", n2=4, k_box=5)
    fixture = build_strategy_fixture(spec)
    # first 3 boxes are pyramid, last 2 are zeros
    for d in range(3, 5):
        assert np.all(fixture.f_tables[d] == 0)
    # comm_table is identical across unused out_a bits
    for field in range(16):
        for out_a in range(8):
            for extra in range(4):
                out_a_full = (extra << 3) | out_a
                assert fixture.comm_table[field, out_a_full] == fixture.comm_table[field, out_a]


def test_horizontal_maj_pyr():
    spec = HybridStrategySpec(family="horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0)
    fixture = build_strategy_fixture(spec)
    for field in range(16):
        left = [(field >> 0) & 1, (field >> 1) & 1]
        maj = 1 if sum(left) > 1 else 0
        for out_a in range(8):
            left_sum = sum(left)
            is_tie = abs(2 * left_sum - 2) <= 0

            if is_tie:
                branch_left_bit = (field >> 2) & 1
                apex = branch_left_bit ^ ((out_a >> 0) & 1)
                assert fixture.comm_table[field, out_a] == apex
            else:
                maj = 1 if left_sum > 1 else 0
                assert fixture.comm_table[field, out_a] == maj          

def test_horizontal_pyr_maj():
    spec = HybridStrategySpec(family="horizontal", n2=4, k_box=3, split=2, order="pyr_maj", tie_bandwidth=0)
    fixture = build_strategy_fixture(spec)
    for field in range(16):
        right = [(field >> 2) & 1, (field >> 3) & 1]
        maj = 1 if sum(right) > 1 else 0
        for out_a in range(8):
            right_sum = sum(right)
            is_tie = abs(2 * right_sum - 2) <= 0

            if is_tie:
                branch_left_bit = (field >> 0) & 1
                apex = branch_left_bit ^ ((out_a >> 0) & 1)
                assert fixture.comm_table[field, out_a] == apex
            else:
                maj = 1 if right_sum > 1 else 0
                assert fixture.comm_table[field, out_a] == maj          


def test_vertical_maj_pyr():
    spec = HybridStrategySpec(family="vertical", n2=4, k_box=1, depth_s=1, order="maj_pyr", tie_bandwidth=0)
    fixture = build_strategy_fixture(spec)
    for field in range(16):
        block0 = [(field >> 0) & 1, (field >> 1) & 1]
        block1 = [(field >> 2) & 1, (field >> 3) & 1]
        maj0 = 1 if sum(block0) > 1 else 0
        maj1 = 1 if sum(block1) > 1 else 0
        for out_a in range(2):
            apex = maj0 ^ ((out_a >> 0) & 1)
            assert fixture.comm_table[field, out_a] == apex
            if out_a == 0:
                assert fixture.f_tables[0][field, out_a] == maj0 ^ maj1


def test_vertical_pyr_maj():
    spec = HybridStrategySpec(family="vertical", n2=4, k_box=2, depth_s=1, order="pyr_maj", tie_bandwidth=0)
    fixture = build_strategy_fixture(spec)
    for field in range(16):
        x0 = (field >> 0) & 1
        x1 = (field >> 1) & 1
        x2 = (field >> 2) & 1
        x3 = (field >> 3) & 1
        for out_a in range(4):
            apex0 = x0 ^ ((out_a >> 0) & 1)
            apex1 = x2 ^ ((out_a >> 1) & 1)
            maj = _majority_or_tie([apex0, apex1], 0, 0)
            assert fixture.comm_table[field, out_a] == maj
            if out_a & 0b01 == 0:
                assert fixture.f_tables[0][field, out_a & 0b01] == x0 ^ x1
            if out_a & 0b11 == 0:
                assert fixture.f_tables[1][field, out_a & 0b11] == x2 ^ x3


def test_write_strategy_fixture(tmp_path):
    spec = HybridStrategySpec(family="pyramid", n2=4, k_box=3)
    out_path = write_strategy_fixture(spec, tmp_path)
    assert out_path.name == candidate_name(spec) + ".npz"
    data = np.load(out_path)
    assert data["n2"] == 4
    assert data["k_box"] == 3
    assert data["comm_table"].shape == (16, 8)
    for i in range(3):
        assert data[f"f_table_{i}"].shape == (16, 2 ** i)


def test_as_strategy_dict():
    spec = HybridStrategySpec(family="pyramid", n2=4, k_box=3)
    fixture = build_strategy_fixture(spec)
    d = fixture.as_strategy_dict()
    assert set(d.keys()) == {"n2", "k_box", "f_tables", "comm", "comm_table"}
    assert np.array_equal(d["comm"], d["comm_table"])
    assert len(d["f_tables"]) == 3


def test_invalid_spec_raises():
    # k_box too small
    spec = HybridStrategySpec(family="pyramid", n2=4, k_box=1)
    with pytest.raises(ValueError):
        build_strategy_fixture(spec)
    # invalid family
    from ssf.hybrid_strategy_spec import HybridStrategySpec as HSS
    bad = HSS(family="notafam", n2=4, k_box=3)
    with pytest.raises(ValueError):
        build_strategy_fixture(bad)
