from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_fixtures import build_strategy_fixture
from ssf.hybrid_strategy_spec import HybridStrategySpec


def _majority_fixture(n2: int, k_box: int):
    spec = HybridStrategySpec(family="majority", n2=n2, k_box=k_box, tie_bandwidth=0)
    return build_strategy_fixture(spec)


def _assert_comm_invariant_ignoring_unused_bits(fixtures: list) -> None:
    baseline = fixtures[0]
    required = baseline.boxes_used
    assert required == 0
    low_mask = (1 << required) - 1

    for fixture in fixtures[1:]:
        assert fixture.boxes_used == required
        for field in range(2 ** baseline.n2):
            for out_a in range(2 ** fixture.k_box):
                out_a_low = out_a & low_mask
                assert fixture.comm_table[field, out_a] == baseline.comm_table[field, out_a_low]


def _assert_unused_f_tables_zero(fixtures: list) -> None:
    for fixture in fixtures:
        for d in range(fixture.boxes_used, fixture.k_box):
            assert np.all(fixture.f_tables[d] == 0)


def test_majority_n2_4_k_1_2_3_5_comm_invariant_and_unused_tables_zero():
    ks = [1, 2, 3, 5]
    fixtures = [_majority_fixture(n2=4, k_box=k_box) for k_box in ks]
    _assert_comm_invariant_ignoring_unused_bits(fixtures)
    _assert_unused_f_tables_zero(fixtures)


