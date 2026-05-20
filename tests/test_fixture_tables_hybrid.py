from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.strategy_io import load_manifest, load_strategy


FIXTURE_DIR = PROJECT_ROOT / "fixtures" / "persistent" / "a_strategies"


def _load_strategy_by_name(canonical_name: str):
    manifest = load_manifest(FIXTURE_DIR / "manifest.json")
    for entry in manifest:
        if entry["name"] == canonical_name:
            return load_strategy(FIXTURE_DIR / entry["file"])
    raise AssertionError(f"Missing fixture: {canonical_name}")


def test_hybrid_fixture_tables_n2_4_k3():
    strategy = _load_strategy_by_name("hybrid_horiz_maj_pyr_n2_4_k_3_split_2_2_b_0")

    f_tables = strategy["f_tables"]
    comm = strategy["comm"]

    assert len(f_tables) == 3
    assert f_tables[0].shape == (16, 1)
    assert f_tables[1].shape == (16, 2)
    assert f_tables[2].shape == (16, 4)
    assert comm.shape == (16, 8)

    for field in range(16):
        x0 = (field >> 0) & 1
        x1 = (field >> 1) & 1
        x2 = (field >> 2) & 1
        x3 = (field >> 3) & 1

        # Effective hybrid box on the last pair.
        assert int(f_tables[0][field, 0]) == (x2 ^ x3)

        for out_a in range(8):
            a0 = out_a & 1
            if x0 == x1:
                expected_comm = x0
            else:
                expected_comm = x2 ^ a0
            assert int(comm[field, out_a]) == expected_comm

    for table in f_tables[1:]:
        assert np.all((table == 0) | (table == 1))


