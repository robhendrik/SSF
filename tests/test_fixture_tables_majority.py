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


def test_majority_fixture_tables_n2_4_k3():
    strategy = _load_strategy_by_name("majority_n2_4_k_3_b_0")

    f_tables = strategy["f_tables"]
    comm = strategy["comm"]

    assert len(f_tables) == 3
    assert f_tables[0].shape == (16, 1)
    assert f_tables[1].shape == (16, 2)
    assert f_tables[2].shape == (16, 4)
    assert comm.shape == (16, 8)

    for table in f_tables:
        assert np.all(table == 0)

    for field in range(16):
        majority_bit = 1 if field.bit_count() > 2 else 0

        row_values = [int(comm[field, out_a]) for out_a in range(8)]
        assert all(v == majority_bit for v in row_values)

        # Explicitly verify communication is independent of out_a.
        assert len(set(row_values)) == 1


