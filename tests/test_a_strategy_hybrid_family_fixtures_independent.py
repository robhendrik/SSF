from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_fixtures import build_strategy_fixture
from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name


FIXTURE_DIR = ROOT / "fixtures" / "persistent" / "a_strategies"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        pytest.fail(f"Missing manifest file: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        pytest.fail("Manifest must be a non-empty list.")
    return manifest


def _assert_binary(array: np.ndarray, label: str) -> None:
    if not np.all((array == 0) | (array == 1)):
        pytest.fail(f"{label} contains non-binary values")


def _entry_to_spec(entry: dict) -> HybridStrategySpec:
    return HybridStrategySpec(
        family=entry["family"],
        n2=int(entry["n2"]),
        k_box=int(entry["k_box"]),
        split=entry.get("split"),
        depth_s=entry.get("depth_s"),
        order=entry.get("order"),
        tie_bandwidth=entry.get("tie_bandwidth"),
        tie_value=entry.get("tie_value", 0),
    )


def _manifest_entries_by_file(manifest: list[dict]) -> dict[str, dict]:
    entries_by_file: dict[str, dict] = {}
    for entry in manifest:
        file_name = entry.get("file")
        if not file_name:
            pytest.fail(f"Manifest entry missing file field: {entry}")
        if file_name in entries_by_file:
            pytest.fail(f"Duplicate file in manifest: {file_name}")
        entries_by_file[file_name] = entry
    return entries_by_file


def _manifest_entries() -> list[dict]:
    return _load_manifest()


@pytest.mark.parametrize("entry", _manifest_entries(), ids=lambda e: e["file"])
def test_manifest_name_and_file_match_candidate_name(entry: dict) -> None:
    spec = _entry_to_spec(entry)
    expected_name = candidate_name(spec)

    assert entry["name"] == expected_name
    assert entry["file"] == f"{expected_name}.npz"
    assert (FIXTURE_DIR / entry["file"]).exists(), f"Missing fixture file: {entry['file']}"


def test_all_npz_files_are_referenced_by_manifest() -> None:
    manifest = _load_manifest()
    entries_by_file = _manifest_entries_by_file(manifest)

    npz_files = {path.name for path in FIXTURE_DIR.glob("*.npz")}
    manifest_files = set(entries_by_file.keys())

    assert npz_files == manifest_files, (
        f"Mismatch between fixture files and manifest entries. "
        f"Only on disk: {sorted(npz_files - manifest_files)}; "
        f"Only in manifest: {sorted(manifest_files - npz_files)}"
    )


@pytest.mark.parametrize("entry", _manifest_entries(), ids=lambda e: e["file"])
def test_fixture_shapes_binary_and_exact_arrays_match_builder(entry: dict) -> None:
    spec = _entry_to_spec(entry)
    expected = build_strategy_fixture(spec)
    fixture_path = FIXTURE_DIR / entry["file"]

    with np.load(fixture_path, allow_pickle=False) as data:
        n2 = int(np.asarray(data["n2"]).item())
        k_box = int(np.asarray(data["k_box"]).item())

        assert n2 == expected.n2
        assert k_box == expected.k_box

        rows = 2**n2
        comm = np.asarray(data["comm_table"], dtype=np.uint8)
        assert comm.shape == (rows, 2**k_box)
        _assert_binary(comm, f"{entry['file']}:comm_table")

        expected_f_tables = expected.f_tables
        actual_f_tables = []
        for d in range(k_box):
            key = f"f_table_{d}"
            if key not in data:
                pytest.fail(f"Missing required array {key} in {entry['file']}")
            table = np.asarray(data[key], dtype=np.uint8)
            assert table.shape == (rows, 2**d)
            _assert_binary(table, f"{entry['file']}:{key}")
            actual_f_tables.append(table)

        assert np.array_equal(comm, expected.comm_table), f"comm_table mismatch for {entry['file']}"
        for d in range(k_box):
            assert np.array_equal(actual_f_tables[d], expected_f_tables[d]), (
                f"f_table_{d} mismatch for {entry['file']}"
            )

