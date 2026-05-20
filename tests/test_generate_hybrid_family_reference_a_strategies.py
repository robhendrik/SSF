from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "generate" / "generate_a_strategy_fixtures.py"


def _run_script(args: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )


def _load_manifest(out_dir: Path):
    with (out_dir / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def test_generated_npz_shapes_and_dtypes(tmp_path: Path):
    out_dir = tmp_path / "fixtures"
    _run_script(
        [
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--families",
            "horizontal",
            "vertical",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
            "--include-all-legal",
        ]
    )
    manifest = _load_manifest(out_dir)

    for entry in manifest:
        n2 = int(entry["n2"])
        k_box = int(entry["k_box"])
        rows = 2**n2

        with np.load(out_dir / entry["file"], allow_pickle=False) as data:
            comm_table = data["comm_table"]
            assert comm_table.shape == (rows, 2**k_box)
            assert comm_table.dtype == np.uint8

            for d in range(k_box):
                table = data[f"f_table_{d}"]
                assert table.shape == (rows, 2**d)
                assert table.dtype == np.uint8


def test_unused_boxes_are_dummy_zero_tables(tmp_path: Path):
    out_dir = tmp_path / "fixtures"
    _run_script(
        [
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--families",
            "horizontal",
            "vertical",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
            "--include-all-legal",
        ]
    )
    manifest = _load_manifest(out_dir)

    for entry in manifest:
        k_box = int(entry["k_box"])
        boxes_used = int(entry["required_boxes"])
        with np.load(out_dir / entry["file"], allow_pickle=False) as data:
            for d in range(boxes_used, k_box):
                table = data[f"f_table_{d}"]
                assert np.all(table == 0), f"Expected zero dummy table for d={d} in {entry['name']}"


def test_known_rows_n2_4_horizontal_majority_left_fixture(tmp_path: Path):
    out_dir = tmp_path / "fixtures"
    _run_script(
        [
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--families",
            "horizontal",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
            "--include-all-legal",
        ]
    )
    manifest = _load_manifest(out_dir)

    target = next(e for e in manifest if e["name"] == "hybrid_horiz_maj_pyr_n2_4_k_3_split_2_2_b_0")
    with np.load(out_dir / target["file"], allow_pickle=False) as data:
        comm = data["comm_table"]
        f0 = data["f_table_0"]

        field = 0b0011  # bits: x0=1, x1=1, x2=0, x3=0
        x0 = (field >> 0) & 1
        x1 = (field >> 1) & 1
        x2 = (field >> 2) & 1
        x3 = (field >> 3) & 1
        assert x0 == 1 and x1 == 1 and x2 == 0 and x3 == 0
        assert int(f0[field, 0]) == (x2 ^ x3)
        for out_a in range(8):
            assert int(comm[field, out_a]) == 1

        field = 0b0101  # bits: x0=1, x1=0, x2=1, x3=0
        x0 = (field >> 0) & 1
        x1 = (field >> 1) & 1
        x2 = (field >> 2) & 1
        x3 = (field >> 3) & 1
        assert x0 == 1 and x1 == 0 and x2 == 1 and x3 == 0
        assert int(f0[field, 0]) == (x2 ^ x3)
        for out_a in range(8):
            a0 = (out_a >> 0) & 1
            expected = x2 ^ a0
            assert int(comm[field, out_a]) == expected


def test_manifest_contains_canonical_fields(tmp_path: Path):
    out_dir = tmp_path / "fixtures"
    _run_script(
        [
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--families",
            "horizontal",
            "vertical",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
            "--include-all-legal",
        ]
    )
    manifest = _load_manifest(out_dir)

    assert len(manifest) > 0
    forbidden = {"split_left", "split_right", "majority_side", "pyramid_side", "legal_config", "formula_notes", "boxes_used", "expected_scores"}
    required = {
        "split",
        "depth_s",
        "order",
        "required_boxes",
        "unused_boxes",
        "analytical_success",
        "analytical_method",
        "analytical_exact",
        "analytical_notes",
    }
    for entry in manifest:
        assert required.issubset(entry.keys())
        assert forbidden.isdisjoint(entry.keys())


def test_vertical_pyr_maj_n2_8_s2_k3_rejected_clearly(tmp_path: Path):
    out_dir = tmp_path / "fixtures"
    _run_script(
        [
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--families",
            "vertical",
            "--n2-values",
            "8",
            "--k-box-values",
            "3",
            "--include-all-legal",
        ]
    )
    manifest = _load_manifest(out_dir)
    names = {entry["name"] for entry in manifest}
    assert "hybrid_vert_pyr_maj_n2_8_k_3_s_2_b_0" not in names
    skipped = json.loads((out_dir / "skipped.json").read_text(encoding="utf-8"))
    assert any("k_box too small" in entry["reason"] for entry in skipped)


def test_default_generation_skips_invalid_vertical_pyr_maj_k3(tmp_path: Path):
    out_dir = tmp_path / "fixtures"
    _run_script(["--output-dir", str(out_dir), "--overwrite", "--legacy-defaults"])
    manifest = _load_manifest(out_dir)
    names = {entry["name"] for entry in manifest}
    assert "hybrid_vert_pyr_maj_n2_8_k_3_s_2_b_0" not in names


def test_hybrid_analytical_key_set(tmp_path: Path):
    out_dir = tmp_path / "fixtures"
    _run_script(["--output-dir", str(out_dir), "--overwrite", "--legacy-defaults"])
    manifest = _load_manifest(out_dir)
    hybrid_entries = [entry for entry in manifest if entry["family"] in {"horizontal", "vertical"}]
    assert hybrid_entries
    for entry in hybrid_entries:
        assert set(entry["analytical_success"].keys()) == {"0.50", "0.65", "0.77", "0.80", "0.90", "1.00"}
        assert entry["analytical_method"] == "closed_form"
        assert isinstance(entry["analytical_exact"], bool)
        assert isinstance(entry["analytical_notes"], str)



