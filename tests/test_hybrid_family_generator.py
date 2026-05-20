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


def _assert_unused_tables_zero(fixture_path: Path, boxes_used: int, k_box: int) -> None:
    with np.load(fixture_path, allow_pickle=False) as data:
        for d in range(boxes_used, k_box):
            table = np.asarray(data[f"f_table_{d}"], dtype=np.uint8)
            assert np.all(table == 0), f"unused table f_table_{d} is not zeroed"


def test_vertical_and_horizontal_build_for_n2_8_k7(tmp_path: Path) -> None:
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
            "8",
            "--k-box-values",
            "7",
            "--include-all-legal",
        ]
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest

    horizontal = [entry for entry in manifest if entry["family"] == "horizontal"]
    vertical = [entry for entry in manifest if entry["family"] == "vertical"]
    assert horizontal
    assert vertical

    for entry in manifest:
        k_box = int(entry["k_box"])
        required = int(entry["required_boxes"])
        _assert_unused_tables_zero(out_dir / entry["file"], required, k_box)


def test_illegal_configs_rejected_clearly(tmp_path: Path) -> None:
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
            "5",
            "--include-all-legal",
        ]
    )

    skipped = json.loads((out_dir / "skipped.json").read_text(encoding="utf-8"))
    assert skipped
    assert any("k_box too small" in item["reason"] for item in skipped)


def test_include_all_legal_generates_manifest_fields(tmp_path: Path) -> None:
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
            "8",
            "--k-box-values",
            "1",
            "3",
            "7",
            "--include-all-legal",
        ]
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) > 0

    required_keys = {
        "name",
        "family",
        "n2",
        "k_box",
        "required_boxes",
        "unused_boxes",
        "split",
        "order",
        "depth_s",
        "tie_bandwidth",
        "file",
        "analytical_success",
        "analytical_method",
        "analytical_exact",
        "analytical_notes",
    }
    forbidden = {"split_left", "split_right", "majority_side", "pyramid_side", "legal_config", "formula_notes", "boxes_used"}
    for entry in manifest:
        assert required_keys.issubset(entry.keys())
        assert forbidden.isdisjoint(entry.keys())
