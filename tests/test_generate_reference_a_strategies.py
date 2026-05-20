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


def _load_manifest(out_dir: Path) -> list[dict]:
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def test_canonical_reference_generation_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixtures"
    _run_script(
        [
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--families",
            "majority",
            "pyramid",
            "--n2-values",
            "4",
            "8",
            "--k-box-values",
            "3",
            "7",
            "--include-all-legal",
        ]
    )

    manifest = _load_manifest(out_dir)
    assert manifest
    assert {entry["family"] for entry in manifest} == {"majority", "pyramid"}

    names = {entry["name"] for entry in manifest}
    assert "majority_n2_4_k_3_b_0" in names
    assert "pyramid_n2_4_k_3" in names
    assert "pyramid_n2_8_k_7" in names


def test_canonical_manifest_schema_reference_families(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixtures"
    _run_script(
        [
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--families",
            "majority",
            "pyramid",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
            "--include-all-legal",
        ]
    )

    manifest = _load_manifest(out_dir)
    assert manifest
    forbidden = {"split_left", "split_right", "majority_side", "pyramid_side", "legal_config", "formula_notes", "boxes_used"}
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


def test_canonical_reference_npz_shapes(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixtures"
    _run_script(
        [
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--families",
            "majority",
            "pyramid",
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
        with np.load(out_dir / entry["file"], allow_pickle=False) as data:
            assert int(data["n2"]) == n2
            assert int(data["k_box"]) == k_box
            assert data["comm_table"].shape == (2 ** n2, 2 ** k_box)
            for d in range(k_box):
                assert data[f"f_table_{d}"].shape == (2 ** n2, 2 ** d)


def test_canonical_pyramid_analytical_point(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixtures"
    _run_script(["--output-dir", str(out_dir), "--legacy-defaults", "--overwrite"])
    manifest = _load_manifest(out_dir)
    target = next(entry for entry in manifest if entry["name"] == "pyramid_n2_8_k_7")
    assert target["analytical_success"]["0.90"] == pytest.approx(0.756)


