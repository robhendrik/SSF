from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "generate" / "generate_a_strategy_fixtures.py"


def run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )


def load_manifest(out_dir: Path) -> list[dict]:
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def test_legacy_defaults_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixtures"
    run_script(["--output-dir", str(out_dir), "--legacy-defaults", "--overwrite"])

    assert (out_dir / "manifest.json").exists()
    expected_names = {
        "majority_n2_4_k_3_b_0",
        "pyramid_n2_4_k_3",
        "pyramid_n2_8_k_7",
    }
    names = {entry["name"] for entry in load_manifest(out_dir)}
    assert expected_names.issubset(names)
    for name in expected_names:
        assert (out_dir / f"{name}.npz").exists()


def test_manifest_fields_present(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixtures"
    run_script(["--output-dir", str(out_dir), "--legacy-defaults", "--overwrite"])
    manifest = load_manifest(out_dir)

    assert manifest
    for entry in manifest:
        for key in (
            "name",
            "family",
            "n2",
            "k_box",
            "required_boxes",
            "unused_boxes",
            "file",
            "analytical_success",
        ):
            assert key in entry


def test_script_stays_thin_on_pr_logic() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "write_strategy_fixture" in source
    assert "def build_f_tables_pyramid" not in source
    assert "def build_comm_pyramid" not in source
    assert "def _build_pyramid_layers" not in source


def test_include_all_legal_small_contains_all_families_and_even_ties(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixtures"
    run_script(
        [
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
            "--include-all-legal",
        ]
    )
    manifest = load_manifest(out_dir)

    assert {entry["family"] for entry in manifest} == {"majority", "pyramid", "horizontal", "vertical"}
    tie_values = [entry["tie_bandwidth"] for entry in manifest if entry["tie_bandwidth"] is not None]
    assert all(value % 2 == 0 for value in tie_values)


def test_dense_skip_writes_skipped_json_without_crashing(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixtures"
    run_script(
        [
            "--output-dir",
            str(out_dir),
            "--overwrite",
            "--n2-values",
            "16",
            "--k-box-values",
            "15",
            "--include-all-legal",
            "--max-n2-for-dense-fixtures",
            "8",
        ]
    )

    skipped_path = out_dir / "skipped.json"
    assert skipped_path.exists()
    skipped = json.loads(skipped_path.read_text(encoding="utf-8"))
    assert skipped
    reasons = "\n".join(entry["reason"] for entry in skipped)
    assert "dense threshold" in reasons or "max_comm_cells" in reasons or "comm_table size" in reasons


def test_analytical_value_for_pyramid_n2_8_k_7(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixtures"
    run_script(["--output-dir", str(out_dir), "--legacy-defaults", "--overwrite"])
    manifest = load_manifest(out_dir)

    target = next(entry for entry in manifest if entry["name"] == "pyramid_n2_8_k_7")
    assert target["analytical_success"]["0.90"] == pytest.approx(0.756)


def test_generated_npz_compatibility_shape(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixtures"
    run_script(["--output-dir", str(out_dir), "--legacy-defaults", "--overwrite"])

    target = out_dir / "majority_n2_4_k_3_b_0.npz"
    with np.load(target, allow_pickle=False) as data:
        assert "n2" in data
        assert "k_box" in data
        assert "comm_table" in data
        k_box = int(data["k_box"])
        for idx in range(k_box):
            assert f"f_table_{idx}" in data
