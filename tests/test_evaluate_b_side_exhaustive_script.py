from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import evaluate_b_side_exhaustive as eval_script
from scripts.generate.generate_a_strategy_fixtures import generate_a_strategy_fixtures
from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name


def _generate_fixtures(fixture_dir: Path) -> None:
    generate_a_strategy_fixtures(
        output_dir=fixture_dir,
        overwrite=True,
        legacy_defaults=True,
    )


def test_script_imports_and_exposes_main():
    assert callable(eval_script.main)


def test_fixture_loading_from_dir_and_manifest(tmp_path: Path):
    fixture_dir = tmp_path / "fixtures"
    _generate_fixtures(fixture_dir)

    from_dir = eval_script._load_from_fixture_dir(fixture_dir)
    from_manifest = eval_script._load_from_manifest(fixture_dir / "manifest.json")

    assert len(from_dir) > 0
    assert len(from_manifest) == len(from_dir)


def test_output_directory_creation_and_files(tmp_path: Path):
    out_base = tmp_path / "results" / "exhaustive"
    out_base.mkdir(parents=True, exist_ok=True)

    run_dir = eval_script._timestamp_dir(out_base)
    assert run_dir.exists()
    assert run_dir.parent == out_base

    strategy_name = candidate_name(HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0))
    rows = [
        {
            "strategy_name": strategy_name,
            "family": "majority",
            "n2": 4,
            "k_box": 3,
            "boxes_used": None,
            "p_rule": 1.0,
            "box_limit": 1,
            "subset_policy": "up_to",
            "success": 0.6875,
            "runtime_seconds": 0.001,
        }
    ]
    eval_script.write_outputs(run_dir, rows, args_dict={"quick": True})

    assert (run_dir / "results.csv").exists()
    assert (run_dir / "results.md").exists()
    assert (run_dir / "summary.json").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 1


def test_tiny_exhaustive_majority_n2_4_k3(tmp_path: Path):
    fixture_dir = tmp_path / "fixtures"
    _generate_fixtures(fixture_dir)

    records = eval_script._load_from_fixture_dir(fixture_dir)
    strategy_name = candidate_name(HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0))
    target = next(r for r in records if r.strategy_name == strategy_name)

    rows = eval_script.evaluate_records(
        records=[target],
        p_rules=[1.0],
        box_limits=[1],
        subset_policy="up_to",
    )

    assert len(rows) == 1
    assert rows[0]["strategy_name"] == strategy_name
    assert math.isclose(rows[0]["success"], 0.6875, rel_tol=0.0, abs_tol=1e-12)

