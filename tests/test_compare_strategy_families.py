from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare import compare_strategy_families as comparator


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_compare_strategy_families_outputs_and_aggregations(tmp_path: Path) -> None:
    csv1 = tmp_path / "exhaustive_like.csv"
    _write_csv(
        csv1,
        fieldnames=["strategy_name", "family", "p_rule", "success", "box_limit", "subset_policy"],
        rows=[
            {
                "strategy_name": "majority_n2_4_k_3",
                "family": "majority",
                "p_rule": 0.8,
                "success": 0.70,
                "box_limit": 3,
                "subset_policy": "up_to",
            },
            {
                "strategy_name": "pyramid_n2_4_k_3",
                "family": "pyramid",
                "p_rule": 0.8,
                "success": 0.74,
                "box_limit": 3,
                "subset_policy": "up_to",
            },
        ],
    )

    csv2 = tmp_path / "optimizer_like.csv"
    _write_csv(
        csv2,
        fieldnames=["candidate_name", "family", "p_rule", "success"],
        rows=[
            {
                "candidate_name": "hybrid_horiz_candidate_a",
                "family": "hybrid",
                "p_rule": 0.8,
                "success": 0.78,
            },
            {
                "candidate_name": "hybrid_horiz_candidate_b",
                "family": "hybrid",
                "p_rule": 0.9,
                "success": 0.82,
            },
        ],
    )

    output_dir = tmp_path / "comparison_out"
    exit_code = comparator.main([
        "--input-csv",
        str(csv1),
        str(csv2),
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ])
    assert exit_code == 0

    combined_csv = output_dir / "combined_results.csv"
    best_csv = output_dir / "best_by_p_rule.csv"
    family_csv = output_dir / "family_summary.csv"
    summary_md = output_dir / "comparison_summary.md"
    summary_json = output_dir / "comparison_summary.json"

    assert combined_csv.exists()
    assert best_csv.exists()
    assert family_csv.exists()
    assert summary_md.exists()
    assert summary_json.exists()

    with best_csv.open("r", encoding="utf-8", newline="") as handle:
        best_rows = list(csv.DictReader(handle))
    assert len(best_rows) == 2

    best_map = {float(row["p_rule"]): row for row in best_rows}
    assert float(best_map[0.8]["best_success"]) == 0.78
    assert best_map[0.8]["best_family"] == "hybrid"
    assert best_map[0.8]["best_strategy_name"] == "hybrid_horiz_candidate_a"

    assert float(best_map[0.9]["best_success"]) == 0.82
    assert best_map[0.9]["best_family"] == "hybrid"
    assert best_map[0.9]["best_strategy_name"] == "hybrid_horiz_candidate_b"

    with family_csv.open("r", encoding="utf-8", newline="") as handle:
        family_rows = list(csv.DictReader(handle))
    assert len(family_rows) == 3

    family_map = {row["family"]: row for row in family_rows}
    assert int(family_map["hybrid"]["num_rows"]) == 2
    assert float(family_map["hybrid"]["max_success"]) == 0.82
    assert abs(float(family_map["hybrid"]["mean_success"]) - 0.80) < 1e-12
    assert float(family_map["hybrid"]["best_p_rule"]) == 0.9
    assert family_map["hybrid"]["best_strategy_name"] == "hybrid_horiz_candidate_b"

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["combined_row_count"] == 4
    assert "provided input rows only" in payload["note"]


def test_compare_default_output_dir_uses_results() -> None:
    expected = PROJECT_ROOT / "results" / "comparison"
    assert comparator.DEFAULT_OUTPUT_DIR == expected
