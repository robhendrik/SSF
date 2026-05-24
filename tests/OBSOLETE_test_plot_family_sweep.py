# from __future__ import annotations

# import csv
# import sys
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from scripts.compare import plot_family_sweep as sweep


# def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
#     with path.open("w", encoding="utf-8", newline="") as handle:
#         writer = csv.DictWriter(handle, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in rows:
#             writer.writerow(row)


# def test_plot_family_sweep_outputs_and_best_rows(tmp_path: Path) -> None:
#     input_csv = tmp_path / "input.csv"
#     _write_csv(
#         input_csv,
#         ["strategy_name", "family", "p_rule", "success", "box_limit", "subset_policy"],
#         [
#             {
#                 "strategy_name": "majority_a",
#                 "family": "majority",
#                 "p_rule": 0.8,
#                 "success": 0.70,
#                 "box_limit": 3,
#                 "subset_policy": "up_to",
#             },
#             {
#                 "strategy_name": "majority_b",
#                 "family": "majority",
#                 "p_rule": 0.8,
#                 "success": 0.73,
#                 "box_limit": 3,
#                 "subset_policy": "up_to",
#             },
#             {
#                 "strategy_name": "majority_c",
#                 "family": "majority",
#                 "p_rule": 0.9,
#                 "success": 0.76,
#                 "box_limit": 1,
#                 "subset_policy": "exact",
#             },
#             {
#                 "strategy_name": "hybrid_x",
#                 "family": "hybrid",
#                 "p_rule": 0.8,
#                 "success": 0.78,
#                 "box_limit": 3,
#                 "subset_policy": "up_to",
#             },
#             {
#                 "strategy_name": "hybrid_y",
#                 "family": "hybrid",
#                 "p_rule": 0.9,
#                 "success": 0.80,
#                 "box_limit": 3,
#                 "subset_policy": "up_to",
#             },
#         ],
#     )

#     output_dir = tmp_path / "out"
#     exit_code = sweep.main([
#         "--input-csv",
#         str(input_csv),
#         "--output-dir",
#         str(output_dir),
#         "--overwrite",
#     ])
#     assert exit_code == 0

#     png_path = output_dir / "success_vs_p_rule.png"
#     csv_path = output_dir / "comparison_results.csv"
#     md_path = output_dir / "comparison_summary.md"

#     assert png_path.exists()
#     assert csv_path.exists()
#     assert md_path.exists()

#     with csv_path.open("r", encoding="utf-8", newline="") as handle:
#         rows = list(csv.DictReader(handle))

#     row_map = {(row["family"], float(row["p_rule"])): row for row in rows}
#     assert float(row_map[("majority", 0.8)]["best_success"]) == 0.73
#     assert row_map[("majority", 0.8)]["best_strategy_name"] == "majority_b"
#     assert float(row_map[("hybrid", 0.8)]["best_success"]) == 0.78
#     assert row_map[("hybrid", 0.8)]["best_strategy_name"] == "hybrid_x"


# def test_plot_family_sweep_filter_box_limit(tmp_path: Path) -> None:
#     input_csv = tmp_path / "input_filter.csv"
#     _write_csv(
#         input_csv,
#         ["candidate_name", "family", "p_rule", "success", "box_limit", "subset_policy"],
#         [
#             {
#                 "candidate_name": "hybrid_a",
#                 "family": "hybrid",
#                 "p_rule": 0.9,
#                 "success": 0.79,
#                 "box_limit": 1,
#                 "subset_policy": "up_to",
#             },
#             {
#                 "candidate_name": "hybrid_b",
#                 "family": "hybrid",
#                 "p_rule": 0.9,
#                 "success": 0.82,
#                 "box_limit": 3,
#                 "subset_policy": "up_to",
#             },
#         ],
#     )

#     output_dir = tmp_path / "out_filter"
#     exit_code = sweep.main([
#         "--input-csv",
#         str(input_csv),
#         "--output-dir",
#         str(output_dir),
#         "--filter-box-limit",
#         "3",
#         "--overwrite",
#     ])
#     assert exit_code == 0

#     csv_path = output_dir / "comparison_results.csv"
#     with csv_path.open("r", encoding="utf-8", newline="") as handle:
#         rows = list(csv.DictReader(handle))

#     hybrid_rows = [r for r in rows if r["family"] == "hybrid" and float(r["p_rule"]) == 0.9]
#     assert len(hybrid_rows) == 1
#     assert hybrid_rows[0]["best_strategy_name"] == "hybrid_b"
#     assert float(hybrid_rows[0]["best_success"]) == 0.82


# def test_plot_default_output_dir_uses_results() -> None:
#     expected = PROJECT_ROOT / "results" / "plots"
#     assert sweep.DEFAULT_OUTPUT_DIR == expected
