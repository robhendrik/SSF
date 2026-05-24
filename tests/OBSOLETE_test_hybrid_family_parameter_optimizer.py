# from __future__ import annotations

# import csv
# import json
# import sys
# from pathlib import Path

# import pytest

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from scripts.optimize import optimize_hybrid_family_parameters as optimizer


# def test_vertical_candidate_enumeration_and_skipping():
#     # Should generate and skip some for n2=8, k_box=3
#     candidates, skipped = optimizer.enumerate_vertical_candidate_specs(
#         n2=8,
#         k_box=3,
#         orders=["pyr_maj", "maj_pyr"],
#         depths=None,
#         tie_bandwidth_values=None,
#         exclude_trivial_depths=True,
#     )
#     assert len(candidates) > 0
#     assert any(c.family.startswith("hybrid_vertical") for c in candidates)
#     assert len(skipped) > 0
#     assert any("reason" in item for item in skipped)


# def test_vertical_cli_quick_and_all(tmp_path: Path, monkeypatch):
#     call_log: list[dict] = []

#     def fake_exhaustive(strategy: dict, p_rule: float, box_limit: int | None, subset_policy: str = "up_to") -> float:
#         call_log.append(
#             {
#                 "n2": strategy["n2"],
#                 "k_box": strategy["k_box"],
#                 "p_rule": p_rule,
#                 "box_limit": box_limit,
#                 "subset_policy": subset_policy,
#             }
#         )
#         return float(0.5 + 0.01 * float(p_rule) + 0.001 * int(box_limit or 0))

#     monkeypatch.setattr(optimizer, "evaluate_strategy_exhaustive", fake_exhaustive)

#     for fam in ["vertical", "all"]:
#         output_dir = tmp_path / f"optimizer_output_{fam}"
#         exit_code = optimizer.main([
#             "--quick",
#             "--output-dir",
#             str(output_dir),
#             "--overwrite",
#             "--family",
#             fam,
#         ])
#         assert exit_code == 0
#         assert len(call_log) > 0

#         manifest_path = output_dir / "candidates_manifest.json"
#         csv_path = output_dir / "optimization_results.csv"
#         json_path = output_dir / "optimization_results.json"
#         md_path = output_dir / "optimization_summary.md"

#         assert manifest_path.exists()
#         assert csv_path.exists()
#         assert json_path.exists()
#         assert md_path.exists()

#         manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
#         assert manifest["generated_count"] > 0
#         assert manifest["skipped_count"] >= 0
#         # At least one vertical candidate in manifest
#         assert any(
#             (g.get("family", "").startswith("hybrid_vertical") or g.get("order") in ("pyr_maj", "maj_pyr"))
#             for g in manifest["generated"]
#         )

#         with csv_path.open("r", encoding="utf-8", newline="") as handle:
#             rows = list(csv.DictReader(handle))
#         assert len(rows) > 0
#         # Check vertical fields in output
#         for row in rows:
#             if row["family"].startswith("hybrid_vertical"):
#                 assert row["order"] in ("pyr_maj", "maj_pyr")
#                 assert row["depth_s"] is not None
#                 assert row["boxes_used"] is not None

#         payload = json.loads(json_path.read_text(encoding="utf-8"))
#         assert payload["row_count"] == len(rows)
#         assert payload["row_count"] > 0

#         fixtures_dir = output_dir / "fixtures"
#         best_dir = fixtures_dir / "best"
#         assert fixtures_dir.exists()
#         assert best_dir.exists()


# def test_optimizer_imports():
#     assert callable(optimizer.main)
#     assert callable(optimizer.run_optimizer)


# def test_invalid_configs_are_reported_not_silent():
#     candidates, skipped = optimizer.enumerate_horizontal_candidate_specs(
#         n2=8,
#         k_box=1,
#         tie_bandwidth_values=[0],
#         split_values=[1, 2, 3, 4, 5, 6, 7],
#     )
#     assert len(candidates) > 0
#     assert len(skipped) > 0
#     assert any("reason" in item for item in skipped)


# def test_quick_mode_runs_and_writes_outputs(tmp_path: Path, monkeypatch):
#     call_log: list[dict] = []

#     def fake_exhaustive(strategy: dict, p_rule: float, box_limit: int | None, subset_policy: str = "up_to") -> float:
#         call_log.append(
#             {
#                 "n2": strategy["n2"],
#                 "k_box": strategy["k_box"],
#                 "p_rule": p_rule,
#                 "box_limit": box_limit,
#                 "subset_policy": subset_policy,
#             }
#         )
#         return float(0.5 + 0.01 * float(p_rule) + 0.001 * int(box_limit or 0))

#     monkeypatch.setattr(optimizer, "evaluate_strategy_exhaustive", fake_exhaustive)

#     output_dir = tmp_path / "optimizer_output"
#     exit_code = optimizer.main([
#         "--quick",
#         "--output-dir",
#         str(output_dir),
#         "--overwrite",
#     ])

#     assert exit_code == 0
#     assert len(call_log) > 0

#     manifest_path = output_dir / "candidates_manifest.json"
#     csv_path = output_dir / "optimization_results.csv"
#     json_path = output_dir / "optimization_results.json"
#     md_path = output_dir / "optimization_summary.md"

#     assert manifest_path.exists()
#     assert csv_path.exists()
#     assert json_path.exists()
#     assert md_path.exists()

#     manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
#     assert manifest["generated_count"] > 0
#     assert manifest["skipped_count"] >= 0

#     with csv_path.open("r", encoding="utf-8", newline="") as handle:
#         rows = list(csv.DictReader(handle))
#     assert len(rows) > 0
#     assert "candidate_name" in rows[0]
#     assert "best_for_condition" in rows[0]

#     payload = json.loads(json_path.read_text(encoding="utf-8"))
#     assert payload["row_count"] == len(rows)
#     assert payload["row_count"] > 0

#     fixtures_dir = output_dir / "fixtures"
#     best_dir = fixtures_dir / "best"
#     assert fixtures_dir.exists()
#     assert best_dir.exists()
