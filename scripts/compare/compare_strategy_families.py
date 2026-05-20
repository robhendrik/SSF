from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


SSF_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = SSF_DIR / "results" / "comparison"


def _infer_family(strategy_name: str) -> str:
    lower = strategy_name.lower()
    if lower.startswith("majority"):
        return "majority"
    if lower.startswith("pyramid"):
        return "pyramid"
    if lower.startswith("hybrid"):
        return "hybrid"
    return "unknown"


def _normalize_row(row: Dict[str, str], source_file: str) -> Dict[str, Any]:
    strategy_name = (row.get("strategy_name") or row.get("candidate_name") or "").strip()
    if not strategy_name:
        raise ValueError("Missing strategy_name/candidate_name in input row.")

    if "success" not in row:
        raise ValueError("Missing success in input row.")
    if "p_rule" not in row:
        raise ValueError("Missing p_rule in input row.")

    family = (row.get("family") or "").strip() or _infer_family(strategy_name)
    out: Dict[str, Any] = {
        "strategy_name": strategy_name,
        "success": float(row["success"]),
        "p_rule": float(row["p_rule"]),
        "family": family,
        "source_file": source_file,
    }

    box_limit = row.get("box_limit")
    subset_policy = row.get("subset_policy")
    out["box_limit"] = box_limit if box_limit is not None else ""
    out["subset_policy"] = subset_policy if subset_policy is not None else ""
    return out


def load_and_normalize_rows(input_csv_paths: Iterable[Path]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for csv_path in input_csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            for row in reader:
                normalized.append(_normalize_row(row, csv_path.name))

    if not normalized:
        raise ValueError("No input rows loaded from input CSV files.")
    return normalized


def build_best_by_p_rule(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_for_p: Dict[float, Dict[str, Any]] = {}
    for row in rows:
        p_rule = float(row["p_rule"])
        if p_rule not in best_for_p or float(row["success"]) > float(best_for_p[p_rule]["success"]):
            best_for_p[p_rule] = row

    output: List[Dict[str, Any]] = []
    for p_rule in sorted(best_for_p):
        best = best_for_p[p_rule]
        output.append(
            {
                "p_rule": float(p_rule),
                "best_family": str(best["family"]),
                "best_strategy_name": str(best["strategy_name"]),
                "best_success": float(best["success"]),
                "source_file": str(best["source_file"]),
            }
        )
    return output


def build_family_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)

    output: List[Dict[str, Any]] = []
    for family in sorted(grouped):
        family_rows = grouped[family]
        scores = [float(item["success"]) for item in family_rows]
        best_row = max(family_rows, key=lambda item: float(item["success"]))
        output.append(
            {
                "family": family,
                "max_success": float(max(scores)),
                "mean_success": float(mean(scores)),
                "num_rows": int(len(family_rows)),
                "best_p_rule": float(best_row["p_rule"]),
                "best_strategy_name": str(best_row["strategy_name"]),
            }
        )
    return output


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.12g}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    combined_rows: List[Dict[str, Any]],
    best_rows: List[Dict[str, Any]],
    family_rows: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
) -> None:
    combined_columns = [
        "strategy_name",
        "family",
        "p_rule",
        "success",
        "box_limit",
        "subset_policy",
        "source_file",
    ]
    best_columns = ["p_rule", "best_family", "best_strategy_name", "best_success", "source_file"]
    family_columns = ["family", "max_success", "mean_success", "num_rows", "best_p_rule", "best_strategy_name"]

    _write_csv(output_dir / "combined_results.csv", combined_rows, combined_columns)
    _write_csv(output_dir / "best_by_p_rule.csv", best_rows, best_columns)
    _write_csv(output_dir / "family_summary.csv", family_rows, family_columns)

    md_lines = [
        "# Strategy Family Comparison",
        "",
        "## Best by p_rule",
        "",
        _markdown_table(best_rows, best_columns),
        "",
        "## Family Summary",
        "",
        _markdown_table(family_rows, family_columns),
        "",
        "Comparison is over provided input rows only; not a global optimum claim.",
        "",
    ]
    (output_dir / "comparison_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    payload = {
        "args": args_dict,
        "combined_row_count": len(combined_rows),
        "best_by_p_rule": best_rows,
        "family_summary": family_rows,
        "note": "Comparison is over provided input rows only; not a global optimum claim.",
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare strategy-family performance across input result tables.")
    parser.add_argument("--input-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group-by", type=str, default="family", choices=["family"])
    parser.add_argument("--x-axis", type=str, default="p_rule", choices=["p_rule"])
    parser.add_argument("--metric", type=str, default="success", choices=["success"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {args.output_dir}. Use --overwrite to replace outputs.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in (
        "combined_results.csv",
        "best_by_p_rule.csv",
        "family_summary.csv",
        "comparison_summary.md",
        "comparison_summary.json",
    ):
        out_path = args.output_dir / file_name
        if args.overwrite and out_path.exists():
            out_path.unlink()

    combined_rows = load_and_normalize_rows(args.input_csv)
    best_rows = build_best_by_p_rule(combined_rows)
    family_rows = build_family_summary(combined_rows)

    write_outputs(
        output_dir=args.output_dir,
        combined_rows=combined_rows,
        best_rows=best_rows,
        family_rows=family_rows,
        args_dict={
            "input_csv": [str(path) for path in args.input_csv],
            "output_dir": str(args.output_dir),
            "group_by": args.group_by,
            "x_axis": args.x_axis,
            "metric": args.metric,
            "overwrite": bool(args.overwrite),
        },
    )

    print(
        f"Loaded {len(combined_rows)} rows from {len(args.input_csv)} files. "
        f"Wrote outputs to {args.output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
