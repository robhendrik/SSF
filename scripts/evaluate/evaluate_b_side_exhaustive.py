from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Ensure local src package is importable when script is run directly.
SSF_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SSF_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.bob_exhaustive import evaluate_strategy_exhaustive
from ssf.strategy_io import load_manifest, load_strategy


DEFAULT_FIXTURE_DIR = SSF_DIR / "fixtures" / "persistent" / "a_strategies"
DEFAULT_OUTPUT_DIR = SSF_DIR / "results" / "exhaustive"
DEFAULT_P_RULES = (0.5, 0.65, 0.77, 1.0)
DEFAULT_BOX_LIMITS = (None,)


@dataclass
class StrategyRecord:
    strategy: Dict[str, Any]
    strategy_name: str
    family: str
    boxes_used: Optional[int]
    fixture_path: Path


def _parse_float_list(raw: str) -> List[float]:
    vals = [p.strip() for p in raw.split(",") if p.strip()]
    if not vals:
        raise ValueError("--p-rules cannot be empty.")
    return [float(v) for v in vals]


def _parse_box_limits(raw: str) -> List[Optional[int]]:
    vals = [v.strip().lower() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("--box-limits cannot be empty.")

    out: List[Optional[int]] = []
    for v in vals:
        if v in {"none", "all"}:
            out.append(None)
        else:
            out.append(int(v))
    return out


def _normalize_family(name: str) -> str:
    lower = name.lower()
    if lower.startswith("majority"):
        return "majority"
    if lower.startswith("pyramid"):
        return "pyramid"
    if lower.startswith("hybrid"):
        return "hybrid"
    return "unknown"


def _load_from_fixture_dir(fixture_dir: Path) -> List[StrategyRecord]:
    if not fixture_dir.exists():
        raise FileNotFoundError(f"Fixture directory not found: {fixture_dir}")

    npz_files = sorted(fixture_dir.glob("*.npz"))
    local_manifest = fixture_dir / "manifest.json"
    manifest_path = local_manifest if local_manifest.exists() else None
    records: List[StrategyRecord] = []
    for npz_path in npz_files:
        strategy = load_strategy(npz_path, manifest_path=manifest_path)
        strategy_name = strategy.get("name") or npz_path.stem
        records.append(
            StrategyRecord(
                strategy=strategy,
                strategy_name=str(strategy_name),
                family=_normalize_family(str(strategy_name)),
                boxes_used=None,
                fixture_path=npz_path,
            )
        )
    return records


def _load_from_manifest(manifest_path: Path) -> List[StrategyRecord]:
    entries = load_manifest(manifest_path)
    records: List[StrategyRecord] = []

    for entry in entries:
        file_name = entry.get("file")
        if not file_name:
            continue

        fixture_path = manifest_path.parent / str(file_name)
        strategy = load_strategy(fixture_path, manifest_path=manifest_path)
        strategy_name = str(entry.get("name") or strategy.get("name") or fixture_path.stem)
        family = str(entry.get("family") or _normalize_family(strategy_name))

        boxes_used_raw = entry.get("boxes_used")
        boxes_used = int(boxes_used_raw) if boxes_used_raw is not None else None

        records.append(
            StrategyRecord(
                strategy=strategy,
                strategy_name=strategy_name,
                family=family,
                boxes_used=boxes_used,
                fixture_path=fixture_path,
            )
        )

    return records


def _filter_records(
    records: Iterable[StrategyRecord],
    strategy_filter: Optional[str],
    max_strategies: Optional[int],
) -> List[StrategyRecord]:
    filtered = list(records)
    if strategy_filter:
        needle = strategy_filter.lower()
        filtered = [r for r in filtered if needle in r.strategy_name.lower()]

    if max_strategies is not None and max_strategies >= 0:
        filtered = filtered[:max_strategies]

    return filtered


def _timestamp_dir(base_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _safe_overwrite_base(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)


def _to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "none"
        if isinstance(v, float):
            return f"{v:.12g}"
        return str(v)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]

    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")

    return "\n".join(lines) + "\n"


def evaluate_records(
    records: List[StrategyRecord],
    p_rules: List[float],
    box_limits: List[Optional[int]],
    subset_policy: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for rec in records:
        n2 = int(rec.strategy["n2"])
        k_box = int(rec.strategy["k_box"])

        for p_rule in p_rules:
            for box_limit in box_limits:
                t0 = time.perf_counter()
                success = evaluate_strategy_exhaustive(
                    rec.strategy,
                    p_rule=float(p_rule),
                    box_limit=box_limit,
                    subset_policy=subset_policy,
                )
                dt = time.perf_counter() - t0

                results.append(
                    {
                        "strategy_name": rec.strategy_name,
                        "family": rec.family,
                        "n2": n2,
                        "k_box": k_box,
                        "boxes_used": rec.boxes_used,
                        "p_rule": float(p_rule),
                        "box_limit": box_limit,
                        "subset_policy": subset_policy,
                        "success": float(success),
                        "runtime_seconds": float(dt),
                    }
                )

    return results


def write_outputs(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
) -> None:
    columns = [
        "strategy_name",
        "family",
        "n2",
        "k_box",
        "boxes_used",
        "p_rule",
        "box_limit",
        "subset_policy",
        "success",
        "runtime_seconds",
    ]

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = out_dir / "results.md"
    md_path.write_text(_to_markdown_table(rows, columns), encoding="utf-8")

    summary: Dict[str, Any] = {
        "run_dir": str(out_dir),
        "row_count": len(rows),
        "args": args_dict,
        "best": None,
    }

    if rows:
        best = max(rows, key=lambda r: float(r["success"]))
        summary["best"] = best

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed A-side fixtures with exhaustive Bob-side optimization.")
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--p-rules", type=str, default=",".join(str(v) for v in DEFAULT_P_RULES))
    parser.add_argument("--box-limits", type=str, default="none")
    parser.add_argument("--subset-policy", type=str, default="up_to", choices=["up_to", "exact"])
    parser.add_argument("--max-strategies", type=int, default=None)
    parser.add_argument("--strategy-filter", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.manifest is not None and args.fixture_dir is not None and args.fixture_dir != DEFAULT_FIXTURE_DIR:
        raise ValueError("Provide either --fixture-dir or --manifest, not both.")

    if args.manifest is not None:
        records = _load_from_manifest(args.manifest)
    else:
        records = _load_from_fixture_dir(args.fixture_dir)

    records = _filter_records(records, args.strategy_filter, args.max_strategies)
    if not records:
        raise ValueError("No strategies selected for evaluation.")

    p_rules = _parse_float_list(args.p_rules)
    box_limits = _parse_box_limits(args.box_limits)

    if args.quick:
        records = records[:2]
        p_rules = p_rules[:1]
        box_limits = box_limits[:1]

    if args.overwrite:
        _safe_overwrite_base(args.output_dir)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _timestamp_dir(args.output_dir)

    rows = evaluate_records(
        records=records,
        p_rules=p_rules,
        box_limits=box_limits,
        subset_policy=args.subset_policy,
    )

    write_outputs(
        out_dir=run_dir,
        rows=rows,
        args_dict={
            "fixture_dir": str(args.fixture_dir) if args.fixture_dir is not None else None,
            "manifest": str(args.manifest) if args.manifest is not None else None,
            "output_dir": str(args.output_dir),
            "p_rules": p_rules,
            "box_limits": box_limits,
            "subset_policy": args.subset_policy,
            "max_strategies": args.max_strategies,
            "strategy_filter": args.strategy_filter,
            "quick": args.quick,
        },
    )

    print(f"Evaluated {len(rows)} rows. Results written to: {run_dir}")


if __name__ == "__main__":
    main()
