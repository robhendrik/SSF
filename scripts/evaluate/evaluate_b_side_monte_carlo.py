from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SSF_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SSF_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.bob_mc import evaluate_strategy_mc
from ssf.bob_mc_procedural import evaluate_strategy_mc_procedural
from ssf.strategy_io import load_manifest, load_strategy


DEFAULT_FIXTURE_DIR = SSF_DIR / "fixtures" / "persistent" / "a_strategies"
DEFAULT_OUTPUT_BASE = SSF_DIR / "results" / "monte_carlo"
DEFAULT_P_RULES = (0.5, 0.65, 0.77, 1.0)
DEFAULT_SEEDS = (0,)
DEFAULT_SAMPLES = 100_000


@dataclass
class StrategyRecord:
    strategy: Dict[str, Any]
    strategy_name: str
    family: str
    boxes_used: Optional[int]
    fixture_path: Path


class DenseFixtureProceduralAdapter:
    def __init__(self, strategy_dict: Dict[str, Any]):
        self.n2 = int(strategy_dict["n2"])
        self.k_box = int(strategy_dict["k_box"])
        self._f_tables = list(strategy_dict["f_tables"])
        comm = strategy_dict.get("comm")
        if comm is None:
            comm = strategy_dict.get("comm_table")
        self._comm = comm

    def measure_a(self, box_idx: int, field: int, out_a_prefix: int) -> int:
        return int(self._f_tables[box_idx][field, out_a_prefix])

    def comm(self, field: int, out_a: int) -> int:
        return int(self._comm[field, out_a])


def _normalize_family(name: str) -> str:
    lower = name.lower()
    if lower.startswith("majority"):
        return "majority"
    if lower.startswith("pyramid"):
        return "pyramid"
    if lower.startswith("hybrid"):
        return "hybrid"
    return "unknown"


def _parse_float_list(raw: Sequence[float] | Sequence[str] | None, default: Sequence[float]) -> List[float]:
    if raw is None:
        return [float(v) for v in default]
    return [float(v) for v in raw]


def _parse_int_list(raw: Sequence[int] | None, default: Sequence[int]) -> List[int]:
    if raw is None:
        return [int(v) for v in default]
    return [int(v) for v in raw]


def _parse_box_limits(raw: Sequence[int] | None) -> List[Optional[int]]:
    if raw is None:
        return [None]
    if len(raw) == 0:
        return [None]
    return [int(v) for v in raw]


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
        filtered = [record for record in filtered if needle in record.strategy_name.lower()]

    if max_strategies is not None and max_strategies >= 0:
        filtered = filtered[:max_strategies]

    return filtered


def _timestamp_dir(base_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "none"
        if isinstance(value, float):
            return f"{value:.12g}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def _build_summary_by_seed(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for seed in sorted(grouped):
        seed_rows = grouped[seed]
        successes = [float(row["success"]) for row in seed_rows]
        runtimes = [float(row["runtime_seconds"]) for row in seed_rows]
        summary_rows.append(
            {
                "seed": seed,
                "row_count": len(seed_rows),
                "mean_success": sum(successes) / len(successes),
                "mean_runtime_seconds": sum(runtimes) / len(runtimes),
                "min_success": min(successes),
                "max_success": max(successes),
            }
        )
    return summary_rows


def evaluate_records(
    records: List[StrategyRecord],
    p_rules: List[float],
    box_limits: List[Optional[int]],
    seeds: List[int],
    subset_policy: str,
    samples: int,
    procedural: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for record in records:
        n2 = int(record.strategy["n2"])
        k_box = int(record.strategy["k_box"])

        for p_rule in p_rules:
            for box_limit in box_limits:
                for seed in seeds:
                    t0 = time.perf_counter()
                    if procedural:
                        result = evaluate_strategy_mc_procedural(
                            DenseFixtureProceduralAdapter(record.strategy),
                            p_rule=float(p_rule),
                            box_limit=box_limit,
                            subset_policy=subset_policy,
                            n_train_samples=int(samples),
                            n_eval_samples=int(samples),
                            seed=int(seed),
                            record_time=True,
                        )
                        runtime_seconds = (
                            float(result.elapsed_seconds)
                            if result.elapsed_seconds is not None
                            else float(time.perf_counter() - t0)
                        )
                    else:
                        result = evaluate_strategy_mc(
                            record.strategy,
                            p_rule=float(p_rule),
                            box_limit=box_limit,
                            subset_policy=subset_policy,
                            n_train_samples=int(samples),
                            n_eval_samples=int(samples),
                            confidence=0.95,
                            seed=int(seed),
                        )
                        runtime_seconds = float(time.perf_counter() - t0)

                    rows.append(
                        {
                            "strategy_name": record.strategy_name,
                            "family": record.family,
                            "n2": n2,
                            "k_box": k_box,
                            "boxes_used": record.boxes_used,
                            "p_rule": float(p_rule),
                            "box_limit": box_limit,
                            "subset_policy": subset_policy,
                            "samples": int(samples),
                            "seed": int(seed),
                            "success": float(result.success),
                            "ci_low": float(result.ci_low),
                            "ci_high": float(result.ci_high),
                            "stderr": float(result.stderr),
                            "runtime_seconds": float(runtime_seconds),
                            "strategy_file": record.fixture_path.name,
                        }
                    )

    return rows


def write_outputs(out_dir: Path, rows: List[Dict[str, Any]], args_dict: Dict[str, Any]) -> None:
    columns = [
        "strategy_name",
        "family",
        "n2",
        "k_box",
        "boxes_used",
        "p_rule",
        "box_limit",
        "subset_policy",
        "samples",
        "seed",
        "success",
        "ci_low",
        "ci_high",
        "stderr",
        "runtime_seconds",
        "strategy_file",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = out_dir / "results.md"
    md_path.write_text(_to_markdown_table(rows, columns), encoding="utf-8")

    summary_by_seed = _build_summary_by_seed(rows)
    json_path = out_dir / "results.json"
    json_path.write_text(
        json.dumps(
            {
                "run_dir": str(out_dir),
                "row_count": len(rows),
                "args": args_dict,
                "rows": rows,
                "summary_by_seed": summary_by_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if len(args_dict.get("seeds", [])) > 1:
        summary_csv = out_dir / "summary_by_seed.csv"
        summary_columns = [
            "seed",
            "row_count",
            "mean_success",
            "mean_runtime_seconds",
            "min_success",
            "max_success",
        ]
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_columns)
            writer.writeheader()
            for row in summary_by_seed:
                writer.writerow(row)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed A-side fixtures with Monte Carlo Bob-side optimization."
    )
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--p-rules",
        type=float,
        nargs="+",
        default=list(DEFAULT_P_RULES),
        help="One or more PR-box rule probabilities.",
    )
    parser.add_argument(
        "--box-limits",
        type=int,
        nargs="*",
        default=None,
        help="Zero or more box limits; omit to use each strategy's full k_box.",
    )
    parser.add_argument(
        "--subset-policy",
        type=str,
        default="up_to",
        choices=["up_to", "exact"],
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-strategies", type=int, default=None)
    parser.add_argument("--strategy-filter", type=str, default=None)
    parser.add_argument("--procedural", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.manifest is not None:
        records = _load_from_manifest(args.manifest)
    else:
        records = _load_from_fixture_dir(args.fixture_dir)

    records = _filter_records(records, args.strategy_filter, args.max_strategies)
    if not records:
        raise ValueError("No strategies selected for evaluation.")

    p_rules = _parse_float_list(args.p_rules, DEFAULT_P_RULES)
    box_limits = _parse_box_limits(args.box_limits)
    seeds = _parse_int_list(args.seeds, DEFAULT_SEEDS)
    samples = int(args.samples)

    if args.quick:
        records = records[:1]
        p_rules = p_rules[:1]
        box_limits = box_limits[:1]
        seeds = seeds[:1]
        samples = min(samples, 1000)

    if args.output_dir is None:
        out_dir = _timestamp_dir(DEFAULT_OUTPUT_BASE)
    else:
        out_dir = args.output_dir
        if args.overwrite and out_dir.exists():
            for file_name in ("results.csv", "results.json", "results.md", "summary_by_seed.csv"):
                file_path = out_dir / file_name
                if file_path.exists():
                    file_path.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluate_records(
        records=records,
        p_rules=p_rules,
        box_limits=box_limits,
        seeds=seeds,
        subset_policy=args.subset_policy,
        samples=samples,
        procedural=args.procedural,
    )

    write_outputs(
        out_dir=out_dir,
        rows=rows,
        args_dict={
            "fixture_dir": str(args.fixture_dir) if args.fixture_dir is not None else None,
            "manifest": str(args.manifest) if args.manifest is not None else None,
            "output_dir": str(out_dir),
            "p_rules": p_rules,
            "box_limits": box_limits,
            "subset_policy": args.subset_policy,
            "samples": samples,
            "seeds": seeds,
            "max_strategies": args.max_strategies,
            "strategy_filter": args.strategy_filter,
            "procedural": args.procedural,
            "quick": args.quick,
            "overwrite": args.overwrite,
        },
    )

    print(f"Evaluated {len(rows)} rows. Results written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())