"""CLI wrapper for canonical exhaustive strategy optimizer over legal specs.

Architectural role:
- command-line front end for strategy_optimizer.optimize_over_parameter_grid
- keeps optimizer logic in src/ssf while exposing reproducible runs

Invariants:
- candidate generation uses canonical enumerate_legal_specs indirectly
- scoring flows through strategy_evaluation and optional cache policy

Failure behavior:
- argument or optimizer validation errors propagate as non-zero process exit
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

SSF_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SSF_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.strategy_cache import CachePolicy, EvaluationCache
from ssf.strategy_optimizer import optimize_over_parameter_grid


DEFAULT_P_RULES = [0.5, 0.7, 0.8, 0.9, 1.0]
DEFAULT_N2_VALUES = [4, 8]
DEFAULT_K_BOX_VALUES = [1, 3, 7]
DEFAULT_FAMILIES = ["majority", "pyramid", "horizontal", "vertical"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for exhaustive optimizer execution."""
    parser = argparse.ArgumentParser(
        description="Run exhaustive strategy optimizer over legal HybridStrategySpec candidates."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["analytical", "dense_exhaustive", "dense_mc", "procedural_mc"],
    )
    parser.add_argument("--p-rules", nargs="+", type=float, default=DEFAULT_P_RULES)
    parser.add_argument("--n2-values", nargs="+", type=int, default=DEFAULT_N2_VALUES)
    parser.add_argument("--k-box-values", nargs="+", type=int, default=DEFAULT_K_BOX_VALUES)
    parser.add_argument(
        "--families",
        nargs="+",
        type=str,
        default=DEFAULT_FAMILIES,
        choices=["majority", "pyramid", "horizontal", "vertical"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--box-limit",
        type=int,
        default=None,
        help="Restrict Bob-visible box outputs (dense_exhaustive only).",
    )
    parser.add_argument(
        "--subset-policy",
        type=str,
        default="up_to",
        choices=["up_to", "exact"],
        help="Subset restriction policy for box-limit (dense_exhaustive only).",
    )
    parser.add_argument("--cache-enabled", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("work") / "fixture_bank" / "evaluations",
    )
    parser.add_argument("--persist-analytical", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _print_ranked_table(p_rule: float, rows: list[dict[str, object]]) -> None:
    print(f"p_rule={p_rule}")
    print("rank | success | candidate | cache_hit")
    for row in rows:
        cache_hit = row["metadata"].get("cache_hit") if isinstance(row.get("metadata"), dict) else None
        print(
            f"{row['rank']} | {float(row['success']):.4f} | {row['candidate_name']} | {cache_hit}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run exhaustive optimizer and optionally persist JSON report."""
    args = _parse_args(argv)

    cache: EvaluationCache | None = None
    cache_policy: CachePolicy | None = None

    if args.cache_enabled:
        cache = EvaluationCache(Path(args.cache_dir))
        cache.load()
        cache_policy = CachePolicy(
            enabled=True,
            read=not bool(args.force_recompute),
            write=True,
            persist_analytical=bool(args.persist_analytical),
            cache_dir=Path(args.cache_dir),
        )

    results_by_p_rule = optimize_over_parameter_grid(
        p_rules=[float(v) for v in args.p_rules],
        n2_values=[int(v) for v in args.n2_values],
        k_box_values=[int(v) for v in args.k_box_values],
        families=[str(v) for v in args.families],
        evaluation_mode=str(args.mode),
        box_limit=args.box_limit,
        subset_policy=str(args.subset_policy),
        top_k=int(args.top_k),
        cache_policy=cache_policy,
        cache=cache,
    )

    serializable_results: list[dict[str, object]] = []
    cache_hit_count = 0
    total_ranked = 0

    for p_rule in sorted(results_by_p_rule.keys()):
        ranked_rows: list[dict[str, object]] = []
        for item in results_by_p_rule[p_rule]:
            row = {
                "rank": int(item.rank),
                "candidate_name": item.candidate_name,
                "success": float(item.success),
                "p_rule": float(item.p_rule),
                "evaluation_mode": item.evaluation_mode,
                "spec": asdict(item.spec),
                "metadata": dict(item.metadata),
            }
            ranked_rows.append(row)
            total_ranked += 1
            if bool(item.metadata.get("cache_hit", False)):
                cache_hit_count += 1

        serializable_results.append({"p_rule": float(p_rule), "ranked_results": ranked_rows})

        if not args.quiet:
            _print_ranked_table(float(p_rule), ranked_rows)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameters": {
                "mode": str(args.mode),
                "p_rules": [float(v) for v in args.p_rules],
                "n2_values": [int(v) for v in args.n2_values],
                "k_box_values": [int(v) for v in args.k_box_values],
                "families": [str(v) for v in args.families],
                "top_k": int(args.top_k),
                "box_limit": args.box_limit,
                "subset_policy": str(args.subset_policy),
                "cache_enabled": bool(args.cache_enabled),
                "cache_dir": str(Path(args.cache_dir)),
                "persist_analytical": bool(args.persist_analytical),
                "force_recompute": bool(args.force_recompute),
                "quiet": bool(args.quiet),
            },
            "results": serializable_results,
            "cache_statistics": {
                "cache_enabled": bool(args.cache_enabled),
                "cache_hits": int(cache_hit_count),
                "evaluated_ranked_candidates": int(total_ranked),
            },
        }
        args.json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
