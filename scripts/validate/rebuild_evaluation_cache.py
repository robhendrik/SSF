from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ensure local src package is importable when script is run directly from repo root.
SSF_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SSF_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_spec import candidate_name
from ssf.strategy_cache import CachePolicy, EvaluationCache
from ssf.strategy_evaluation import EvaluationRequest, StrategyCandidate, evaluate_candidate
from ssf.strategy_search import enumerate_legal_specs


DEFAULT_P_RULES = [0.5, 0.8, 1.0]
DEFAULT_N2_VALUES = [4, 8]
DEFAULT_K_BOX_VALUES = [1, 3, 7]
DEFAULT_FAMILIES = ["majority", "pyramid", "horizontal", "vertical"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep legal strategy specs and refresh evaluation cache using unified evaluator dispatcher."
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
    parser.add_argument("--families", nargs="+", type=str, default=DEFAULT_FAMILIES)
    parser.add_argument("--box-limit", type=int, default=None)
    parser.add_argument("--subset-policy", type=str, default="up_to")
    parser.add_argument("--n-train-samples", type=int, default=10_000)
    parser.add_argument("--n-eval-samples", type=int, default=20_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("work") / "fixture_bank" / "evaluations",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--persist-analytical", action="store_true")
    parser.add_argument("--min-trust-level", type=int, default=None)
    parser.add_argument("--max-strategies", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def _jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.mode == "analytical" and not args.persist_analytical:
        print("Analytical results are not persisted unless --persist-analytical is set.")

    specs = enumerate_legal_specs(
        n2_values=[int(v) for v in args.n2_values],
        k_box_values=[int(v) for v in args.k_box_values],
        families=[str(v) for v in args.families],
    )

    if args.max_strategies is not None:
        if args.max_strategies < 0:
            raise ValueError("--max-strategies must be non-negative when provided.")
        specs = specs[: args.max_strategies]

    cache = EvaluationCache(Path(args.cache_dir))
    cache.load()
    initial_jsonl_lines = _jsonl_line_count(cache.jsonl_path)

    policy = CachePolicy(
        enabled=True,
        read=not bool(args.force_recompute),
        write=True,
        persist_analytical=bool(args.persist_analytical),
        min_trust_level=args.min_trust_level,
        cache_dir=Path(args.cache_dir),
    )

    evaluated_count = 0
    cache_hits = 0
    rows: list[dict[str, Any]] = []

    for spec in specs:
        spec_name = candidate_name(spec)
        for p_rule in args.p_rules:
            request = EvaluationRequest(
                candidate=StrategyCandidate(spec=spec, source="rebuild_evaluation_cache"),
                p_rule=float(p_rule),
                mode=args.mode,
                box_limit=args.box_limit,
                subset_policy=args.subset_policy,
                n_train_samples=args.n_train_samples,
                n_eval_samples=args.n_eval_samples,
                confidence=args.confidence,
                seed=args.seed,
            )
            result = evaluate_candidate(request, cache_policy=policy, cache=cache)

            evaluated_count += 1
            is_cache_hit = bool(result.metadata.get("cache_hit"))
            if is_cache_hit:
                cache_hits += 1

            rows.append(
                {
                    "candidate_name": spec_name,
                    "spec": asdict(spec),
                    "p_rule": float(p_rule),
                    "mode": args.mode,
                    "success": float(result.success),
                    "cache_hit": is_cache_hit,
                }
            )

            if not args.quiet:
                status = "HIT" if is_cache_hit else "EVAL"
                print(
                    f"[{status}] {spec_name} p_rule={float(p_rule):.6g} mode={args.mode} "
                    f"success={float(result.success):.6g}"
                )

    final_jsonl_lines = _jsonl_line_count(cache.jsonl_path)
    written_or_replaced = max(0, final_jsonl_lines - initial_jsonl_lines)
    recomputed_count = evaluated_count - cache_hits

    summary = {
        "mode": args.mode,
        "evaluated_count": evaluated_count,
        "cache_hits": cache_hits,
        "recomputed_count": recomputed_count,
        "records_written_or_replaced": written_or_replaced,
        "cache_dir": str(Path(args.cache_dir)),
        "spec_count": len(specs),
        "p_rules": [float(v) for v in args.p_rules],
        "force_recompute": bool(args.force_recompute),
        "persist_analytical": bool(args.persist_analytical),
    }

    if not args.quiet:
        print("Summary")
        print(f"evaluated count: {evaluated_count}")
        print(f"cache hits: {cache_hits}")
        print(f"recomputed count: {recomputed_count}")
        print(f"records written/replaced: {written_or_replaced}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameters": {
                "mode": args.mode,
                "p_rules": [float(v) for v in args.p_rules],
                "n2_values": [int(v) for v in args.n2_values],
                "k_box_values": [int(v) for v in args.k_box_values],
                "families": [str(v) for v in args.families],
                "box_limit": args.box_limit,
                "subset_policy": args.subset_policy,
                "n_train_samples": args.n_train_samples,
                "n_eval_samples": args.n_eval_samples,
                "confidence": args.confidence,
                "seed": args.seed,
                "cache_dir": str(Path(args.cache_dir)),
                "force_recompute": bool(args.force_recompute),
                "persist_analytical": bool(args.persist_analytical),
                "min_trust_level": args.min_trust_level,
                "max_strategies": args.max_strategies,
                "quiet": bool(args.quiet),
            },
            "summary": summary,
            "results": rows,
        }
        args.json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
