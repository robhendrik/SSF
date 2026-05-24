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
from ssf.strategy_optimizer import BeamSearchConfig, beam_search_optimizer_with_summaries
from ssf.strategy_search import enumerate_legal_specs


DEFAULT_P_RULE = 0.8
DEFAULT_N2_VALUES = [4, 8]
DEFAULT_K_BOX_VALUES = [1, 3, 7]
DEFAULT_FAMILIES = ["majority", "pyramid", "horizontal", "vertical"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run backend-agnostic beam search strategy optimizer over legal HybridStrategySpec candidates."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["analytical", "dense_exhaustive", "dense_mc", "procedural_mc"],
    )
    parser.add_argument("--p-rule", type=float, default=DEFAULT_P_RULE)
    parser.add_argument("--n2-values", nargs="+", type=int, default=DEFAULT_N2_VALUES)
    parser.add_argument("--k-box-values", nargs="+", type=int, default=DEFAULT_K_BOX_VALUES)
    parser.add_argument(
        "--families",
        nargs="+",
        type=str,
        default=DEFAULT_FAMILIES,
        choices=["majority", "pyramid", "horizontal", "vertical"],
    )
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--mutations-per-candidate", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
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


def _print_generation_summaries(generation_summaries: list[dict[str, object]]) -> None:
    print("generation summary:")
    print("generation | evaluated | beam_best | candidate")
    for row in generation_summaries:
        print(
            f"{row['generation']} | {row['evaluated']} | {float(row['beam_best']):.4f} | {row['candidate']}"
        )


def _print_final_results(rows: list[dict[str, object]]) -> None:
    print("Final table:")
    print("rank | success | candidate | generation | parent | cache_hit")
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        print(
            f"{row['rank']} | {float(row['success']):.4f} | {row['candidate_name']} | "
            f"{row['generation']} | {row['parent_name']} | {metadata.get('cache_hit')}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    initial_specs = enumerate_legal_specs(
        n2_values=[int(v) for v in args.n2_values],
        k_box_values=[int(v) for v in args.k_box_values],
        families=[str(v) for v in args.families],
    )

    cache: EvaluationCache | None = None
    policy: CachePolicy | None = None

    if args.cache_enabled:
        cache = EvaluationCache(Path(args.cache_dir))
        cache.load()
        policy = CachePolicy(
            enabled=True,
            read=not bool(args.force_recompute),
            write=True,
            persist_analytical=bool(args.persist_analytical),
            cache_dir=Path(args.cache_dir),
        )

    config = BeamSearchConfig(
        p_rule=float(args.p_rule),
        evaluation_mode=str(args.mode),
        beam_width=int(args.beam_width),
        generations=int(args.generations),
        mutations_per_candidate=int(args.mutations_per_candidate),
        top_k=int(args.top_k),
        seed=int(args.seed),
        cache_policy=policy,
    )

    final_results, generation_summaries = beam_search_optimizer_with_summaries(
        initial_specs=initial_specs,
        n2_values=[int(v) for v in args.n2_values],
        k_box_values=[int(v) for v in args.k_box_values],
        families=[str(v) for v in args.families],
        config=config,
        cache=cache,
    )

    serialized_results: list[dict[str, object]] = []
    for row in final_results:
        serialized_results.append(
            {
                "rank": int(row.rank),
                "success": float(row.success),
                "candidate_name": row.candidate_name,
                "generation": int(row.generation),
                "parent_name": row.parent_name,
                "evaluation_mode": row.evaluation_mode,
                "metadata": dict(row.metadata),
                "spec": asdict(row.spec),
            }
        )

    if not args.quiet:
        _print_generation_summaries(generation_summaries)
        _print_final_results(serialized_results)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameters": {
                "mode": str(args.mode),
                "p_rule": float(args.p_rule),
                "n2_values": [int(v) for v in args.n2_values],
                "k_box_values": [int(v) for v in args.k_box_values],
                "families": [str(v) for v in args.families],
                "beam_width": int(args.beam_width),
                "generations": int(args.generations),
                "mutations_per_candidate": int(args.mutations_per_candidate),
                "top_k": int(args.top_k),
                "seed": int(args.seed),
                "cache_enabled": bool(args.cache_enabled),
                "cache_dir": str(Path(args.cache_dir)),
                "persist_analytical": bool(args.persist_analytical),
                "force_recompute": bool(args.force_recompute),
                "quiet": bool(args.quiet),
            },
            "generation_summaries": generation_summaries,
            "final_results": serialized_results,
        }
        args.json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
