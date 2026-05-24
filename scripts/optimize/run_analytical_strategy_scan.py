from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

SSF_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SSF_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.strategy_search import enumerate_legal_specs, exhaustive_search


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run analytical exhaustive scan over legal HybridStrategySpec candidates."
    )
    parser.add_argument(
        "--p-rules",
        nargs="+",
        type=float,
        default=[0.5 + 0.05 * i for i in range(11)],
        help="Rule probabilities to evaluate.",
    )
    parser.add_argument(
        "--n2-values",
        nargs="+",
        type=int,
        default=[4, 8],
        help="n2 values to scan.",
    )
    parser.add_argument(
        "--k-box-values",
        nargs="+",
        type=int,
        default=[1, 3, 7],
        help="k_box values to scan.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        type=str,
        default=["majority", "pyramid", "horizontal", "vertical"],
        choices=["majority", "pyramid", "horizontal", "vertical"],
        help="Strategy families to include.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k ranked rows per p_rule.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write JSON scan results.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress table output.")
    return parser.parse_args(argv)


def _print_ranked_table(p_rule: float, ranked_rows: list[dict[str, object]]) -> None:
    print(f"p_rule={p_rule}")
    print("rank | success | candidate")
    for row in ranked_rows:
        print(f"{row['rank']} | {row['success']:.12g} | {row['candidate_name']}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    specs = enumerate_legal_specs(
        n2_values=[int(v) for v in args.n2_values],
        k_box_values=[int(v) for v in args.k_box_values],
        families=[str(v) for v in args.families],
    )

    per_rule_results: list[dict[str, object]] = []
    for p_rule in args.p_rules:
        ranked = exhaustive_search(
            specs=specs,
            p_rule=float(p_rule),
            evaluation_mode="analytical",
            top_k=args.top_k,
        )

        ranked_rows: list[dict[str, object]] = []
        for idx, item in enumerate(ranked, start=1):
            ranked_rows.append(
                {
                    "rank": idx,
                    "candidate_name": item.candidate_name,
                    "success": float(item.success),
                    "p_rule": float(item.p_rule),
                    "evaluation_mode": item.evaluation_mode,
                    "spec": asdict(item.spec),
                }
            )

        per_rule_results.append(
            {
                "p_rule": float(p_rule),
                "ranked_results": ranked_rows,
            }
        )

        if not args.quiet:
            _print_ranked_table(float(p_rule), ranked_rows)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameters": {
                "p_rules": [float(v) for v in args.p_rules],
                "n2_values": [int(v) for v in args.n2_values],
                "k_box_values": [int(v) for v in args.k_box_values],
                "families": [str(v) for v in args.families],
                "top_k": int(args.top_k),
                "evaluation_mode": "analytical",
                "quiet": bool(args.quiet),
            },
            "results": per_rule_results,
        }
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
