"""Validate agreement across canonical evaluator backends on tiny reference specs.

Architectural role:
- integration check for strategy_evaluation dispatcher backends
- confirms analytical/dense/procedural paths stay aligned within configured tolerances

Invariants:
- spec identities are canonical HybridStrategySpec values
- comparisons use evaluate_candidate mode dispatch only

Failure behavior:
- required comparison tolerance failures produce non-zero process exit
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure local src package is importable when script is run directly from repo root.
SSF_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SSF_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name
from ssf.strategy_evaluation import EvaluationRequest, StrategyCandidate, evaluate_candidate


DEFAULT_P_RULES = [0.5, 0.8, 1.0]


@dataclass(frozen=True)
class ComparisonOutcome:
    """One backend comparison result row with pass/fail semantics."""

    name: str
    lhs: str
    rhs: str
    diff: float
    tolerance: float | None
    required: bool
    passed: bool
    warning_only: bool = False
    message: str | None = None


def _build_tiny_specs() -> list[HybridStrategySpec]:
    return [
        HybridStrategySpec(family="majority", n2=4, k_box=3, tie_bandwidth=0),
        HybridStrategySpec(family="pyramid", n2=4, k_box=3),
        HybridStrategySpec(
            family="horizontal",
            order="maj_pyr",
            n2=4,
            k_box=3,
            split=2,
            tie_bandwidth=0,
        ),
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options controlling backend comparison checks and tolerances."""
    parser = argparse.ArgumentParser(
        description="Compare strategy evaluation backends via unified evaluator dispatcher."
    )
    parser.add_argument(
        "--p-rules",
        nargs="+",
        type=float,
        default=DEFAULT_P_RULES,
        help="Rule probabilities to evaluate (default: 0.5 0.8 1.0).",
    )
    parser.add_argument(
        "--spec-set",
        type=str,
        choices=["tiny"],
        default="tiny",
        help="Named canonical spec set to validate.",
    )
    parser.add_argument(
        "--n-train-samples",
        type=int,
        default=10_000,
        help="MC training samples for dense_mc/procedural_mc.",
    )
    parser.add_argument(
        "--n-eval-samples",
        type=int,
        default=20_000,
        help="MC evaluation samples for dense_mc/procedural_mc.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for MC evaluators.")
    parser.add_argument(
        "--mc-tolerance",
        type=float,
        default=0.03,
        help="Maximum allowed absolute difference for MC comparisons.",
    )
    parser.add_argument(
        "--exact-tolerance",
        type=float,
        default=1e-12,
        help="Maximum allowed absolute difference for exact comparisons.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write JSON report.",
    )
    return parser.parse_args(argv)


def _specs_for_set(spec_set: str) -> list[HybridStrategySpec]:
    if spec_set == "tiny":
        return _build_tiny_specs()
    raise ValueError(f"Unsupported spec-set: {spec_set}")


def _evaluate_all_modes(
    spec: HybridStrategySpec,
    p_rule: float,
    n_train_samples: int,
    n_eval_samples: int,
    seed: int,
) -> dict[str, Any]:
    candidate = StrategyCandidate(spec=spec, source="compare_strategy_evaluation_backends")
    modes = ["analytical", "dense_exhaustive", "dense_mc", "procedural_mc"]

    results: dict[str, Any] = {}
    for mode in modes:
        req = EvaluationRequest(
            candidate=candidate,
            p_rule=p_rule,
            mode=mode,
            n_train_samples=n_train_samples,
            n_eval_samples=n_eval_samples,
            seed=seed,
        )
        results[mode] = evaluate_candidate(req)
    return results


def _compare_row(
    mode_results: dict[str, Any],
    mc_tolerance: float,
    exact_tolerance: float,
) -> tuple[list[ComparisonOutcome], list[dict[str, Any]]]:
    analytical = mode_results["analytical"]
    dense_exhaustive = mode_results["dense_exhaustive"]
    dense_mc = mode_results["dense_mc"]
    procedural_mc = mode_results["procedural_mc"]

    outcomes: list[ComparisonOutcome] = []

    diff_exact = abs(float(analytical.success) - float(dense_exhaustive.success))
    if analytical.exact and dense_exhaustive.exact:
        outcomes.append(
            ComparisonOutcome(
                name="analytical_vs_dense_exhaustive",
                lhs="analytical",
                rhs="dense_exhaustive",
                diff=diff_exact,
                tolerance=exact_tolerance,
                required=True,
                passed=diff_exact <= exact_tolerance,
            )
        )
    else:
        outcomes.append(
            ComparisonOutcome(
                name="analytical_vs_dense_exhaustive",
                lhs="analytical",
                rhs="dense_exhaustive",
                diff=diff_exact,
                tolerance=None,
                required=False,
                passed=True,
                warning_only=True,
                message="Analytical result not exact; skipping required exact comparison.",
            )
        )

    def add_mc_check(name: str, lhs: str, rhs: str, lhs_value: float, rhs_value: float) -> None:
        diff = abs(lhs_value - rhs_value)
        outcomes.append(
            ComparisonOutcome(
                name=name,
                lhs=lhs,
                rhs=rhs,
                diff=diff,
                tolerance=mc_tolerance,
                required=True,
                passed=diff <= mc_tolerance,
            )
        )

    add_mc_check(
        "dense_mc_vs_dense_exhaustive",
        "dense_mc",
        "dense_exhaustive",
        float(dense_mc.success),
        float(dense_exhaustive.success),
    )
    add_mc_check(
        "procedural_mc_vs_dense_exhaustive",
        "procedural_mc",
        "dense_exhaustive",
        float(procedural_mc.success),
        float(dense_exhaustive.success),
    )
    add_mc_check(
        "dense_mc_vs_procedural_mc",
        "dense_mc",
        "procedural_mc",
        float(dense_mc.success),
        float(procedural_mc.success),
    )

    failures = [
        {
            "comparison": outcome.name,
            "lhs": outcome.lhs,
            "rhs": outcome.rhs,
            "diff": outcome.diff,
            "tolerance": outcome.tolerance,
            "message": outcome.message,
        }
        for outcome in outcomes
        if outcome.required and not outcome.passed
    ]

    return outcomes, failures


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        "spec_name | p_rule | analytical | dense_exhaustive | dense_mc | procedural_mc | status"
    )
    print(header)
    for row in rows:
        print(
            " | ".join(
                [
                    row["spec_name"],
                    _format_float(float(row["p_rule"])),
                    _format_float(float(row["analytical"])),
                    _format_float(float(row["dense_exhaustive"])),
                    _format_float(float(row["dense_mc"])),
                    _format_float(float(row["procedural_mc"])),
                    row["status"],
                ]
            )
        )


def main(argv: list[str] | None = None) -> int:
    """Run backend consistency checks and return 1 when required checks fail."""
    args = _parse_args(argv)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    specs = _specs_for_set(args.spec_set)
    for spec in specs:
        spec_name = candidate_name(spec)
        for p_rule in args.p_rules:
            mode_results = _evaluate_all_modes(
                spec=spec,
                p_rule=float(p_rule),
                n_train_samples=args.n_train_samples,
                n_eval_samples=args.n_eval_samples,
                seed=args.seed,
            )
            outcomes, row_failures = _compare_row(
                mode_results=mode_results,
                mc_tolerance=args.mc_tolerance,
                exact_tolerance=args.exact_tolerance,
            )

            required_ok = all(o.passed for o in outcomes if o.required)
            has_warning = any(o.warning_only for o in outcomes)
            status = "PASS"
            if not required_ok:
                status = "FAIL"
            elif has_warning:
                status = "WARN"

            row = {
                "spec_name": spec_name,
                "spec": asdict(spec),
                "p_rule": float(p_rule),
                "analytical": float(mode_results["analytical"].success),
                "dense_exhaustive": float(mode_results["dense_exhaustive"].success),
                "dense_mc": float(mode_results["dense_mc"].success),
                "procedural_mc": float(mode_results["procedural_mc"].success),
                "status": status,
                "comparisons": [asdict(outcome) for outcome in outcomes],
            }
            rows.append(row)

            for failure in row_failures:
                failure_with_context = {
                    "spec_name": spec_name,
                    "p_rule": float(p_rule),
                    **failure,
                }
                failures.append(failure_with_context)

    if not args.quiet:
        _print_table(rows)
        if failures:
            print(f"Required comparison failures: {len(failures)}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameters": {
                "p_rules": [float(v) for v in args.p_rules],
                "spec_set": args.spec_set,
                "n_train_samples": args.n_train_samples,
                "n_eval_samples": args.n_eval_samples,
                "seed": args.seed,
                "mc_tolerance": args.mc_tolerance,
                "exact_tolerance": args.exact_tolerance,
                "quiet": args.quiet,
            },
            "rows": rows,
            "failures": failures,
        }
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
