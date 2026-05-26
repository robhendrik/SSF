"""Canonical backend-agnostic evaluation dispatcher for strategy candidates.

Architectural role:
- central entry point between spec/search/optimizer and evaluator backends
- normalizes analytical, dense exhaustive, dense MC, and procedural MC results
- coordinates with strategy_cache without leaking cache logic into evaluators

Invariants:
- candidate specs are validated before dispatch
- optimizers consume only EvaluationResult.success and metadata
- backend selection is mode-driven, not candidate-name-driven

Failure behavior:
- raises ValueError for invalid p_rule or unsupported mode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .bob_exhaustive import evaluate_strategy_exhaustive
from .bob_mc import evaluate_strategy_mc
from .bob_mc_procedural import evaluate_strategy_mc_procedural
from .hybrid_strategy_analytics import analytical_win_rate
from .hybrid_strategy_fixtures import build_strategy_fixture
from .hybrid_strategy_procedural import build_procedural_strategy
from .hybrid_strategy_spec import HybridStrategySpec, stable_key, validate_spec

EvaluationMode = Literal["analytical", "dense_exhaustive", "dense_mc", "procedural_mc"]


@dataclass(frozen=True)
class StrategyCandidate:
    """Canonical optimization candidate carrying a validated HybridStrategySpec."""

    spec: HybridStrategySpec
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRequest:
    """Immutable request for evaluator dispatch and optional cache lookup."""

    candidate: StrategyCandidate
    p_rule: float
    mode: EvaluationMode
    box_limit: int | None = None
    subset_policy: str = "up_to"
    n_train_samples: int = 20_000
    n_eval_samples: int = 50_000
    confidence: float = 0.95
    seed: int = 0


@dataclass(frozen=True)
class EvaluationResult:
    """Normalized evaluator output returned to search/optimizer/caching layers."""

    success: float
    mode: EvaluationMode
    exact: bool
    method: str
    notes: str = ""
    ci_low: float | None = None
    ci_high: float | None = None
    stderr: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluation_cache_key(request: EvaluationRequest) -> tuple:
    """Build canonical cache key components for one evaluation request.

    Invariants:
    - includes stable spec identity and evaluator settings that affect outcomes
    """
    spec = validate_spec(request.candidate.spec)
    return (
        stable_key(spec),
        str(request.mode),
        float(request.p_rule),
        request.box_limit,
        str(request.subset_policy),
        int(request.n_train_samples),
        int(request.n_eval_samples),
        float(request.confidence),
        int(request.seed),
    )


def _evaluate_candidate_uncached(request: EvaluationRequest) -> EvaluationResult:
    """Execute backend dispatch without reading/writing cache records."""
    spec = validate_spec(request.candidate.spec)
    p_rule = float(request.p_rule)
    if not (0.0 <= p_rule <= 1.0):
        raise ValueError(f"p_rule must be in [0, 1], got {request.p_rule}.")

    if request.mode == "analytical":
        analytical = analytical_win_rate(spec, p_rule)
        return EvaluationResult(
            success=float(analytical.success),
            mode=request.mode,
            exact=bool(analytical.exact),
            method=str(analytical.method),
            notes=str(analytical.notes),
        )

    if request.mode == "dense_exhaustive":
        fixture = build_strategy_fixture(spec)
        success = evaluate_strategy_exhaustive(
            fixture.as_strategy_dict(),
            p_rule,
            box_limit=request.box_limit,
            subset_policy=request.subset_policy,
        )
        return EvaluationResult(
            success=float(success),
            mode=request.mode,
            exact=True,
            method="exhaustive",
            notes="Dense exhaustive Bob evaluator.",
        )

    if request.mode == "dense_mc":
        fixture = build_strategy_fixture(spec)
        mc = evaluate_strategy_mc(
            fixture.as_strategy_dict(),
            p_rule,
            box_limit=request.box_limit,
            subset_policy=request.subset_policy,
            n_train_samples=request.n_train_samples,
            n_eval_samples=request.n_eval_samples,
            confidence=request.confidence,
            seed=request.seed,
        )
        return EvaluationResult(
            success=float(mc.success),
            mode=request.mode,
            exact=False,
            method="mc_dense",
            notes="Dense Monte Carlo Bob evaluator.",
            ci_low=float(mc.ci_low),
            ci_high=float(mc.ci_high),
            stderr=float(mc.stderr),
        )

    if request.mode == "procedural_mc":
        strategy = build_procedural_strategy(spec)
        mc = evaluate_strategy_mc_procedural(
            strategy,
            p_rule,
            box_limit=request.box_limit,
            subset_policy=request.subset_policy,
            n_train_samples=request.n_train_samples,
            n_eval_samples=request.n_eval_samples,
            confidence=request.confidence,
            seed=request.seed,
        )
        return EvaluationResult(
            success=float(mc.success),
            mode=request.mode,
            exact=False,
            method="mc_procedural",
            notes="Procedural Monte Carlo Bob evaluator.",
            ci_low=float(mc.ci_low),
            ci_high=float(mc.ci_high),
            stderr=float(mc.stderr),
        )

    raise ValueError(f"Unsupported evaluation mode: {request.mode}")


def evaluate_candidate(
    request: EvaluationRequest,
    cache_policy: "CachePolicy | None" = None,
    cache: "EvaluationCache | None" = None,
) -> EvaluationResult:
    """Evaluate one candidate via canonical dispatcher with optional cache policy.

    Architectural role:
    - single optimizer-facing API for all backends and cache behavior

    Failure behavior:
    - propagates validation/backend errors from request validation and evaluators
    """
    # Lazy import avoids circular dependency: strategy_cache imports strategy_evaluation.
    from .strategy_cache import CachePolicy, EvaluationCache, result_from_record

    if cache_policy is None or not cache_policy.enabled:
        return _evaluate_candidate_uncached(request)

    policy = cache_policy

    active_cache = cache
    if active_cache is None:
        if policy.cache_dir is not None:
            cache_dir = Path(policy.cache_dir)
        else:
            project_root = Path(__file__).resolve().parents[2]
            cache_dir = project_root / "work" / "fixture_bank" / "evaluations"
        active_cache = EvaluationCache(cache_dir)

    active_cache.load()

    if policy.read:
        cached_record = active_cache.lookup(request, policy)
        if cached_record is not None:
            cached_result = result_from_record(cached_record)
            return EvaluationResult(
                success=cached_result.success,
                mode=cached_result.mode,
                exact=cached_result.exact,
                method=cached_result.method,
                notes=cached_result.notes,
                ci_low=cached_result.ci_low,
                ci_high=cached_result.ci_high,
                stderr=cached_result.stderr,
                metadata={**cached_result.metadata, "cache_hit": True},
            )

    fresh_result = _evaluate_candidate_uncached(request)

    result_with_flag = EvaluationResult(
        success=fresh_result.success,
        mode=fresh_result.mode,
        exact=fresh_result.exact,
        method=fresh_result.method,
        notes=fresh_result.notes,
        ci_low=fresh_result.ci_low,
        ci_high=fresh_result.ci_high,
        stderr=fresh_result.stderr,
        metadata={**fresh_result.metadata, "cache_hit": False},
    )

    if policy.write:
        active_cache.upsert(request, result_with_flag, policy)

    return result_with_flag
