"""Backend-agnostic exhaustive and beam search optimizers over canonical strategy candidates."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from .hybrid_strategy_spec import HybridStrategySpec, candidate_name, validate_spec
from .strategy_cache import CachePolicy, EvaluationCache
from .strategy_evaluation import EvaluationRequest, StrategyCandidate, evaluate_candidate
from .strategy_search import enumerate_legal_specs


@dataclass(frozen=True)
class OptimizationResult:
    spec: HybridStrategySpec
    candidate_name: str
    success: float
    rank: int
    p_rule: float
    evaluation_mode: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class BeamSearchConfig:
    p_rule: float
    evaluation_mode: str
    beam_width: int = 10
    generations: int = 5
    mutations_per_candidate: int = 5
    top_k: int = 10
    seed: int = 0
    cache_policy: CachePolicy | None = None


@dataclass(frozen=True)
class BeamSearchResult:
    spec: HybridStrategySpec
    candidate_name: str
    success: float
    rank: int
    generation: int
    parent_name: str | None
    evaluation_mode: str
    metadata: dict[str, object]


def _value_index(value: int, values: list[int]) -> int | None:
    try:
        return sorted(set(int(v) for v in values)).index(int(value))
    except ValueError:
        return None


def mutate_spec(
    spec: HybridStrategySpec,
    *,
    n2_values: list[int],
    k_box_values: list[int],
    families: list[str],
    rng: random.Random,
) -> list[HybridStrategySpec]:
    validated = validate_spec(spec)
    current_name = candidate_name(validated)

    legal = enumerate_legal_specs(
        n2_values=[int(v) for v in n2_values],
        k_box_values=[int(v) for v in k_box_values],
        families=[str(v) for v in families],
    )

    current_n2_idx = _value_index(validated.n2, n2_values)
    current_k_idx = _value_index(validated.k_box, k_box_values)

    neighbors: list[HybridStrategySpec] = []
    seen: set[str] = set()

    for candidate in legal:
        cand_name = candidate_name(candidate)
        if cand_name == current_name or cand_name in seen:
            continue

        changed_dims = 0
        for attr in (
            "family",
            "k_box",
            "split",
            "order",
            "depth_s",
            "tie_bandwidth",
            "n2",
        ):
            if getattr(candidate, attr) != getattr(validated, attr):
                changed_dims += 1

        if changed_dims == 0:
            continue

        cand_n2_idx = _value_index(candidate.n2, n2_values)
        cand_k_idx = _value_index(candidate.k_box, k_box_values)

        if current_n2_idx is not None and cand_n2_idx is not None:
            if abs(cand_n2_idx - current_n2_idx) > 1:
                continue
        if current_k_idx is not None and cand_k_idx is not None:
            if abs(cand_k_idx - current_k_idx) > 1:
                continue

        if validated.family == candidate.family == "horizontal":
            if validated.split is not None and candidate.split is not None:
                if abs(candidate.split - validated.split) > 1:
                    continue

        if validated.family == candidate.family == "vertical":
            if validated.depth_s is not None and candidate.depth_s is not None:
                if abs(candidate.depth_s - validated.depth_s) > 1:
                    continue

        if validated.tie_bandwidth is not None and candidate.tie_bandwidth is not None:
            if abs(candidate.tie_bandwidth - validated.tie_bandwidth) > 2:
                continue

        seen.add(cand_name)
        neighbors.append(candidate)

    neighbors.sort(key=candidate_name)
    rng.shuffle(neighbors)
    return neighbors


def _to_beam_row(
    *,
    spec: HybridStrategySpec,
    generation: int,
    parent_name: str | None,
    config: BeamSearchConfig,
    cache: EvaluationCache | None,
) -> BeamSearchResult:
    validated = validate_spec(spec)
    cand_name = candidate_name(validated)
    request = EvaluationRequest(
        candidate=StrategyCandidate(spec=validated, source="strategy_optimizer_beam"),
        p_rule=float(config.p_rule),
        mode=str(config.evaluation_mode),
    )
    result = evaluate_candidate(request, cache_policy=config.cache_policy, cache=cache)

    metadata: dict[str, object] = {
        "parent_name": parent_name,
        "exact": bool(result.exact),
        "analytical": bool(result.mode == "analytical"),
        "generation_created": int(generation),
    }
    if "cache_hit" in result.metadata:
        metadata["cache_hit"] = bool(result.metadata["cache_hit"])

    return BeamSearchResult(
        spec=validated,
        candidate_name=cand_name,
        success=float(result.success),
        rank=0,
        generation=int(generation),
        parent_name=parent_name,
        evaluation_mode=str(config.evaluation_mode),
        metadata=metadata,
    )


def _rank_beam_rows(rows: list[BeamSearchResult]) -> list[BeamSearchResult]:
    sorted_rows = sorted(rows, key=lambda item: (-item.success, item.candidate_name))
    ranked: list[BeamSearchResult] = []
    for idx, row in enumerate(sorted_rows, start=1):
        ranked.append(
            BeamSearchResult(
                spec=row.spec,
                candidate_name=row.candidate_name,
                success=row.success,
                rank=idx,
                generation=row.generation,
                parent_name=row.parent_name,
                evaluation_mode=row.evaluation_mode,
                metadata=dict(row.metadata),
            )
        )
    return ranked


def beam_search_optimizer_with_summaries(
    *,
    initial_specs: list[HybridStrategySpec],
    n2_values: list[int],
    k_box_values: list[int],
    families: list[str],
    config: BeamSearchConfig,
    cache: EvaluationCache | None = None,
) -> tuple[list[BeamSearchResult], list[dict[str, Any]]]:
    if config.beam_width <= 0:
        raise ValueError("beam_width must be positive.")
    if config.generations < 0:
        raise ValueError("generations must be non-negative.")
    if config.mutations_per_candidate < 0:
        raise ValueError("mutations_per_candidate must be non-negative.")
    if config.top_k < 0:
        raise ValueError("top_k must be non-negative.")

    dedup_initial: dict[str, HybridStrategySpec] = {}
    for spec in initial_specs:
        try:
            validated = validate_spec(spec)
        except ValueError:
            continue
        dedup_initial[candidate_name(validated)] = validated

    if not dedup_initial:
        raise ValueError("No valid initial specs provided.")

    rng = random.Random(int(config.seed))

    all_rows: dict[str, BeamSearchResult] = {}
    generation_summaries: list[dict[str, Any]] = []

    evaluated_gen0 = 0
    for name in sorted(dedup_initial.keys()):
        row = _to_beam_row(
            spec=dedup_initial[name],
            generation=0,
            parent_name=None,
            config=config,
            cache=cache,
        )
        all_rows[name] = row
        evaluated_gen0 += 1

    ranked_all = _rank_beam_rows(list(all_rows.values()))
    beam_names = [row.candidate_name for row in ranked_all[: config.beam_width]]
    beam_best = ranked_all[0]
    generation_summaries.append(
        {
            "generation": 0,
            "evaluated": evaluated_gen0,
            "beam_best": float(beam_best.success),
            "candidate": beam_best.candidate_name,
        }
    )

    for generation in range(1, config.generations + 1):
        if not beam_names:
            break

        newly_evaluated = 0
        for parent_name in sorted(beam_names):
            parent_row = all_rows[parent_name]
            mutated = mutate_spec(
                parent_row.spec,
                n2_values=n2_values,
                k_box_values=k_box_values,
                families=families,
                rng=rng,
            )
            if config.mutations_per_candidate > 0:
                mutated = mutated[: config.mutations_per_candidate]
            else:
                mutated = []

            for child in mutated:
                child_name = candidate_name(child)
                if child_name in all_rows:
                    continue
                row = _to_beam_row(
                    spec=child,
                    generation=generation,
                    parent_name=parent_name,
                    config=config,
                    cache=cache,
                )
                all_rows[child_name] = row
                newly_evaluated += 1

        ranked_all = _rank_beam_rows(list(all_rows.values()))
        beam_names = [row.candidate_name for row in ranked_all[: config.beam_width]]
        if not ranked_all:
            break

        beam_best = ranked_all[0]
        generation_summaries.append(
            {
                "generation": generation,
                "evaluated": newly_evaluated,
                "beam_best": float(beam_best.success),
                "candidate": beam_best.candidate_name,
            }
        )

        if newly_evaluated == 0:
            break

    final_ranked = _rank_beam_rows(list(all_rows.values()))
    return final_ranked[: config.top_k], generation_summaries


def beam_search_optimizer(
    *,
    initial_specs: list[HybridStrategySpec],
    n2_values: list[int],
    k_box_values: list[int],
    families: list[str],
    config: BeamSearchConfig,
    cache: EvaluationCache | None = None,
) -> list[BeamSearchResult]:
    results, _ = beam_search_optimizer_with_summaries(
        initial_specs=initial_specs,
        n2_values=n2_values,
        k_box_values=k_box_values,
        families=families,
        config=config,
        cache=cache,
    )
    return results


def exhaustive_strategy_optimizer(
    *,
    specs: list[HybridStrategySpec],
    p_rule: float,
    evaluation_mode: str,
    top_k: int = 10,
    cache_policy: CachePolicy | None = None,
    cache: EvaluationCache | None = None,
) -> list[OptimizationResult]:
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")

    evaluated: list[tuple[HybridStrategySpec, str, float, dict[str, object]]] = []

    for spec in specs:
        validated = validate_spec(spec)
        cand_name = candidate_name(validated)
        request = EvaluationRequest(
            candidate=StrategyCandidate(spec=validated, source="strategy_optimizer"),
            p_rule=float(p_rule),
            mode=evaluation_mode,
        )
        result = evaluate_candidate(request, cache_policy=cache_policy, cache=cache)

        metadata: dict[str, object] = {
            "exact": bool(result.exact),
            "analytical": bool(result.mode == "analytical"),
        }
        if "cache_hit" in result.metadata:
            metadata["cache_hit"] = bool(result.metadata["cache_hit"])

        evaluated.append((validated, cand_name, float(result.success), metadata))

    evaluated.sort(key=lambda item: (-item[2], item[1]))

    trimmed = evaluated[:top_k]
    ranked: list[OptimizationResult] = []
    for idx, (spec, cand_name, success, metadata) in enumerate(trimmed, start=1):
        ranked.append(
            OptimizationResult(
                spec=spec,
                candidate_name=cand_name,
                success=success,
                rank=idx,
                p_rule=float(p_rule),
                evaluation_mode=str(evaluation_mode),
                metadata=metadata,
            )
        )

    return ranked


def optimize_over_parameter_grid(
    *,
    p_rules: list[float],
    n2_values: list[int],
    k_box_values: list[int],
    families: list[str],
    evaluation_mode: str,
    top_k: int,
    cache_policy: CachePolicy | None = None,
    cache: EvaluationCache | None = None,
) -> dict[float, list[OptimizationResult]]:
    specs = enumerate_legal_specs(
        n2_values=[int(v) for v in n2_values],
        k_box_values=[int(v) for v in k_box_values],
        families=[str(v) for v in families],
    )

    per_rule: dict[float, list[OptimizationResult]] = {}
    for p_rule in p_rules:
        value = float(p_rule)
        per_rule[value] = exhaustive_strategy_optimizer(
            specs=specs,
            p_rule=value,
            evaluation_mode=evaluation_mode,
            top_k=top_k,
            cache_policy=cache_policy,
            cache=cache,
        )

    return per_rule
