"""Canonical legal-spec enumeration and exhaustive ranking utilities.

Architectural role:
- generates valid HybridStrategySpec candidates for optimizer front-ends
- provides exhaustive ranking through strategy_evaluation.evaluate_candidate

Invariants:
- all emitted specs pass validate_spec and required_boxes <= k_box
- no strategy-name shortcuts; ranking uses evaluator outputs only
"""

from __future__ import annotations

from dataclasses import dataclass

from .hybrid_strategy_spec import (
    HybridStrategySpec,
    candidate_name,
    required_boxes,
    validate_spec,
)
from .strategy_evaluation import EvaluationRequest, StrategyCandidate, evaluate_candidate


@dataclass(frozen=True)
class SearchResult:
    """One ranked search row linking canonical spec identity to success score."""

    spec: HybridStrategySpec
    candidate_name: str
    success: float
    p_rule: float
    evaluation_mode: str


def _tie_bandwidth_values(majority_len: int) -> list[int]:
    # Valid tie bandwidths are even and satisfy 0 <= b < majority_len // 2.
    return [b for b in range(0, majority_len // 2) if b % 2 == 0]


def enumerate_legal_specs(
    *,
    n2_values: list[int],
    k_box_values: list[int],
    families: list[str],
) -> list[HybridStrategySpec]:
    """Enumerate canonical legal specs over provided grid dimensions.

    Failure behavior:
    - invalid intermediate specs are skipped via validate_spec exceptions
    """
    dedup: dict[str, HybridStrategySpec] = {}

    for n2 in n2_values:
        for k_box in k_box_values:
            for family in families:
                if family == "majority":
                    for tie_bandwidth in _tie_bandwidth_values(n2):
                        spec = HybridStrategySpec(
                            family="majority",
                            n2=n2,
                            k_box=k_box,
                            tie_bandwidth=tie_bandwidth,
                        )
                        try:
                            validated = validate_spec(spec)
                            if required_boxes(validated) <= k_box:
                                dedup[candidate_name(validated)] = validated
                        except ValueError:
                            continue

                elif family == "pyramid":
                    spec = HybridStrategySpec(family="pyramid", n2=n2, k_box=k_box)
                    try:
                        validated = validate_spec(spec)
                        if required_boxes(validated) <= k_box:
                            dedup[candidate_name(validated)] = validated
                    except ValueError:
                        continue

                elif family == "horizontal":
                    for order in ("maj_pyr", "pyr_maj"):
                        for split in range(1, n2):
                            majority_len = split if order == "maj_pyr" else (n2 - split)
                            for tie_bandwidth in _tie_bandwidth_values(majority_len):
                                spec = HybridStrategySpec(
                                    family="horizontal",
                                    n2=n2,
                                    k_box=k_box,
                                    order=order,
                                    split=split,
                                    tie_bandwidth=tie_bandwidth,
                                )
                                try:
                                    validated = validate_spec(spec)
                                    if required_boxes(validated) <= k_box:
                                        dedup[candidate_name(validated)] = validated
                                except ValueError:
                                    continue

                elif family == "vertical":
                    if n2 <= 1:
                        continue
                    max_depth = n2.bit_length() - 1
                    for order in ("maj_pyr", "pyr_maj"):
                        for depth_s in range(1, max_depth):
                            if order == "maj_pyr":
                                majority_len = n2 // (2**depth_s)
                            else:
                                majority_len = n2 // (2**depth_s)
                            for tie_bandwidth in _tie_bandwidth_values(majority_len):
                                spec = HybridStrategySpec(
                                    family="vertical",
                                    n2=n2,
                                    k_box=k_box,
                                    order=order,
                                    depth_s=depth_s,
                                    tie_bandwidth=tie_bandwidth,
                                )
                                try:
                                    validated = validate_spec(spec)
                                    if required_boxes(validated) <= k_box:
                                        dedup[candidate_name(validated)] = validated
                                except ValueError:
                                    continue

                else:
                    continue

    return list(dedup.values())


def exhaustive_search(
    *,
    specs: list[HybridStrategySpec],
    p_rule: float,
    evaluation_mode: str = "analytical",
    top_k: int | None = None,
) -> list[SearchResult]:
    """Evaluate and rank candidate specs through canonical evaluator dispatch.

    Raises:
        ValueError: if ``top_k`` is negative when provided.
    """
    scored: list[SearchResult] = []

    for spec in specs:
        validated = validate_spec(spec)
        candidate = StrategyCandidate(spec=validated, source="strategy_search")
        request = EvaluationRequest(
            candidate=candidate,
            p_rule=float(p_rule),
            mode=evaluation_mode,
        )
        result = evaluate_candidate(request)
        scored.append(
            SearchResult(
                spec=validated,
                candidate_name=candidate_name(validated),
                success=float(result.success),
                p_rule=float(p_rule),
                evaluation_mode=evaluation_mode,
            )
        )

    scored.sort(key=lambda item: (-item.success, item.candidate_name))

    if top_k is not None:
        if top_k < 0:
            raise ValueError("top_k must be non-negative or None.")
        return scored[:top_k]

    return scored
