from __future__ import annotations

from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_spec import HybridStrategySpec
from ssf.strategy_evaluation import EvaluationRequest, StrategyCandidate, evaluate_candidate


def _request(mode: str) -> EvaluationRequest:
    return EvaluationRequest(
        candidate=StrategyCandidate(
            spec=HybridStrategySpec(family="majority", n2=2, k_box=1, tie_bandwidth=0),
            source="test",
        ),
        p_rule=0.8,
        mode=mode,
    )


def test_evaluate_candidate_analytical_smoke():
    result = evaluate_candidate(_request("analytical"))
    assert 0.0 <= result.success <= 1.0
    assert result.mode == "analytical"
    assert "cache_hit" not in result.metadata


def test_evaluate_candidate_dense_exhaustive_smoke():
    result = evaluate_candidate(_request("dense_exhaustive"))
    assert 0.0 <= result.success <= 1.0
    assert result.mode == "dense_exhaustive"
    assert result.exact is True
    assert "cache_hit" not in result.metadata


def test_dense_exhaustive_none_equals_full_k_box_up_to():
    req_none = EvaluationRequest(
        candidate=StrategyCandidate(
            spec=HybridStrategySpec(family="majority", n2=2, k_box=1, tie_bandwidth=0),
            source="test",
        ),
        p_rule=0.8,
        mode="dense_exhaustive",
        box_limit=None,
        subset_policy="up_to",
    )
    req_full = EvaluationRequest(
        candidate=req_none.candidate,
        p_rule=req_none.p_rule,
        mode=req_none.mode,
        box_limit=1,
        subset_policy="up_to",
    )

    s_none = evaluate_candidate(req_none).success
    s_full = evaluate_candidate(req_full).success
    assert s_none == pytest.approx(s_full)


def test_dense_exhaustive_monotonicity_for_up_to_limits():
    candidate = StrategyCandidate(
        spec=HybridStrategySpec(family="majority", n2=2, k_box=1, tie_bandwidth=0),
        source="test",
    )
    req0 = EvaluationRequest(candidate=candidate, p_rule=0.8, mode="dense_exhaustive", box_limit=0)
    req1 = EvaluationRequest(candidate=candidate, p_rule=0.8, mode="dense_exhaustive", box_limit=1)

    s0 = evaluate_candidate(req0).success
    s1 = evaluate_candidate(req1).success
    assert s0 <= s1 + 1e-12


def test_dense_exhaustive_exact_policy_out_of_range_raises():
    req_bad = EvaluationRequest(
        candidate=StrategyCandidate(
            spec=HybridStrategySpec(family="majority", n2=2, k_box=1, tie_bandwidth=0),
            source="test",
        ),
        p_rule=0.8,
        mode="dense_exhaustive",
        box_limit=2,
        subset_policy="exact",
    )
    with pytest.raises(ValueError):
        evaluate_candidate(req_bad)


def test_non_dense_mode_rejects_box_limit_and_subset_policy_restrictions():
    req = EvaluationRequest(
        candidate=StrategyCandidate(
            spec=HybridStrategySpec(family="majority", n2=2, k_box=1, tie_bandwidth=0),
            source="test",
        ),
        p_rule=0.8,
        mode="analytical",
        box_limit=0,
        subset_policy="up_to",
    )
    with pytest.raises(ValueError):
        evaluate_candidate(req)
