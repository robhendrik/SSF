from __future__ import annotations

from pathlib import Path
import sys

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
