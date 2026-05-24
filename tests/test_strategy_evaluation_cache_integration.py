from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_spec import HybridStrategySpec
from ssf.strategy_cache import CachePolicy, EvaluationCache, EvaluationCacheRecord, record_from_result
from ssf.strategy_evaluation import EvaluationRequest, StrategyCandidate, evaluate_candidate


def _request(mode: str) -> EvaluationRequest:
    spec = HybridStrategySpec(family="majority", n2=2, k_box=1, tie_bandwidth=0)
    return EvaluationRequest(
        candidate=StrategyCandidate(spec=spec, source="integration_test"),
        p_rule=0.8,
        mode=mode,
        box_limit=None,
        subset_policy="up_to",
        n_train_samples=200,
        n_eval_samples=400,
        confidence=0.95,
        seed=1,
    )


def test_cache_disabled_preserves_old_behavior(tmp_path):
    req = _request("analytical")
    baseline = evaluate_candidate(req)
    policy = CachePolicy(enabled=False, cache_dir=tmp_path)
    result = evaluate_candidate(req, cache_policy=policy)

    assert result.success == baseline.success
    assert result.mode == baseline.mode
    assert result.exact == baseline.exact
    assert result.method == baseline.method
    assert "cache_hit" not in result.metadata


def test_cache_enabled_write_creates_files_in_tmp_path(tmp_path):
    req = _request("dense_exhaustive")
    policy = CachePolicy(enabled=True, cache_dir=tmp_path)
    cache = EvaluationCache(tmp_path)

    result = evaluate_candidate(req, cache_policy=policy, cache=cache)

    assert result.metadata.get("cache_hit") is False
    assert (tmp_path / "b_performance.jsonl").exists()
    assert (tmp_path / "b_performance_latest.json").exists()


def test_second_call_same_request_returns_cache_hit_true(tmp_path):
    req = _request("dense_exhaustive")
    policy = CachePolicy(enabled=True, cache_dir=tmp_path)
    cache = EvaluationCache(tmp_path)

    first = evaluate_candidate(req, cache_policy=policy, cache=cache)
    second = evaluate_candidate(req, cache_policy=policy, cache=cache)

    assert first.metadata.get("cache_hit") is False
    assert second.metadata.get("cache_hit") is True
    assert second.success == first.success


def test_read_false_forces_recomputation_and_cache_hit_false(tmp_path):
    req = _request("dense_exhaustive")
    priming_policy = CachePolicy(enabled=True, read=True, write=True, cache_dir=tmp_path)
    cache = EvaluationCache(tmp_path)
    evaluate_candidate(req, cache_policy=priming_policy, cache=cache)

    read_disabled_policy = CachePolicy(enabled=True, read=False, write=False, cache_dir=tmp_path)
    result = evaluate_candidate(req, cache_policy=read_disabled_policy, cache=cache)

    assert result.metadata.get("cache_hit") is False


def test_write_false_does_not_create_cache_files(tmp_path):
    req = _request("dense_exhaustive")
    policy = CachePolicy(enabled=True, read=True, write=False, cache_dir=tmp_path)
    cache = EvaluationCache(tmp_path)

    result = evaluate_candidate(req, cache_policy=policy, cache=cache)

    assert result.metadata.get("cache_hit") is False
    assert not (tmp_path / "b_performance.jsonl").exists()


def test_analytical_not_written_by_default(tmp_path):
    req = _request("analytical")
    policy = CachePolicy(enabled=True, cache_dir=tmp_path)
    cache = EvaluationCache(tmp_path)

    result = evaluate_candidate(req, cache_policy=policy, cache=cache)

    assert result.metadata.get("cache_hit") is False
    assert not (tmp_path / "b_performance.jsonl").exists()


def test_analytical_written_when_persist_enabled(tmp_path):
    req = _request("analytical")
    policy = CachePolicy(enabled=True, persist_analytical=True, cache_dir=tmp_path)
    cache = EvaluationCache(tmp_path)

    first = evaluate_candidate(req, cache_policy=policy, cache=cache)
    second = evaluate_candidate(req, cache_policy=policy, cache=cache)

    assert first.metadata.get("cache_hit") is False
    assert second.metadata.get("cache_hit") is True
    assert (tmp_path / "b_performance.jsonl").exists()


def test_dense_exhaustive_cached_result_is_reused(tmp_path):
    req = _request("dense_exhaustive")
    policy = CachePolicy(enabled=True, cache_dir=tmp_path)
    cache = EvaluationCache(tmp_path)

    evaluate_candidate(req, cache_policy=policy, cache=cache)
    cached = evaluate_candidate(req, cache_policy=policy, cache=cache)

    assert cached.metadata.get("cache_hit") is True


def test_dense_mc_lower_quality_cached_record_replaced_by_dense_exhaustive(tmp_path):
    req_dense_ex = _request("dense_exhaustive")
    policy = CachePolicy(enabled=True, cache_dir=tmp_path)
    cache = EvaluationCache(tmp_path)
    cache.load()

    # Inject a lower-trust record under the dense_exhaustive key to verify replacement.
    dense_key = record_from_result(req_dense_ex, evaluate_candidate(req_dense_ex), policy).key
    low_record = EvaluationCacheRecord(
        key=dense_key,
        candidate_name="majority_n2_2_k_1_b_0",
        spec={"family": "majority", "n2": 2, "k_box": 1, "tie_bandwidth": 0, "tie_value": 0},
        success=0.6,
        mode="dense_mc",
        p_rule=0.8,
        box_limit=None,
        subset_policy="up_to",
        exact=False,
        analytical=False,
        trust_level=2,
        n_train_samples=100,
        n_eval_samples=200,
        confidence=0.95,
        seed=1,
        ci_low=0.50,
        ci_high=0.90,
        stderr=0.10,
        evaluator_version="v1",
        created_at_utc="2026-05-22T00:00:00+00:00",
        metadata={"method": "mc_dense", "notes": "injected low quality"},
    )
    cache.append(low_record)

    recompute_policy = CachePolicy(enabled=True, read=False, write=True, cache_dir=tmp_path)
    fresh = evaluate_candidate(req_dense_ex, cache_policy=recompute_policy, cache=cache)
    assert fresh.metadata.get("cache_hit") is False

    cache.load()
    records = cache.all_records()
    target = next(record for record in records if record.key == dense_key)
    assert target.mode == "dense_exhaustive"
    assert target.trust_level == 4


def test_cache_key_distinguishes_evaluation_modes(tmp_path):
    policy = CachePolicy(enabled=True, persist_analytical=True, cache_dir=tmp_path)
    cache = EvaluationCache(tmp_path)

    req_analytical = _request("analytical")
    req_dense_ex = _request("dense_exhaustive")

    evaluate_candidate(req_analytical, cache_policy=policy, cache=cache)
    evaluate_candidate(req_dense_ex, cache_policy=policy, cache=cache)

    keys = {record.key for record in cache.all_records()}
    assert len(keys) == 2
