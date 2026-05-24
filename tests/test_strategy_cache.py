from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_spec import HybridStrategySpec
from ssf.strategy_cache import (
    CachePolicy,
    EvaluationCache,
    EvaluationCacheRecord,
    TRUST_LEVELS,
    should_replace_cached,
)
from ssf.strategy_evaluation import EvaluationRequest, EvaluationResult, StrategyCandidate


def _request(mode: str = "dense_mc", n_eval_samples: int = 50_000) -> EvaluationRequest:
    spec = HybridStrategySpec(family="majority", n2=4, k_box=1, tie_bandwidth=0)
    return EvaluationRequest(
        candidate=StrategyCandidate(spec=spec, source="test"),
        p_rule=0.8,
        mode=mode,
        box_limit=1,
        subset_policy="up_to",
        n_train_samples=20_000,
        n_eval_samples=n_eval_samples,
        confidence=0.95,
        seed=7,
    )


def _result(
    mode: str = "dense_mc",
    *,
    success: float = 0.73,
    exact: bool = False,
    ci_low: float | None = 0.70,
    ci_high: float | None = 0.76,
    stderr: float | None = 0.01,
) -> EvaluationResult:
    return EvaluationResult(
        success=success,
        mode=mode,
        exact=exact,
        method="test_method",
        notes="test_notes",
        ci_low=ci_low,
        ci_high=ci_high,
        stderr=stderr,
    )


def _record(
    *,
    key: str,
    mode: str,
    n_eval_samples: int | None,
    ci_low: float | None,
    ci_high: float | None,
    evaluator_version: str = "v1",
) -> EvaluationCacheRecord:
    return EvaluationCacheRecord(
        key=key,
        candidate_name="majority_n2_4_k_1_b_0",
        spec={"family": "majority", "n2": 4, "k_box": 1, "tie_bandwidth": 0, "tie_value": 0},
        success=0.7,
        mode=mode,
        p_rule=0.8,
        box_limit=1,
        subset_policy="up_to",
        exact=(mode == "dense_exhaustive"),
        analytical=(mode == "analytical"),
        trust_level=TRUST_LEVELS[mode],
        n_train_samples=20_000 if "mc" in mode else None,
        n_eval_samples=n_eval_samples,
        confidence=0.95 if "mc" in mode else None,
        seed=7 if "mc" in mode else None,
        ci_low=ci_low,
        ci_high=ci_high,
        stderr=0.01 if "mc" in mode else None,
        evaluator_version=evaluator_version,
        created_at_utc="2026-05-22T00:00:00+00:00",
        metadata={"method": "test"},
    )


def test_analytical_not_persisted_by_default(tmp_path):
    cache = EvaluationCache(tmp_path)
    cache.load()
    policy = CachePolicy(enabled=True)
    out = cache.upsert(_request(mode="analytical"), _result(mode="analytical", exact=True), policy)
    assert out is None
    assert cache.all_records() == []
    assert not (tmp_path / "b_performance.jsonl").exists()


def test_analytical_persisted_only_when_enabled(tmp_path):
    cache = EvaluationCache(tmp_path)
    cache.load()
    policy = CachePolicy(enabled=True, persist_analytical=True)
    out = cache.upsert(_request(mode="analytical"), _result(mode="analytical", exact=True), policy)
    assert out is not None
    assert out.analytical is True
    assert len(cache.all_records()) == 1
    assert (tmp_path / "b_performance.jsonl").exists()
    assert (tmp_path / "b_performance_latest.json").exists()


def test_dense_exhaustive_replaces_dense_mc():
    policy = CachePolicy(enabled=True)
    existing = _record(key="k", mode="dense_mc", n_eval_samples=50_000, ci_low=0.69, ci_high=0.75)
    new = _record(key="k", mode="dense_exhaustive", n_eval_samples=None, ci_low=None, ci_high=None)
    assert should_replace_cached(existing, new, policy) is True


def test_procedural_mc_replaces_dense_mc():
    policy = CachePolicy(enabled=True)
    existing = _record(key="k", mode="dense_mc", n_eval_samples=50_000, ci_low=0.69, ci_high=0.75)
    new = _record(key="k", mode="procedural_mc", n_eval_samples=50_000, ci_low=0.70, ci_high=0.74)
    assert should_replace_cached(existing, new, policy) is True


def test_dense_mc_does_not_replace_dense_exhaustive():
    policy = CachePolicy(enabled=True)
    existing = _record(key="k", mode="dense_exhaustive", n_eval_samples=None, ci_low=None, ci_high=None)
    new = _record(key="k", mode="dense_mc", n_eval_samples=200_000, ci_low=0.70, ci_high=0.74)
    assert should_replace_cached(existing, new, policy) is False


def test_same_mode_mc_more_samples_replaces_lower_samples():
    policy = CachePolicy(enabled=True, recompute_if_uncertain=False)
    existing = _record(key="k", mode="dense_mc", n_eval_samples=50_000, ci_low=0.70, ci_high=0.74)
    new = _record(key="k", mode="dense_mc", n_eval_samples=60_000, ci_low=0.70, ci_high=0.74)
    assert should_replace_cached(existing, new, policy) is True


def test_same_mode_mc_wider_ci_does_not_replace_if_samples_equal():
    policy = CachePolicy(enabled=True, recompute_if_uncertain=False)
    existing = _record(key="k", mode="dense_mc", n_eval_samples=50_000, ci_low=0.70, ci_high=0.74)
    new = _record(key="k", mode="dense_mc", n_eval_samples=50_000, ci_low=0.69, ci_high=0.75)
    assert should_replace_cached(existing, new, policy) is False


def test_lookup_respects_min_trust_level(tmp_path):
    cache = EvaluationCache(tmp_path)
    cache.load()
    policy_write = CachePolicy(enabled=True)
    req = _request(mode="dense_mc", n_eval_samples=40_000)
    cache.upsert(req, _result(mode="dense_mc"), policy_write)

    strict = CachePolicy(enabled=True, min_trust_level=TRUST_LEVELS["procedural_mc"])
    assert cache.lookup(req, strict) is None


def test_lookup_respects_min_n_eval_samples(tmp_path):
    cache = EvaluationCache(tmp_path)
    cache.load()
    policy_write = CachePolicy(enabled=True)
    req = _request(mode="dense_mc", n_eval_samples=10_000)
    cache.upsert(req, _result(mode="dense_mc"), policy_write)

    strict = CachePolicy(enabled=True, min_n_eval_samples=20_000)
    assert cache.lookup(req, strict) is None


def test_lookup_respects_max_ci_width(tmp_path):
    cache = EvaluationCache(tmp_path)
    cache.load()
    policy_write = CachePolicy(enabled=True)
    req = _request(mode="dense_mc", n_eval_samples=10_000)
    cache.upsert(
        req,
        _result(mode="dense_mc", ci_low=0.60, ci_high=0.80, stderr=0.02),
        policy_write,
    )

    strict = CachePolicy(enabled=True, max_ci_width=0.05)
    assert cache.lookup(req, strict) is None


def test_clear_removes_cache_files_or_empties_cache(tmp_path):
    cache = EvaluationCache(tmp_path)
    cache.load()
    policy = CachePolicy(enabled=True)
    cache.upsert(_request(mode="dense_mc"), _result(mode="dense_mc"), policy)
    assert len(cache.all_records()) == 1

    cache.clear()

    assert cache.all_records() == []
    jsonl = tmp_path / "b_performance.jsonl"
    latest = tmp_path / "b_performance_latest.json"
    assert (not jsonl.exists()) or (jsonl.exists() and jsonl.read_text(encoding="utf-8").strip() == "")
    assert (not latest.exists()) or (latest.exists() and latest.read_text(encoding="utf-8").strip() in ("", "{}"))


def test_invalidate_by_mode_removes_matching_latest_records(tmp_path):
    cache = EvaluationCache(tmp_path)
    cache.load()
    cache.append(_record(key="k1", mode="dense_mc", n_eval_samples=10_000, ci_low=0.69, ci_high=0.75))
    cache.append(_record(key="k2", mode="dense_exhaustive", n_eval_samples=None, ci_low=None, ci_high=None))

    cache.invalidate_by_mode("dense_mc")

    modes = {record.mode for record in cache.all_records()}
    assert "dense_mc" not in modes
    assert "dense_exhaustive" in modes


def test_cache_writes_jsonl_and_latest_json_in_tmp_path(tmp_path):
    cache = EvaluationCache(tmp_path)
    cache.load()
    policy = CachePolicy(enabled=True)
    cache.upsert(_request(mode="dense_mc"), _result(mode="dense_mc"), policy)

    assert (tmp_path / "b_performance.jsonl").exists()
    assert (tmp_path / "b_performance_latest.json").exists()
