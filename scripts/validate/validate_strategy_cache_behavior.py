"""Strict integration validation for canonical evaluation cache behavior.

Architectural role:
- verifies strategy_cache policy, replacement, persistence, and invalidation semantics
- exercises evaluate_candidate cache integration in realistic dispatcher flows

Invariants:
- cache identity is request+policy-version sensitive
- trust hierarchy and lookup filters gate cache reuse deterministically

Failure behavior:
- assertion failures or malformed cache payloads produce non-zero process exit
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Callable

# Ensure local src package is importable when script is run directly from repo root.
SSF_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SSF_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name
from ssf.strategy_cache import (
    CachePolicy,
    EvaluationCache,
    record_from_result,
    result_from_record,
    should_replace_cached,
)
from ssf.strategy_evaluation import (
    EvaluationRequest,
    EvaluationResult,
    StrategyCandidate,
    evaluate_candidate,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for cache validation workspace and reporting."""
    parser = argparse.ArgumentParser(description="Strict validation for strategy evaluation cache behavior.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("work") / "fixture_bank" / "evaluations_validation_tmp",
    )
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def _assert_close(lhs: float, rhs: float, eps: float = 1e-12) -> None:
    assert abs(float(lhs) - float(rhs)) <= eps, f"Expected {lhs} ~= {rhs}"


def _jsonl_nonempty_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _safe_rmtree(path: Path, retries: int = 5, sleep_seconds: float = 0.25) -> None:
    if not path.exists():
        return

    # Some Windows/OneDrive/AV workflows can leave files read-only or briefly locked.
    def _onexc(func: Callable[..., object], target: str, exc: BaseException) -> None:
        if not isinstance(exc, PermissionError):
            raise exc
        os.chmod(target, stat.S_IWRITE)
        func(target)

    last_error: PermissionError | None = None
    for attempt in range(retries):
        try:
            shutil.rmtree(path, onexc=_onexc)
            return
        except PermissionError as err:
            last_error = err
            if attempt == retries - 1:
                break
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error


def _make_request(
    *,
    spec: HybridStrategySpec,
    mode: str,
    p_rule: float = 0.8,
    box_limit: int | None = None,
    subset_policy: str = "up_to",
    n_train_samples: int = 10_000,
    n_eval_samples: int = 20_000,
    confidence: float = 0.95,
    seed: int = 123,
) -> EvaluationRequest:
    """Build canonical evaluation request used by cache validation checks."""
    return EvaluationRequest(
        candidate=StrategyCandidate(spec=spec, source="validate_strategy_cache_behavior"),
        p_rule=float(p_rule),
        mode=mode,
        box_limit=box_limit,
        subset_policy=subset_policy,
        n_train_samples=n_train_samples,
        n_eval_samples=n_eval_samples,
        confidence=confidence,
        seed=seed,
    )


def _expect_value_error(fn: Callable[[], None], label: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError: {label}")


def _check_cross_process_persistence(cache_dir: Path, tiny_spec: HybridStrategySpec) -> None:
    cache = EvaluationCache(cache_dir)
    cache.load()
    policy = CachePolicy(enabled=True, read=True, write=True, cache_dir=cache_dir)

    request = _make_request(spec=tiny_spec, mode="dense_exhaustive", p_rule=0.8)
    result = evaluate_candidate(request, cache_policy=policy, cache=cache)

    assert cache.jsonl_path.exists(), "Expected cache JSONL file to exist."
    assert cache.latest_path.exists(), "Expected latest cache JSON file to exist."

    cache_reloaded = EvaluationCache(cache_dir)
    cache_reloaded.load()
    record = cache_reloaded.lookup(request, policy)
    assert record is not None, "Expected cross-process lookup to find persisted record."

    roundtrip = result_from_record(record)
    _assert_close(roundtrip.success, result.success)


def _check_analytical_persistence_policy(cache_dir: Path, majority_spec: HybridStrategySpec) -> None:
    cache = EvaluationCache(cache_dir)
    cache.load()

    request = _make_request(spec=majority_spec, mode="analytical", p_rule=0.8)

    policy_no_persist = CachePolicy(
        enabled=True,
        read=False,
        write=True,
        persist_analytical=False,
        cache_dir=cache_dir,
    )
    evaluate_candidate(request, cache_policy=policy_no_persist, cache=cache)
    cache.load()
    analytical_records = [r for r in cache.all_records() if r.mode == "analytical"]
    assert not analytical_records, "Analytical records must not persist by default."

    policy_persist = CachePolicy(
        enabled=True,
        read=False,
        write=True,
        persist_analytical=True,
        cache_dir=cache_dir,
    )
    evaluate_candidate(request, cache_policy=policy_persist, cache=cache)
    cache.load()
    analytical_records = [r for r in cache.all_records() if r.mode == "analytical"]
    assert analytical_records, "Expected analytical record when persist_analytical=True."


def _check_trust_hierarchy_replacement(cache_dir: Path, tiny_spec: HybridStrategySpec) -> None:
    policy_default = CachePolicy(enabled=True, cache_dir=cache_dir)
    policy_analytical_allowed = CachePolicy(enabled=True, persist_analytical=True, cache_dir=cache_dir)

    req_dense_mc = _make_request(spec=tiny_spec, mode="dense_mc", n_eval_samples=10_000)
    req_dense_ex = _make_request(spec=tiny_spec, mode="dense_exhaustive")
    req_proc_mc = _make_request(spec=tiny_spec, mode="procedural_mc", n_eval_samples=10_000)
    req_analytical = _make_request(spec=tiny_spec, mode="analytical")

    dense_mc = record_from_result(
        req_dense_mc,
        EvaluationResult(
            success=0.70,
            mode="dense_mc",
            exact=False,
            method="mc_dense",
            ci_low=0.68,
            ci_high=0.72,
            stderr=0.01,
        ),
        policy_default,
    )
    dense_ex = record_from_result(
        req_dense_ex,
        EvaluationResult(success=0.71, mode="dense_exhaustive", exact=True, method="exhaustive"),
        policy_default,
    )
    proc_mc = record_from_result(
        req_proc_mc,
        EvaluationResult(
            success=0.705,
            mode="procedural_mc",
            exact=False,
            method="mc_procedural",
            ci_low=0.69,
            ci_high=0.72,
            stderr=0.01,
        ),
        policy_default,
    )
    analytical = record_from_result(
        req_analytical,
        EvaluationResult(success=0.69, mode="analytical", exact=True, method="analytical"),
        policy_analytical_allowed,
    )

    assert should_replace_cached(dense_mc, dense_ex, policy_default)
    assert should_replace_cached(dense_mc, proc_mc, policy_default)
    assert not should_replace_cached(dense_ex, dense_mc, policy_default)
    assert not should_replace_cached(dense_mc, analytical, policy_default)


def _check_same_mode_mc_replacement_edge_cases(cache_dir: Path, majority_spec: HybridStrategySpec) -> None:
    policy_uncertain_true = CachePolicy(
        enabled=True,
        persist_analytical=True,
        recompute_if_uncertain=True,
        cache_dir=cache_dir,
    )
    policy_uncertain_false = CachePolicy(
        enabled=True,
        persist_analytical=True,
        recompute_if_uncertain=False,
        cache_dir=cache_dir,
    )

    req_10k = _make_request(spec=majority_spec, mode="dense_mc", n_eval_samples=10_000)
    req_20k = _make_request(spec=majority_spec, mode="dense_mc", n_eval_samples=20_000)

    existing_10k = record_from_result(
        req_10k,
        EvaluationResult(
            success=0.70,
            mode="dense_mc",
            exact=False,
            method="mc_dense",
            ci_low=0.68,
            ci_high=0.72,
            stderr=0.01,
        ),
        policy_uncertain_true,
    )
    newer_20k = record_from_result(
        req_20k,
        EvaluationResult(
            success=0.70,
            mode="dense_mc",
            exact=False,
            method="mc_dense",
            ci_low=0.68,
            ci_high=0.72,
            stderr=0.01,
        ),
        policy_uncertain_true,
    )
    assert should_replace_cached(existing_10k, newer_20k, policy_uncertain_true)

    existing_narrow = record_from_result(
        req_20k,
        EvaluationResult(
            success=0.70,
            mode="dense_mc",
            exact=False,
            method="mc_dense",
            ci_low=0.69,
            ci_high=0.71,
            stderr=0.01,
        ),
        policy_uncertain_true,
    )
    wider_ci = record_from_result(
        req_20k,
        EvaluationResult(
            success=0.70,
            mode="dense_mc",
            exact=False,
            method="mc_dense",
            ci_low=0.65,
            ci_high=0.75,
            stderr=0.03,
        ),
        policy_uncertain_true,
    )
    assert not should_replace_cached(existing_narrow, wider_ci, policy_uncertain_true)

    missing_ci = record_from_result(
        req_20k,
        EvaluationResult(
            success=0.70,
            mode="dense_mc",
            exact=False,
            method="mc_dense",
            ci_low=None,
            ci_high=None,
            stderr=None,
        ),
        policy_uncertain_true,
    )
    assert should_replace_cached(existing_narrow, missing_ci, policy_uncertain_true)
    assert not should_replace_cached(existing_narrow, missing_ci, policy_uncertain_false)


def _check_lookup_filters(cache_dir: Path, majority_spec: HybridStrategySpec) -> None:
    filter_dir = cache_dir / "lookup_filters"
    cache = EvaluationCache(filter_dir)
    cache.clear()
    cache.load()

    write_policy = CachePolicy(enabled=True, read=False, write=True, cache_dir=filter_dir)
    request = _make_request(spec=majority_spec, mode="dense_mc", n_eval_samples=5_000)
    result = EvaluationResult(
        success=0.70,
        mode="dense_mc",
        exact=False,
        method="mc_dense",
        ci_low=0.50,
        ci_high=0.90,
        stderr=0.05,
    )
    inserted = cache.upsert(request, result, write_policy)
    assert inserted is not None

    cache.load()
    policy_min_trust = CachePolicy(enabled=True, read=True, write=False, min_trust_level=3, cache_dir=filter_dir)
    assert cache.lookup(request, policy_min_trust) is None

    policy_min_samples = CachePolicy(
        enabled=True,
        read=True,
        write=False,
        min_n_eval_samples=10_000,
        cache_dir=filter_dir,
    )
    assert cache.lookup(request, policy_min_samples) is None

    policy_max_ci = CachePolicy(enabled=True, read=True, write=False, max_ci_width=0.20, cache_dir=filter_dir)
    assert cache.lookup(request, policy_max_ci) is None


def _check_force_recompute_and_rw_modes(cache_dir: Path, tiny_spec: HybridStrategySpec) -> None:
    request = _make_request(spec=tiny_spec, mode="dense_exhaustive", p_rule=0.8)

    disabled_dir = cache_dir / "rw_disabled"
    cache_disabled = EvaluationCache(disabled_dir)
    cache_disabled.clear()
    res_disabled = evaluate_candidate(
        request,
        cache_policy=CachePolicy(enabled=False, cache_dir=disabled_dir),
        cache=cache_disabled,
    )
    assert "cache_hit" not in res_disabled.metadata
    assert not cache_disabled.jsonl_path.exists()

    rw_dir = cache_dir / "rw_modes"
    cache_rw = EvaluationCache(rw_dir)
    cache_rw.clear()
    write_only = CachePolicy(enabled=True, read=False, write=True, cache_dir=rw_dir)
    first = evaluate_candidate(request, cache_policy=write_only, cache=cache_rw)
    second = evaluate_candidate(request, cache_policy=write_only, cache=cache_rw)
    assert first.metadata.get("cache_hit") is False
    assert second.metadata.get("cache_hit") is False
    assert cache_rw.jsonl_path.exists()

    before_lines = _jsonl_nonempty_line_count(cache_rw.jsonl_path)
    read_only = CachePolicy(enabled=True, read=True, write=False, cache_dir=rw_dir)
    read_hit = evaluate_candidate(request, cache_policy=read_only, cache=cache_rw)
    after_lines = _jsonl_nonempty_line_count(cache_rw.jsonl_path)
    assert read_hit.metadata.get("cache_hit") is True
    assert before_lines == after_lines

    rw_hit_dir = cache_dir / "rw_hit"
    cache_rw_hit = EvaluationCache(rw_hit_dir)
    cache_rw_hit.clear()
    read_write = CachePolicy(enabled=True, read=True, write=True, cache_dir=rw_hit_dir)
    first_rw = evaluate_candidate(request, cache_policy=read_write, cache=cache_rw_hit)
    second_rw = evaluate_candidate(request, cache_policy=read_write, cache=cache_rw_hit)
    assert first_rw.metadata.get("cache_hit") is False
    assert second_rw.metadata.get("cache_hit") is True

    force_recompute = CachePolicy(enabled=True, read=False, write=True, cache_dir=rw_hit_dir)
    forced = evaluate_candidate(request, cache_policy=force_recompute, cache=cache_rw_hit)
    assert forced.metadata.get("cache_hit") is False


def _check_invalidation(cache_dir: Path, majority_spec: HybridStrategySpec) -> None:
    inv_dir = cache_dir / "invalidate"
    cache = EvaluationCache(inv_dir)
    cache.clear()
    cache.load()

    req_mc = _make_request(spec=majority_spec, mode="dense_mc", n_eval_samples=10_000, seed=1)
    req_ex = _make_request(spec=majority_spec, mode="dense_exhaustive", seed=1)

    policy_v1 = CachePolicy(enabled=True, cache_dir=inv_dir, evaluator_version="v1")
    rec_mc = record_from_result(
        req_mc,
        EvaluationResult(
            success=0.70,
            mode="dense_mc",
            exact=False,
            method="mc_dense",
            ci_low=0.68,
            ci_high=0.72,
            stderr=0.01,
        ),
        policy_v1,
    )
    rec_ex = record_from_result(
        req_ex,
        EvaluationResult(success=0.71, mode="dense_exhaustive", exact=True, method="exhaustive"),
        policy_v1,
    )
    cache.append(rec_mc)
    cache.append(rec_ex)

    cache.invalidate_by_mode("dense_mc")
    modes = {record.mode for record in cache.all_records()}
    assert "dense_mc" not in modes
    assert "dense_exhaustive" in modes

    policy_obsolete = CachePolicy(enabled=True, cache_dir=inv_dir, evaluator_version="obsolete_version")
    rec_obsolete = record_from_result(
        req_ex,
        EvaluationResult(success=0.71, mode="dense_exhaustive", exact=True, method="exhaustive"),
        policy_obsolete,
    )
    cache.append(rec_obsolete)

    cache.invalidate_by_evaluator_version("obsolete_version")
    versions = {record.evaluator_version for record in cache.all_records()}
    assert "obsolete_version" not in versions
    assert "v1" in versions

    cache.clear()
    assert cache.all_records() == []
    assert not cache.jsonl_path.exists()
    assert not cache.latest_path.exists()


def _check_key_isolation_edge_cases(cache_dir: Path, majority_spec: HybridStrategySpec) -> None:
    policy = CachePolicy(enabled=True, persist_analytical=True, cache_dir=cache_dir)

    req_base = _make_request(spec=majority_spec, mode="dense_exhaustive", p_rule=0.8, box_limit=None)
    req_p = _make_request(spec=majority_spec, mode="dense_exhaustive", p_rule=0.5, box_limit=None)
    req_box = _make_request(spec=majority_spec, mode="dense_exhaustive", p_rule=0.8, box_limit=1)
    req_subset = _make_request(spec=majority_spec, mode="dense_exhaustive", p_rule=0.8, subset_policy="exactly")
    req_mode = _make_request(spec=majority_spec, mode="analytical", p_rule=0.8)

    key_base = record_from_result(
        req_base,
        EvaluationResult(success=0.71, mode="dense_exhaustive", exact=True, method="exhaustive"),
        policy,
    ).key
    key_p = record_from_result(
        req_p,
        EvaluationResult(success=0.71, mode="dense_exhaustive", exact=True, method="exhaustive"),
        policy,
    ).key
    key_box = record_from_result(
        req_box,
        EvaluationResult(success=0.71, mode="dense_exhaustive", exact=True, method="exhaustive"),
        policy,
    ).key
    key_subset = record_from_result(
        req_subset,
        EvaluationResult(success=0.71, mode="dense_exhaustive", exact=True, method="exhaustive"),
        policy,
    ).key
    key_mode = record_from_result(
        req_mode,
        EvaluationResult(success=0.71, mode="analytical", exact=True, method="analytical"),
        policy,
    ).key

    req_mc_10k_seed1 = _make_request(
        spec=majority_spec,
        mode="dense_mc",
        n_eval_samples=10_000,
        seed=1,
    )
    req_mc_20k_seed1 = _make_request(
        spec=majority_spec,
        mode="dense_mc",
        n_eval_samples=20_000,
        seed=1,
    )
    req_mc_10k_seed2 = _make_request(
        spec=majority_spec,
        mode="dense_mc",
        n_eval_samples=10_000,
        seed=2,
    )
    mc_result = EvaluationResult(
        success=0.70,
        mode="dense_mc",
        exact=False,
        method="mc_dense",
        ci_low=0.68,
        ci_high=0.72,
        stderr=0.01,
    )
    key_mc_10k_seed1 = record_from_result(req_mc_10k_seed1, mc_result, policy).key
    key_mc_20k_seed1 = record_from_result(req_mc_20k_seed1, mc_result, policy).key
    key_mc_10k_seed2 = record_from_result(req_mc_10k_seed2, mc_result, policy).key

    assert key_base != key_p
    assert key_base != key_box
    assert key_base != key_subset
    assert key_base != key_mode
    assert key_mc_10k_seed1 != key_mc_20k_seed1
    assert key_mc_10k_seed1 != key_mc_10k_seed2


def _check_corruption_invalid_input_behavior(cache_dir: Path) -> None:
    corruption_root = cache_dir / "corruption"
    corruption_root.mkdir(parents=True, exist_ok=True)

    malformed_jsonl_dir = corruption_root / "malformed_jsonl"
    malformed_jsonl_dir.mkdir(parents=True, exist_ok=True)
    (malformed_jsonl_dir / "b_performance.jsonl").write_text("not-json\n", encoding="utf-8")
    _expect_value_error(lambda: EvaluationCache(malformed_jsonl_dir).load(), "malformed JSONL line")

    malformed_latest_dir = corruption_root / "malformed_latest"
    malformed_latest_dir.mkdir(parents=True, exist_ok=True)
    (malformed_latest_dir / "b_performance_latest.json").write_text("{broken", encoding="utf-8")
    _expect_value_error(lambda: EvaluationCache(malformed_latest_dir).load(), "malformed latest JSON")

    missing_fields_dir = corruption_root / "missing_fields"
    missing_fields_dir.mkdir(parents=True, exist_ok=True)
    (missing_fields_dir / "b_performance.jsonl").write_text(json.dumps({"key": "k"}) + "\n", encoding="utf-8")
    _expect_value_error(lambda: EvaluationCache(missing_fields_dir).load(), "missing required fields")


def main(argv: list[str] | None = None) -> int:
    """Run cache-behavior validation suite and return 0 on full pass."""
    args = _parse_args(argv)
    cache_dir = Path(args.cache_dir)

    if not args.keep_cache and cache_dir.exists():
        _safe_rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Fail early on existing corruption when --keep-cache is used.
    EvaluationCache(cache_dir).load()

    tiny_spec = HybridStrategySpec(
        "horizontal",
        n2=4,
        k_box=3,
        split=2,
        order="maj_pyr",
        tie_bandwidth=0,
    )
    majority_spec = HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0)

    tiny_name = candidate_name(tiny_spec)
    majority_name = candidate_name(majority_spec)
    assert tiny_name.startswith("hybrid_horiz_")
    assert majority_name.startswith("majority_")

    checks: list[str] = []

    _check_cross_process_persistence(cache_dir, tiny_spec)
    checks.append("cross_process_persistence")
    if not args.quiet:
        print("PASS cross_process_persistence")

    _check_analytical_persistence_policy(cache_dir, majority_spec)
    checks.append("analytical_persistence_policy")
    if not args.quiet:
        print("PASS analytical_persistence_policy")

    _check_trust_hierarchy_replacement(cache_dir, tiny_spec)
    checks.append("trust_hierarchy_replacement")
    if not args.quiet:
        print("PASS trust_hierarchy_replacement")

    _check_same_mode_mc_replacement_edge_cases(cache_dir, majority_spec)
    checks.append("same_mode_mc_replacement_edge_cases")
    if not args.quiet:
        print("PASS same_mode_mc_replacement_edge_cases")

    _check_lookup_filters(cache_dir, majority_spec)
    checks.append("lookup_filters")
    if not args.quiet:
        print("PASS lookup_filters")

    _check_force_recompute_and_rw_modes(cache_dir, tiny_spec)
    checks.append("force_recompute_and_rw_modes")
    if not args.quiet:
        print("PASS force_recompute_and_rw_modes")

    _check_invalidation(cache_dir, majority_spec)
    checks.append("invalidation")
    if not args.quiet:
        print("PASS invalidation")

    _check_key_isolation_edge_cases(cache_dir, majority_spec)
    checks.append("key_isolation_edge_cases")
    if not args.quiet:
        print("PASS key_isolation_edge_cases")

    _check_corruption_invalid_input_behavior(cache_dir)
    checks.append("corruption_invalid_input_behavior")
    if not args.quiet:
        print("PASS corruption_invalid_input_behavior")

    final_cache = EvaluationCache(cache_dir)
    final_cache.load()

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checks_passed": checks,
            "cache_dir": str(cache_dir),
            "number_of_records": len(final_cache.all_records()),
            "specs": {
                "tiny": asdict(tiny_spec),
                "majority": asdict(majority_spec),
            },
        }
        args.json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
