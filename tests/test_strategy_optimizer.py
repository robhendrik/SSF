from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_spec import HybridStrategySpec
from ssf.strategy_cache import CachePolicy, EvaluationCache
from ssf.strategy_evaluation import EvaluationResult
from ssf.strategy_optimizer import exhaustive_strategy_optimizer, optimize_over_parameter_grid


SCRIPT = PROJECT_ROOT / "scripts" / "optimize" / "run_exhaustive_strategy_optimizer.py"


def _run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )


def test_exhaustive_optimizer_returns_sorted_results() -> None:
    specs = [
        HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0),
        HybridStrategySpec("pyramid", n2=4, k_box=3),
        HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0),
    ]

    results = exhaustive_strategy_optimizer(
        specs=specs,
        p_rule=0.8,
        evaluation_mode="analytical",
        top_k=10,
    )

    assert len(results) > 0

    pairs = [(item.success, item.candidate_name) for item in results]
    expected = sorted(pairs, key=lambda pair: (-pair[0], pair[1]))
    assert pairs == expected

    for idx, item in enumerate(results, start=1):
        assert item.rank == idx


def test_optimizer_uses_dispatcher_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import ssf.strategy_optimizer as optimizer_module

    calls: list[tuple[float, str]] = []

    def _fake_eval(request, cache_policy=None, cache=None):
        calls.append((float(request.p_rule), str(request.mode)))
        return EvaluationResult(success=0.5, mode=request.mode, exact=False, method="fake")

    monkeypatch.setattr(optimizer_module, "evaluate_candidate", _fake_eval)

    results = optimizer_module.exhaustive_strategy_optimizer(
        specs=[HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0)],
        p_rule=0.8,
        evaluation_mode="analytical",
        top_k=10,
    )

    assert len(results) == 1
    assert len(calls) == 1
    assert calls[0] == (0.8, "analytical")


def test_optimizer_top_k() -> None:
    specs = [
        HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0),
        HybridStrategySpec("pyramid", n2=4, k_box=3),
        HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0),
    ]

    results = exhaustive_strategy_optimizer(
        specs=specs,
        p_rule=0.8,
        evaluation_mode="analytical",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_optimizer_parameter_grid() -> None:
    result = optimize_over_parameter_grid(
        p_rules=[0.7, 0.8],
        n2_values=[4],
        k_box_values=[3],
        families=["majority", "pyramid"],
        evaluation_mode="analytical",
        top_k=3,
    )

    assert set(result.keys()) == {0.7, 0.8}
    assert all(len(rows) <= 3 for rows in result.values())


def test_optimizer_cache_integration(tmp_path: Path) -> None:
    specs = [HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0)]
    cache = EvaluationCache(tmp_path)
    policy = CachePolicy(
        enabled=True,
        read=True,
        write=True,
        persist_analytical=True,
        cache_dir=tmp_path,
    )

    first = exhaustive_strategy_optimizer(
        specs=specs,
        p_rule=0.8,
        evaluation_mode="analytical",
        top_k=10,
        cache_policy=policy,
        cache=cache,
    )
    second = exhaustive_strategy_optimizer(
        specs=specs,
        p_rule=0.8,
        evaluation_mode="analytical",
        top_k=10,
        cache_policy=policy,
        cache=cache,
    )

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].metadata.get("cache_hit") is False
    assert second[0].metadata.get("cache_hit") is True


def test_optimizer_force_recompute(tmp_path: Path) -> None:
    specs = [HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0)]
    cache = EvaluationCache(tmp_path)

    warm_policy = CachePolicy(
        enabled=True,
        read=True,
        write=True,
        persist_analytical=True,
        cache_dir=tmp_path,
    )
    exhaustive_strategy_optimizer(
        specs=specs,
        p_rule=0.8,
        evaluation_mode="analytical",
        top_k=10,
        cache_policy=warm_policy,
        cache=cache,
    )

    force_policy = CachePolicy(
        enabled=True,
        read=False,
        write=True,
        persist_analytical=True,
        cache_dir=tmp_path,
    )
    forced = exhaustive_strategy_optimizer(
        specs=specs,
        p_rule=0.8,
        evaluation_mode="analytical",
        top_k=10,
        cache_policy=force_policy,
        cache=cache,
    )

    assert len(forced) == 1
    assert forced[0].metadata.get("cache_hit") is False


def test_optimizer_smoke_dense_exhaustive(tmp_path: Path) -> None:
    result = _run_script(
        [
            "--mode",
            "dense_exhaustive",
            "--p-rules",
            "0.8",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
            "--families",
            "majority",
            "--top-k",
            "1",
            "--cache-enabled",
            "--cache-dir",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert result.returncode == 0


def test_optimizer_smoke_analytical(tmp_path: Path) -> None:
    result = _run_script(
        [
            "--mode",
            "analytical",
            "--p-rules",
            "0.8",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
            "--families",
            "majority",
            "--top-k",
            "1",
            "--cache-enabled",
            "--cache-dir",
            str(tmp_path),
            "--persist-analytical",
            "--quiet",
        ]
    )

    assert result.returncode == 0


def test_strategy_optimizer_no_shortcuts_source_text() -> None:
    source_path = PROJECT_ROOT / "src" / "ssf" / "strategy_optimizer.py"
    source = source_path.read_text(encoding="utf-8")

    forbidden_tokens = [
        "expected_scores",
        "hardcoded winners",
        "pyramid_n2_8",
        "return 0.75",
        "manual evaluator formulas",
    ]

    for token in forbidden_tokens:
        assert token not in source
