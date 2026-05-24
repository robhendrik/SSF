from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_spec import candidate_name, validate_spec
from ssf.strategy_cache import CachePolicy, EvaluationCache
from ssf.strategy_evaluation import EvaluationResult
from ssf.strategy_optimizer import (
    BeamSearchConfig,
    beam_search_optimizer,
    mutate_spec,
)
from ssf.strategy_search import enumerate_legal_specs


SCRIPT = PROJECT_ROOT / "scripts" / "optimize" / "run_beam_strategy_optimizer.py"


def _run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )


def _small_legal_specs() -> list:
    return enumerate_legal_specs(
        n2_values=[4],
        k_box_values=[1, 3],
        families=["majority", "pyramid", "horizontal", "vertical"],
    )


def test_mutate_spec_returns_valid_unique_specs() -> None:
    specs = _small_legal_specs()
    assert specs

    base = specs[0]
    mutated = mutate_spec(
        base,
        n2_values=[4, 8],
        k_box_values=[1, 3, 7],
        families=["majority", "pyramid", "horizontal", "vertical"],
        rng=random.Random(123),
    )

    names = [candidate_name(spec) for spec in mutated]
    assert len(names) == len(set(names))
    for spec in mutated:
        validate_spec(spec)


def test_beam_search_uses_dispatcher_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import ssf.strategy_optimizer as optimizer_module

    calls: list[tuple[float, str]] = []

    def _fake_eval(request, cache_policy=None, cache=None):
        calls.append((float(request.p_rule), str(request.mode)))
        return EvaluationResult(success=0.5, mode=request.mode, exact=False, method="fake")

    monkeypatch.setattr(optimizer_module, "evaluate_candidate", _fake_eval)

    initial = _small_legal_specs()[:3]
    config = BeamSearchConfig(
        p_rule=0.8,
        evaluation_mode="analytical",
        beam_width=2,
        generations=1,
        mutations_per_candidate=2,
        top_k=2,
        seed=1,
    )

    results = beam_search_optimizer(
        initial_specs=initial,
        n2_values=[4],
        k_box_values=[1, 3],
        families=["majority", "pyramid", "horizontal", "vertical"],
        config=config,
    )

    assert results
    assert calls
    assert all(item[1] == "analytical" for item in calls)


def test_beam_search_deterministic_with_seed() -> None:
    initial = _small_legal_specs()
    config = BeamSearchConfig(
        p_rule=0.8,
        evaluation_mode="analytical",
        beam_width=3,
        generations=2,
        mutations_per_candidate=2,
        top_k=3,
        seed=17,
    )

    first = beam_search_optimizer(
        initial_specs=initial,
        n2_values=[4],
        k_box_values=[1, 3],
        families=["majority", "pyramid", "horizontal", "vertical"],
        config=config,
    )
    second = beam_search_optimizer(
        initial_specs=initial,
        n2_values=[4],
        k_box_values=[1, 3],
        families=["majority", "pyramid", "horizontal", "vertical"],
        config=config,
    )

    assert [row.candidate_name for row in first] == [row.candidate_name for row in second]


def test_beam_search_returns_sorted_results() -> None:
    initial = _small_legal_specs()
    config = BeamSearchConfig(
        p_rule=0.8,
        evaluation_mode="analytical",
        beam_width=4,
        generations=2,
        mutations_per_candidate=2,
        top_k=4,
        seed=4,
    )

    results = beam_search_optimizer(
        initial_specs=initial,
        n2_values=[4],
        k_box_values=[1, 3],
        families=["majority", "pyramid", "horizontal", "vertical"],
        config=config,
    )

    assert results
    pairs = [(row.success, row.candidate_name) for row in results]
    assert pairs == sorted(pairs, key=lambda pair: (-pair[0], pair[1]))


def test_beam_search_raises_on_empty_initial_specs() -> None:
    config = BeamSearchConfig(
        p_rule=0.8,
        evaluation_mode="analytical",
        beam_width=3,
        generations=1,
        mutations_per_candidate=1,
        top_k=3,
        seed=0,
    )

    with pytest.raises(ValueError, match="No valid initial specs"):
        beam_search_optimizer(
            initial_specs=[],
            n2_values=[4],
            k_box_values=[1, 3],
            families=["majority", "pyramid", "horizontal", "vertical"],
            config=config,
        )


def test_beam_search_cache_integration(tmp_path: Path) -> None:
    initial = _small_legal_specs()[:3]
    cache = EvaluationCache(tmp_path)
    policy = CachePolicy(
        enabled=True,
        read=True,
        write=True,
        persist_analytical=True,
        cache_dir=tmp_path,
    )
    config = BeamSearchConfig(
        p_rule=0.8,
        evaluation_mode="analytical",
        beam_width=2,
        generations=1,
        mutations_per_candidate=2,
        top_k=2,
        seed=0,
        cache_policy=policy,
    )

    first = beam_search_optimizer(
        initial_specs=initial,
        n2_values=[4],
        k_box_values=[1, 3],
        families=["majority", "pyramid", "horizontal", "vertical"],
        config=config,
        cache=cache,
    )
    second = beam_search_optimizer(
        initial_specs=initial,
        n2_values=[4],
        k_box_values=[1, 3],
        families=["majority", "pyramid", "horizontal", "vertical"],
        config=config,
        cache=cache,
    )

    assert first
    assert second
    assert any(row.metadata.get("cache_hit") is True for row in second)


def test_beam_search_script_smoke_analytical() -> None:
    result = _run_script(
        [
            "--mode",
            "analytical",
            "--p-rule",
            "0.8",
            "--n2-values",
            "4",
            "--k-box-values",
            "1",
            "3",
            "--beam-width",
            "3",
            "--generations",
            "2",
            "--mutations-per-candidate",
            "2",
            "--top-k",
            "3",
            "--quiet",
        ]
    )

    assert result.returncode == 0


def test_beam_search_no_shortcuts_source() -> None:
    source_path = PROJECT_ROOT / "src" / "ssf" / "strategy_optimizer.py"
    source = source_path.read_text(encoding="utf-8")

    forbidden_tokens = [
        "expected_scores",
        "pyramid_n2_8",
        "return 0.75",
        "analytical formulas",
        "from .bob_exhaustive",
        "from .bob_mc",
        "from .bob_mc_procedural",
    ]

    for token in forbidden_tokens:
        assert token not in source
