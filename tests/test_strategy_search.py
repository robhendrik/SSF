from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name, validate_spec
from ssf.strategy_evaluation import EvaluationResult
from ssf.strategy_search import enumerate_legal_specs, exhaustive_search


SCRIPT = PROJECT_ROOT / "scripts" / "optimize" / "run_analytical_strategy_scan.py"


def test_enumerate_legal_specs_returns_unique_valid_specs() -> None:
    specs = enumerate_legal_specs(
        n2_values=[4, 8],
        k_box_values=[1, 3, 7],
        families=["majority", "pyramid", "horizontal", "vertical"],
    )

    assert specs

    names: list[str] = []
    families_seen: set[str] = set()
    for spec in specs:
        validate_spec(spec)
        names.append(candidate_name(spec))
        families_seen.add(spec.family)

    assert len(names) == len(set(names))
    assert families_seen == {"majority", "pyramid", "horizontal", "vertical"}


def test_exhaustive_search_returns_sorted_results() -> None:
    specs = [
        HybridStrategySpec(family="majority", n2=4, k_box=3, tie_bandwidth=0),
        HybridStrategySpec(family="pyramid", n2=4, k_box=3),
    ]

    results = exhaustive_search(specs=specs, p_rule=0.8, evaluation_mode="analytical")

    assert len(results) == 2
    assert results[0].success >= results[1].success


def test_exhaustive_search_uses_evaluator_dispatcher(monkeypatch) -> None:
    from ssf import strategy_search

    calls: list[object] = []

    def fake_evaluate_candidate(request):
        calls.append(request)
        return EvaluationResult(
            success=float(request.candidate.spec.n2),
            mode=request.mode,
            exact=True,
            method="fake",
            notes="",
        )

    monkeypatch.setattr(strategy_search, "evaluate_candidate", fake_evaluate_candidate)

    specs = [
        HybridStrategySpec(family="majority", n2=4, k_box=3, tie_bandwidth=0),
        HybridStrategySpec(family="majority", n2=8, k_box=7, tie_bandwidth=0),
    ]
    results = strategy_search.exhaustive_search(
        specs=specs,
        p_rule=0.8,
        evaluation_mode="analytical",
    )

    assert len(calls) == len(specs)
    assert [item.success for item in results] == [8.0, 4.0]


def test_analytical_scan_script_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--p-rules",
            "0.8",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
            "--families",
            "majority",
            "pyramid",
            "--top-k",
            "3",
            "--quiet",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "traceback" not in result.stderr.lower()
