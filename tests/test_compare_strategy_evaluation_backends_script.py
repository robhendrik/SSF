from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

from scripts.validate.compare_strategy_evaluation_backends import _compare_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "validate" / "compare_strategy_evaluation_backends.py"


def test_compare_strategy_evaluation_backends_script_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--p-rules",
            "0.8",
            "--spec-set",
            "tiny",
            "--n-train-samples",
            "1000",
            "--n-eval-samples",
            "2000",
            "--mc-tolerance",
            "0.08",
            "--quiet",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "traceback" not in result.stderr.lower()


def test_exact_comparison_required_only_when_both_backends_exact() -> None:
    def build_mode_results(analytical_exact: bool, dense_exhaustive_exact: bool) -> dict[str, object]:
        return {
            "analytical": SimpleNamespace(success=0.5, exact=analytical_exact),
            "dense_exhaustive": SimpleNamespace(success=0.5, exact=dense_exhaustive_exact),
            "dense_mc": SimpleNamespace(success=0.5),
            "procedural_mc": SimpleNamespace(success=0.5),
        }

    outcomes_both, _ = _compare_row(
        mode_results=build_mode_results(analytical_exact=True, dense_exhaustive_exact=True),
        mc_tolerance=0.08,
        exact_tolerance=1e-12,
    )
    exact_outcome_both = next(o for o in outcomes_both if o.name == "analytical_vs_dense_exhaustive")
    assert exact_outcome_both.required is True
    assert exact_outcome_both.warning_only is False

    outcomes_one, _ = _compare_row(
        mode_results=build_mode_results(analytical_exact=True, dense_exhaustive_exact=False),
        mc_tolerance=0.08,
        exact_tolerance=1e-12,
    )
    exact_outcome_one = next(o for o in outcomes_one if o.name == "analytical_vs_dense_exhaustive")
    assert exact_outcome_one.required is False
    assert exact_outcome_one.warning_only is True
