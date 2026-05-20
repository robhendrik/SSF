import sys
import subprocess
import json
import csv
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_GENERATOR = PROJECT_ROOT / "scripts" / "generate" / "generate_a_strategy_fixtures.py"
EXHAUSTIVE = PROJECT_ROOT / "scripts" / "evaluate" / "evaluate_b_side_exhaustive.py"
MONTE_CARLO = PROJECT_ROOT / "scripts" / "evaluate" / "evaluate_b_side_monte_carlo.py"
OPTIMIZER = PROJECT_ROOT / "scripts" / "optimize" / "optimize_hybrid_family_parameters.py"
OPTIMIZER_LEGACY_GENERATOR_DEP = (
    PROJECT_ROOT / "scripts" / "generate" / "generate_hybrid_family_reference_a_strategies.py"
)

def run_script(script: Path, args, cwd: Path, env=None, check=True):
    cmd = [sys.executable, str(script)] + list(args)
    result = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Script {script} failed: {result.stderr}\nSTDOUT:\n{result.stdout}")
    return result

def run_py_module(module: str, args, cwd: Path, env=None, check=True):
    cmd = [sys.executable, "-m", module] + list(args)
    result = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Module {module} failed: {result.stderr}\nSTDOUT:\n{result.stdout}")
    return result


def latest_run_dir(base_dir: Path) -> Path:
    run_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    assert run_dirs, f"No run directory found under {base_dir}"
    return run_dirs[-1]


def test_canonical_generator_majority_pyramid_to_exhaustive_evaluator(tmp_path):
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    # Generate canonical majority/pyramid fixtures for a small representative setting.
    run_script(
        CANONICAL_GENERATOR,
        [
            "--output-dir",
            str(ref_dir),
            "--overwrite",
            "--families",
            "majority",
            "pyramid",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
        ],
        cwd=PROJECT_ROOT,
    )
    # Evaluate with exhaustive
    results_dir = tmp_path / "exhaustive_results"
    results_dir.mkdir()
    run_script(
        EXHAUSTIVE,
        [
            "--fixture-dir", str(ref_dir),
            "--output-dir", str(results_dir),
            "--p-rules", "1.0",
            "--subset-policy", "up_to",
            "--overwrite",
        ],
        cwd=PROJECT_ROOT,
    )
    run_dir = latest_run_dir(results_dir)
    # Check outputs
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "results.md").exists()
    # Check at least majority and pyramid rows appear
    with (run_dir / "results.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [row["strategy_name"] for row in rows if "strategy_name" in row]
    assert any("majority" in n for n in names)
    assert any("pyramid" in n for n in names)


def test_canonical_generator_hybrid_to_exhaustive_evaluator(tmp_path):
    hybrid_dir = tmp_path / "hybrid"
    hybrid_dir.mkdir()
    # Generate canonical hybrid fixtures via legacy-default representatives.
    run_script(
        CANONICAL_GENERATOR,
        [
            "--output-dir",
            str(hybrid_dir),
            "--overwrite",
            "--legacy-defaults",
            "--families",
            "horizontal",
            "vertical",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
        ],
        cwd=PROJECT_ROOT,
    )
    # Evaluate with exhaustive
    results_dir = tmp_path / "hybrid_exhaustive_results"
    results_dir.mkdir()
    run_script(
        EXHAUSTIVE,
        [
            "--fixture-dir", str(hybrid_dir),
            "--output-dir", str(results_dir),
            "--p-rules", "0.8,0.9",
            "--subset-policy", "up_to",
            "--overwrite",
        ],
        cwd=PROJECT_ROOT,
    )
    run_dir = latest_run_dir(results_dir)
    # Check outputs
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "results.md").exists()
    # At least one hybrid candidate evaluated
    with (run_dir / "results.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert any("hybrid" in row["strategy_name"] for row in rows if "strategy_name" in row)


def test_monte_carlo_quick_agrees_roughly_with_exhaustive(tmp_path):
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    # Generate canonical majority fixture for n2=4, k=3.
    run_script(
        CANONICAL_GENERATOR,
        [
            "--output-dir",
            str(ref_dir),
            "--overwrite",
            "--families",
            "majority",
            "--n2-values",
            "4",
            "--k-box-values",
            "3",
        ],
        cwd=PROJECT_ROOT,
    )
    # Find a majority fixture
    majority_fixture = None
    for f in ref_dir.iterdir():
        if f.name.startswith("majority_n2_4_k_3") and f.suffix == ".npz":
            majority_fixture = f
            break
    assert majority_fixture is not None
    # Exhaustive evaluation
    ex_results_dir = tmp_path / "exhaustive"
    ex_results_dir.mkdir()
    run_script(
        EXHAUSTIVE,
        [
            "--fixture-dir", str(ref_dir),
            "--output-dir", str(ex_results_dir),
            "--p-rules", "0.9",
            "--subset-policy", "up_to",
            "--overwrite",
        ],
        cwd=PROJECT_ROOT,
    )
    # MC evaluation (quick, small samples)
    mc_results_dir = tmp_path / "mc"
    mc_results_dir.mkdir()
    run_script(
        MONTE_CARLO,
        [
            "--fixture-dir", str(ref_dir),
            "--output-dir", str(mc_results_dir),
            "--p-rules", "0.9",
            "--subset-policy", "up_to",
            "--samples", "2000",
            "--seeds", "0",
            "--quick",
            "--overwrite",
        ],
        cwd=PROJECT_ROOT,
    )
    # Compare results
    ex_run_dir = latest_run_dir(ex_results_dir)
    with (ex_run_dir / "results.csv").open(encoding="utf-8") as f:
        ex_rows = [row for row in csv.DictReader(f) if row["strategy_name"] == majority_fixture.stem]
    with (mc_results_dir / "results.csv").open(encoding="utf-8") as f:
        mc_rows = [row for row in csv.DictReader(f) if row["strategy_name"] == majority_fixture.stem]
    assert ex_rows and mc_rows
    ex_success = float(ex_rows[0]["success"])
    mc_success = float(mc_rows[0]["success"])
    # Loose tolerance
    assert abs(ex_success - mc_success) <= 0.10


def test_hybrid_optimizer_quick_pipeline(tmp_path):
    if not OPTIMIZER_LEGACY_GENERATOR_DEP.exists():
        pytest.skip(
            "Optimizer quick pipeline still depends on legacy hybrid-family generator module, "
            "which is not present in this migration step."
        )

    out_dir = tmp_path / "optimizer"
    out_dir.mkdir()
    run_py_module(
        "scripts.optimize.optimize_hybrid_family_parameters",
        ["--quick", "--output-dir", str(out_dir), "--overwrite"],
        cwd=PROJECT_ROOT,
    )
    # Check outputs
    csv_path = out_dir / "optimization_results.csv"
    json_path = out_dir / "optimization_results.json"
    md_path = out_dir / "optimization_summary.md"
    assert csv_path.exists()
    assert json_path.exists()
    assert md_path.exists()
    # At least one candidate evaluated and one best row
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert any(row.get("best_for_condition", "") in ("True", True, 1, "1") for row in rows)