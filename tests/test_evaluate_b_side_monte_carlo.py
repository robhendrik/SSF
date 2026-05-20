from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import evaluate_b_side_monte_carlo as eval_script
from scripts.generate.generate_a_strategy_fixtures import generate_a_strategy_fixtures


def _generate_fixtures(fixture_dir: Path) -> None:
    generate_a_strategy_fixtures(
        output_dir=fixture_dir,
        overwrite=True,
        legacy_defaults=True,
    )


def test_script_imports_and_help(capsys):
    assert callable(eval_script.main)

    with pytest.raises(SystemExit) as excinfo:
        eval_script.main(["--help"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--fixture-dir" in captured.out
    assert "--procedural" in captured.out


def test_fixture_loading_from_dir_and_manifest(tmp_path: Path):
    fixture_dir = tmp_path / "fixtures"
    _generate_fixtures(fixture_dir)

    from_dir = eval_script._load_from_fixture_dir(fixture_dir)
    from_manifest = eval_script._load_from_manifest(fixture_dir / "manifest.json")

    assert len(from_dir) > 0
    assert len(from_manifest) == len(from_dir)


def test_quick_run_writes_outputs_and_calls_evaluator(tmp_path: Path, monkeypatch):
    fixture_dir = tmp_path / "fixtures"
    _generate_fixtures(fixture_dir)

    calls: list[dict] = []

    def fake_mc(strategy, **kwargs):
        calls.append({"strategy": strategy, "kwargs": kwargs})
        return SimpleNamespace(
            success=0.625,
            ci_low=0.60,
            ci_high=0.65,
            stderr=0.01,
        )

    monkeypatch.setattr(eval_script, "evaluate_strategy_mc", fake_mc)

    out_dir = tmp_path / "results"
    exit_code = eval_script.main(
        [
            "--quick",
            "--fixture-dir",
            str(fixture_dir),
            "--output-dir",
            str(out_dir),
            "--p-rules",
            "0.77",
            "1.0",
            "--box-limits",
            "1",
            "2",
            "--seeds",
            "0",
            "1",
            "--samples",
            "5000",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["kwargs"]["n_train_samples"] == 1000
    assert calls[0]["kwargs"]["n_eval_samples"] == 1000
    assert calls[0]["kwargs"]["seed"] == 0

    assert (out_dir / "results.csv").exists()
    assert (out_dir / "results.json").exists()
    assert (out_dir / "results.md").exists()
    assert not (out_dir / "summary_by_seed.csv").exists()

    data = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert data["row_count"] == 1
    assert data["rows"][0]["success"] == pytest.approx(0.625)


def test_quick_run_with_procedural_uses_procedural_evaluator(tmp_path: Path, monkeypatch):
    fixture_dir = tmp_path / "fixtures"
    _generate_fixtures(fixture_dir)

    procedural_calls: list[dict] = []

    def fake_procedural(strategy, **kwargs):
        procedural_calls.append({"strategy": strategy, "kwargs": kwargs})
        return SimpleNamespace(
            success=0.7,
            ci_low=0.68,
            ci_high=0.72,
            stderr=0.01,
            elapsed_seconds=0.123,
        )

    monkeypatch.setattr(eval_script, "evaluate_strategy_mc_procedural", fake_procedural)

    out_dir = tmp_path / "procedural_results"
    exit_code = eval_script.main(
        [
            "--quick",
            "--procedural",
            "--fixture-dir",
            str(fixture_dir),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert len(procedural_calls) == 1
    assert procedural_calls[0]["kwargs"]["record_time"] is True
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "results.json").exists()
    assert (out_dir / "results.md").exists()
