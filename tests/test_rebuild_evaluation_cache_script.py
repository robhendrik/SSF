from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "validate" / "rebuild_evaluation_cache.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )


def test_rebuild_script_smoke_dense_exhaustive(tmp_path: Path) -> None:
    result = _run(
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
            "--max-strategies",
            "1",
            "--cache-dir",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert result.returncode == 0
    assert (tmp_path / "b_performance.jsonl").exists()
    assert (tmp_path / "b_performance_latest.json").exists()


def test_analytical_default_does_not_persist(tmp_path: Path) -> None:
    result = _run(
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
            "--max-strategies",
            "1",
            "--cache-dir",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert result.returncode == 0
    assert "Analytical results are not persisted unless --persist-analytical is set." in result.stdout

    latest_path = tmp_path / "b_performance_latest.json"
    if not latest_path.exists():
        return

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert all(record.get("mode") != "analytical" for record in payload.values())


def test_analytical_with_persist_writes_record(tmp_path: Path) -> None:
    result = _run(
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
            "--max-strategies",
            "1",
            "--cache-dir",
            str(tmp_path),
            "--persist-analytical",
            "--quiet",
        ]
    )

    assert result.returncode == 0
    latest_path = tmp_path / "b_performance_latest.json"
    assert latest_path.exists()

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert any(record.get("mode") == "analytical" for record in payload.values())


def test_force_recompute_twice_appends_or_replaces(tmp_path: Path) -> None:
    args = [
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
        "--max-strategies",
        "1",
        "--cache-dir",
        str(tmp_path),
        "--force-recompute",
        "--quiet",
    ]

    first = _run(args)
    assert first.returncode == 0

    jsonl_path = tmp_path / "b_performance.jsonl"
    assert jsonl_path.exists()
    lines_after_first = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    second = _run(args)
    assert second.returncode == 0

    lines_after_second = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines_after_second) >= len(lines_after_first)
