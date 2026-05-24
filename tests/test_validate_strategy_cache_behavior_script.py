from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "validate" / "validate_strategy_cache_behavior.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )


def test_validate_strategy_cache_behavior_script_smoke(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    result = _run(["--cache-dir", str(cache_dir), "--quiet"])

    assert result.returncode == 0
    assert "traceback" not in result.stderr.lower()
    assert (cache_dir / "b_performance.jsonl").exists()
    assert (cache_dir / "b_performance_latest.json").exists()


def test_validate_strategy_cache_behavior_script_fails_on_corruption(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "b_performance.jsonl").write_text("this is not json\n", encoding="utf-8")

    result = _run(["--cache-dir", str(cache_dir), "--keep-cache", "--quiet"])

    assert result.returncode != 0


def test_validate_strategy_cache_behavior_script_json_output(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    report_path = tmp_path / "report.json"

    result = _run(["--cache-dir", str(cache_dir), "--quiet", "--json-output", str(report_path)])

    assert result.returncode == 0
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    checks = payload.get("checks_passed")
    assert isinstance(checks, list)
    assert len(checks) > 0
