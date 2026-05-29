from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.bob_exhaustive import evaluate_strategy_exhaustive
from ssf.bob_mc import evaluate_strategy_mc
from ssf.bob_mc_procedural import evaluate_strategy_mc_procedural
from ssf.hybrid_strategy_spec import HybridStrategySpec
from ssf.strategy_evaluation import EvaluationRequest, StrategyCandidate, evaluation_cache_key
from ssf.strategy_search import enumerate_legal_specs


SCRIPT = PROJECT_ROOT / "scripts" / "optimize" / "run_exhaustive_strategy_optimizer.py"


def _majority_comm_bit(field: int, n2: int) -> int:
    bits = [(field >> i) & 1 for i in range(n2)]
    return 1 if sum(bits) > (n2 / 2) else 0


def _zero_box_strategy_dict(n2: int = 4) -> dict:
    n_fields = 1 << n2
    comm = np.zeros((n_fields, 1), dtype=np.uint8)
    for field in range(n_fields):
        comm[field, 0] = _majority_comm_bit(field, n2)
    return {
        "n2": n2,
        "k_box": 0,
        "f_tables": [],
        "comm": comm,
    }


class _ZeroBoxProceduralMajority:
    n2 = 4
    k_box = 0

    def measure_a(self, box_idx: int, field: int, out_a_prefix: int) -> int:
        raise AssertionError("measure_a must not be called when k_box=0")

    def comm(self, field: int, out_a: int) -> int:
        assert out_a == 0
        return _majority_comm_bit(field, self.n2)


def _run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )


def test_dense_exhaustive_k_box_zero_accepts_empty_tables_and_comm_shape() -> None:
    strategy = _zero_box_strategy_dict(n2=4)

    s_none = evaluate_strategy_exhaustive(strategy, p_rule=0.73, box_limit=None, subset_policy="up_to")
    s_up_to_zero = evaluate_strategy_exhaustive(strategy, p_rule=0.73, box_limit=0, subset_policy="up_to")
    s_exact_zero = evaluate_strategy_exhaustive(strategy, p_rule=0.73, box_limit=0, subset_policy="exact")

    assert 0.0 <= s_none <= 1.0
    assert s_none == pytest.approx(s_up_to_zero)
    assert s_none == pytest.approx(s_exact_zero)

    with pytest.raises(ValueError, match="box_limit"):
        evaluate_strategy_exhaustive(strategy, p_rule=0.73, box_limit=1, subset_policy="up_to")

    bad_comm = dict(strategy)
    bad_comm["comm"] = np.zeros((1 << 4, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="Invalid comm shape"):
        evaluate_strategy_exhaustive(bad_comm, p_rule=0.73)


def test_dense_mc_k_box_zero_matches_exhaustive_within_tolerance() -> None:
    strategy = _zero_box_strategy_dict(n2=4)

    exact = evaluate_strategy_exhaustive(strategy, p_rule=0.77, box_limit=0, subset_policy="up_to")
    mc = evaluate_strategy_mc(
        strategy,
        p_rule=0.77,
        box_limit=0,
        subset_policy="up_to",
        n_train_samples=8_000,
        n_eval_samples=12_000,
        seed=123,
    )

    assert mc.ci_low <= mc.success <= mc.ci_high
    assert abs(mc.success - exact) < 0.03


def test_procedural_mc_k_box_zero_matches_exhaustive_within_tolerance() -> None:
    strategy = _zero_box_strategy_dict(n2=4)
    exact = evaluate_strategy_exhaustive(strategy, p_rule=0.77, box_limit=0, subset_policy="up_to")

    proc = evaluate_strategy_mc_procedural(
        _ZeroBoxProceduralMajority(),
        p_rule=0.77,
        box_limit=0,
        subset_policy="exact",
        n_train_samples=8_000,
        n_eval_samples=12_000,
        seed=123,
    )

    assert proc.ci_low <= proc.success <= proc.ci_high
    assert abs(proc.success - exact) < 0.03


def test_candidate_generation_k_box_zero_emits_only_compatible_specs() -> None:
    specs = enumerate_legal_specs(
        n2_values=[4],
        k_box_values=[0],
        families=["majority", "pyramid", "horizontal", "vertical"],
    )

    assert specs
    assert all(spec.k_box == 0 for spec in specs)
    assert {spec.family for spec in specs} == {"majority"}


def test_cli_dense_exhaustive_accepts_k_box_zero_for_majority(tmp_path: Path) -> None:
    output_json = tmp_path / "zero_box.json"
    result = _run_script(
        [
            "--mode",
            "dense_exhaustive",
            "--p-rules",
            "0.8",
            "--n2-values",
            "4",
            "--k-box-values",
            "0",
            "--families",
            "majority",
            "--top-k",
            "1",
            "--json-output",
            str(output_json),
            "--quiet",
        ]
    )

    assert result.returncode == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["parameters"]["k_box_values"] == [0]
    assert payload["results"]
    assert payload["results"][0]["ranked_results"]
    assert payload["results"][0]["ranked_results"][0]["spec"]["k_box"] == 0


def test_cache_key_distinguishes_k_box_zero_from_positive() -> None:
    spec_zero = HybridStrategySpec(family="majority", n2=4, k_box=0, tie_bandwidth=0)
    spec_one = HybridStrategySpec(family="majority", n2=4, k_box=1, tie_bandwidth=0)

    req_zero = EvaluationRequest(
        candidate=StrategyCandidate(spec=spec_zero, source="k0_test"),
        p_rule=0.8,
        mode="dense_exhaustive",
        box_limit=0,
        subset_policy="up_to",
    )
    req_one = EvaluationRequest(
        candidate=StrategyCandidate(spec=spec_one, source="k1_test"),
        p_rule=0.8,
        mode="dense_exhaustive",
        box_limit=0,
        subset_policy="up_to",
    )

    assert evaluation_cache_key(req_zero) != evaluation_cache_key(req_one)
