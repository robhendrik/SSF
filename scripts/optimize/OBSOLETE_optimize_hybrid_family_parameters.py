from __future__ import annotations

import argparse
import csv
import json
import time
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SSF_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SSF_DIR / "src"
if str(SSF_DIR) not in sys.path:
    sys.path.insert(0, str(SSF_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssf.bob_exhaustive import evaluate_strategy_exhaustive
from ssf.bob_mc import evaluate_strategy_mc
from ssf.hybrid_strategy_fixtures import build_strategy_fixture
from ssf.hybrid_strategy_spec import HybridStrategySpec, candidate_name, required_boxes


DEFAULT_OUTPUT_DIR = SSF_DIR / "results" / "hybrid_family_optimizer"
DEFAULT_P_RULES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_BOX_LIMITS = [0, 1, 2, 3]



# Extended CandidateSpec for both horizontal and vertical
@dataclass(frozen=True)
class CandidateSpec:
    candidate_name: str
    family: str
    n2: int
    k_box: int
    # Horizontal only
    split: Optional[str] = None
    split_left: Optional[int] = None
    split_right: Optional[int] = None
    majority_side: Optional[str] = None
    pyramid_side: Optional[str] = None
    tie_bandwidth: Optional[int] = None
    # Vertical only
    order: Optional[str] = None  # 'pyr_maj' or 'maj_pyr'
    depth_s: Optional[int] = None
    majority_len: Optional[int] = None
    boxes_used: Optional[int] = None


@dataclass(frozen=True)
class CandidateBuildResult:
    spec: CandidateSpec
    fixture_file: str
    strategy: Dict[str, Any]


def _is_power_of_two(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def _required_pyramid_boxes(side_len: int) -> int:
    return side_len - 1


def _default_splits(n2: int) -> List[int]:
    return list(range(1, n2))


def _legacy_candidate_to_hybrid_spec(spec: CandidateSpec) -> HybridStrategySpec:
    if spec.family == "hybrid_horizontal":
        if spec.split_left is None:
            raise ValueError("Horizontal candidate missing split_left.")
        if spec.majority_side == "left" and spec.pyramid_side == "right":
            order = "maj_pyr"
        elif spec.majority_side == "right" and spec.pyramid_side == "left":
            order = "pyr_maj"
        else:
            raise ValueError("Horizontal candidate has inconsistent majority/pyramid sides.")

        return HybridStrategySpec(
            family="horizontal",
            n2=spec.n2,
            k_box=spec.k_box,
            split=int(spec.split_left),
            order=order,
            tie_bandwidth=int(spec.tie_bandwidth) if spec.tie_bandwidth is not None else None,
            tie_value=0,
        )

    if spec.family == "hybrid_vertical_pyr_maj":
        return HybridStrategySpec(
            family="vertical",
            n2=spec.n2,
            k_box=spec.k_box,
            depth_s=int(spec.depth_s) if spec.depth_s is not None else None,
            order="pyr_maj",
            tie_bandwidth=int(spec.tie_bandwidth) if spec.tie_bandwidth is not None else None,
            tie_value=0,
        )

    if spec.family == "hybrid_vertical_maj_pyr":
        return HybridStrategySpec(
            family="vertical",
            n2=spec.n2,
            k_box=spec.k_box,
            depth_s=int(spec.depth_s) if spec.depth_s is not None else None,
            order="maj_pyr",
            tie_bandwidth=int(spec.tie_bandwidth) if spec.tie_bandwidth is not None else None,
            tie_value=0,
        )

    raise ValueError(f"Unknown candidate family: {spec.family}")


def enumerate_horizontal_candidate_specs(
    n2: int,
    k_box: int,
    tie_bandwidth_values: Iterable[int],
    split_values: Iterable[int],
) -> Tuple[List[CandidateSpec], List[Dict[str, Any]]]:
    generated: List[CandidateSpec] = []
    skipped: List[Dict[str, Any]] = []

    for left_size in split_values:
        right_size = n2 - left_size
        if left_size <= 0 or right_size <= 0:
            skipped.append(
                {
                    "split_left": left_size,
                    "split_right": right_size,
                    "reason": "split must create non-empty left and right sides",
                }
            )
            continue

        for majority_side in ("left", "right"):
            pyramid_side = "right" if majority_side == "left" else "left"
            majority_len = left_size if majority_side == "left" else right_size
            pyramid_len = right_size if majority_side == "left" else left_size

            if majority_len % 2 != 0:
                skipped.append(
                    {
                        "split_left": left_size,
                        "split_right": right_size,
                        "majority_side": majority_side,
                        "pyramid_side": pyramid_side,
                        "reason": "majority side length must be even",
                    }
                )
                continue

            if not _is_power_of_two(pyramid_len):
                skipped.append(
                    {
                        "split_left": left_size,
                        "split_right": right_size,
                        "majority_side": majority_side,
                        "pyramid_side": pyramid_side,
                        "reason": "pyramid side length must be power of two",
                    }
                )
                continue

            if pyramid_len not in (2, 4):
                skipped.append(
                    {
                        "split_left": left_size,
                        "split_right": right_size,
                        "majority_side": majority_side,
                        "pyramid_side": pyramid_side,
                        "reason": "phase-1 helper supports pyramid side length 2 or 4 only",
                    }
                )
                continue

            req_boxes = _required_pyramid_boxes(pyramid_len)
            if k_box < req_boxes:
                skipped.append(
                    {
                        "split_left": left_size,
                        "split_right": right_size,
                        "majority_side": majority_side,
                        "pyramid_side": pyramid_side,
                        "reason": f"requires at least {req_boxes} boxes but k_box={k_box}",
                    }
                )
                continue

            for tie_bandwidth in tie_bandwidth_values:
                legacy = CandidateSpec(
                    candidate_name="",
                    family="hybrid_horizontal",
                    n2=n2,
                    k_box=k_box,
                    split=f"{left_size}:{right_size}",
                    split_left=left_size,
                    split_right=right_size,
                    majority_side=majority_side,
                    pyramid_side=pyramid_side,
                    tie_bandwidth=int(tie_bandwidth),
                    boxes_used=req_boxes,
                )
                try:
                    canonical_spec = _legacy_candidate_to_hybrid_spec(legacy)
                    generated.append(
                        CandidateSpec(
                            **{
                                **asdict(legacy),
                                "candidate_name": candidate_name(canonical_spec),
                                "boxes_used": required_boxes(canonical_spec),
                            }
                        )
                    )
                except ValueError as exc:
                    skipped.append(
                        {
                            "split_left": left_size,
                            "split_right": right_size,
                            "majority_side": majority_side,
                            "pyramid_side": pyramid_side,
                            "tie_bandwidth": int(tie_bandwidth),
                            "reason": str(exc),
                        }
                    )

    return generated, skipped


def _save_npz_strategy(
    fixture_dir: Path,
    name: str,
    n2: int,
    k_box: int,
    f_tables: List[np.ndarray],
    comm_table: np.ndarray,
) -> str:
    fixture_file = f"{name}.npz"
    payload: Dict[str, np.ndarray] = {
        "n2": np.array(n2, dtype=np.int64),
        "k_box": np.array(k_box, dtype=np.int64),
        "comm_table": comm_table.astype(np.uint8, copy=False),
    }
    for d, table in enumerate(f_tables):
        payload[f"f_table_{d}"] = table.astype(np.uint8, copy=False)
    np.savez(fixture_dir / fixture_file, **payload)
    return fixture_file



def build_candidate_fixture(spec: CandidateSpec, fixture_dir: Path) -> CandidateBuildResult:
    canonical_spec = _legacy_candidate_to_hybrid_spec(spec)
    fixture = build_strategy_fixture(canonical_spec)

    fixture_file = _save_npz_strategy(
        fixture_dir,
        spec.candidate_name,
        spec.n2,
        spec.k_box,
        fixture.f_tables,
        fixture.comm_table,
    )

    strategy: Dict[str, Any] = fixture.as_strategy_dict()
    return CandidateBuildResult(spec=spec, fixture_file=fixture_file, strategy=strategy)


# --- Deterministic vertical builders (local, not in generator) ---
def build_vertical_pyr_maj_candidate(n2: int, k_box: int, s: int, b: int, c_tie: int = 0) -> Dict[str, Any]:
    # Section 5.4: s pyramid layers, then majority over apex
    rows = 2 ** n2
    f_tables = [np.zeros((rows, 2 ** d), dtype=np.uint8) for d in range(k_box)]
    # Build s pyramid layers
    block_size = 2
    block_count = n2 // (2 ** s)
    # For each field, compute pyramid layers
    # For s=1, first layer: pairs (x0^x1), (x2^x3), ...
    # For s=2, first: (x0^x1), (x2^x3), (x4^x5), (x6^x7), then combine pairs, etc.
    # For each field, build pyramid layers
    for field in range(rows):
        bits = [(field >> i) & 1 for i in range(n2)]
        layer = bits[:]
        valid = True
        for d in range(s):
            if len(layer) % 2 != 0 or len(layer) == 0:
                valid = False
                break
            next_layer = []
            for i in range(0, len(layer), 2):
                v = layer[i] ^ layer[i + 1]
                if d < k_box and (i // 2) < f_tables[d].shape[1]:
                    f_tables[d][field, i // 2] = v
                next_layer.append(v)
            layer = next_layer
        if not valid or len(layer) == 0:
            continue
        # After s layers, layer = apex sequence (len = n2 // 2**s)
        ones = sum(layer)
        zeros = len(layer) - ones
        if abs(ones - zeros) <= b:
            comm_val = c_tie
        else:
            comm_val = 1 if ones > zeros else 0
        # comm_table is (rows, 2**k_box), comm is same for all out_a
        # We'll fill comm_table after loop
    comm_table = np.zeros((rows, 2 ** k_box), dtype=np.uint8)
    for field in range(rows):
        bits = [(field >> i) & 1 for i in range(n2)]
        layer = bits[:]
        for d in range(s):
            next_layer = []
            for i in range(0, len(layer), 2):
                v = layer[i] ^ layer[i + 1]
                next_layer.append(v)
            layer = next_layer
        ones = sum(layer)
        zeros = len(layer) - ones
        if abs(ones - zeros) <= b:
            comm_val = c_tie
        else:
            comm_val = 1 if ones > zeros else 0
        comm_table[field, :] = comm_val
    return {"f_tables": f_tables, "comm_table": comm_table}


def build_vertical_maj_pyr_candidate(n2: int, k_box: int, s: int, b: int, c_tie: int = 0) -> Dict[str, Any]:
    # Section 5.5: block majority, then pyramid
    rows = 2 ** n2
    f_tables = [np.zeros((rows, 2 ** d), dtype=np.uint8) for d in range(k_box)]
    block_count = 2 ** s
    block_size = n2 // block_count
    # For each field, compute block majorities
    for field in range(rows):
        bits = [(field >> i) & 1 for i in range(n2)]
        y = []
        valid = True
        for block in range(block_count):
            block_bits = bits[block * block_size : (block + 1) * block_size]
            if len(block_bits) != block_size:
                valid = False
                break
            ones = sum(block_bits)
            zeros = block_size - ones
            if abs(ones - zeros) <= b:
                v = c_tie
            else:
                v = 1 if ones > zeros else 0
            y.append(v)
        if not valid or len(y) == 0:
            continue
        # Now apply pyramid over y using s layers
        layer = y[:]
        for d in range(s):
            if len(layer) % 2 != 0 or len(layer) == 0:
                valid = False
                break
            next_layer = []
            for i in range(0, len(layer), 2):
                v = layer[i] ^ layer[i + 1]
                if d < k_box and (i // 2) < f_tables[d].shape[1]:
                    f_tables[d][field, i // 2] = v
                next_layer.append(v)
            layer = next_layer
        if not valid or len(layer) == 0:
            continue
        # comm_table: apex of pyramid
        comm_val = layer[0]
        # comm_table is (rows, 2**k_box), comm is same for all out_a
    comm_table = np.zeros((rows, 2 ** k_box), dtype=np.uint8)
    for field in range(rows):
        bits = [(field >> i) & 1 for i in range(n2)]
        y = []
        for block in range(block_count):
            block_bits = bits[block * block_size : (block + 1) * block_size]
            ones = sum(block_bits)
            zeros = block_size - ones
            if abs(ones - zeros) <= b:
                v = c_tie
            else:
                v = 1 if ones > zeros else 0
            y.append(v)
        layer = y[:]
        for d in range(s):
            next_layer = []
            for i in range(0, len(layer), 2):
                v = layer[i] ^ layer[i + 1]
                next_layer.append(v)
            layer = next_layer
        comm_val = layer[0]
        comm_table[field, :] = comm_val
    return {"f_tables": f_tables, "comm_table": comm_table}


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace outputs.")
    path.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for file_name in (
            "candidates_manifest.json",
            "optimization_results.csv",
            "optimization_results.json",
            "optimization_summary.md",
        ):
            file_path = path / file_name
            if file_path.exists():
                file_path.unlink()
        fixtures_dir = path / "fixtures"
        best_dir = fixtures_dir / "best"
        if fixtures_dir.exists():
            shutil.rmtree(fixtures_dir)
        best_dir.mkdir(parents=True, exist_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize deterministic hybrid-family parameters over explicit sweeps.")
    parser.add_argument("--family", choices=["horizontal", "vertical", "all"], default="horizontal")
    parser.add_argument("--n2", type=int, default=8)
    parser.add_argument("--k-box", type=int, default=1)
    parser.add_argument("--p-rules", type=float, nargs="+", default=DEFAULT_P_RULES)
    parser.add_argument("--box-limits", type=int, nargs="+", default=DEFAULT_BOX_LIMITS)
    parser.add_argument("--subset-policy", choices=["up_to", "exact"], default="up_to")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _write_candidates_manifest(
    output_dir: Path,
    built: List[CandidateBuildResult],
    skipped: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
) -> None:
    payload = {
        "args": args_dict,
        "generated_count": len(built),
        "skipped_count": len(skipped),
        "generated": [
            {
                **asdict(entry.spec),
                "fixture_file": entry.fixture_file,
            }
            for entry in built
        ],
        "skipped": skipped,
    }
    (output_dir / "candidates_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mark_best_per_condition(rows: List[Dict[str, Any]]) -> None:
    best_by_key: Dict[Tuple[float, int, str], float] = {}
    for row in rows:
        key = (float(row["p_rule"]), int(row["box_limit"]), str(row["subset_policy"]))
        score = float(row["success"])
        if key not in best_by_key or score > best_by_key[key]:
            best_by_key[key] = score

    for row in rows:
        key = (float(row["p_rule"]), int(row["box_limit"]), str(row["subset_policy"]))
        row["best_for_condition"] = bool(abs(float(row["success"]) - best_by_key[key]) <= 1e-15)


def _write_results(output_dir: Path, rows: List[Dict[str, Any]], args_dict: Dict[str, Any], skipped_count: int) -> None:
    columns = [
        "candidate_name",
        "family",
        "n2",
        "k_box",
        "split",
        "split_left",
        "split_right",
        "majority_side",
        "pyramid_side",
        "tie_bandwidth",
        "order",
        "depth_s",
        "majority_len",
        "boxes_used",
        "p_rule",
        "box_limit",
        "subset_policy",
        "success",
        "evaluation_mode",
        "n_train_samples",
        "n_eval_samples",
        "confidence",
        "seed",
        "ci_low",
        "ci_high",
        "stderr",
        "runtime_seconds",
        "best_for_condition",
        "fixture_file",
    ]

    csv_path = output_dir / "optimization_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = output_dir / "optimization_results.json"
    json_path.write_text(
        json.dumps(
            {
                "args": args_dict,
                "row_count": len(rows),
                "skipped_count": skipped_count,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_rows = [row for row in rows if row.get("best_for_condition")]
    summary_lines = [
        "# Hybrid Family Optimization Summary",
        "",
        f"- Evaluated rows: {len(rows)}",
        f"- Skipped invalid configurations: {skipped_count}",
        f"- Best rows (per p_rule/box_limit/subset_policy): {len(best_rows)}",
        "",
    ]
    if best_rows:
        summary_lines.append("## Best Per Condition")
        summary_lines.append("")
        for row in best_rows:
            summary_lines.append(
                "- "
                f"p_rule={row['p_rule']}, box_limit={row['box_limit']}, subset_policy={row['subset_policy']}: "
                f"{row['candidate_name']} (success={row['success']:.12g})"
            )

    (output_dir / "optimization_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def _copy_best_fixtures(output_dir: Path, rows: List[Dict[str, Any]]) -> List[str]:
    fixture_dir = output_dir / "fixtures"
    best_dir = fixture_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    for row in rows:
        if not row.get("best_for_condition"):
            continue
        fixture_file = str(row["fixture_file"])
        src = fixture_dir / fixture_file
        dst = best_dir / fixture_file
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
            copied.append(fixture_file)
    return copied



def enumerate_vertical_candidate_specs(
    n2: int,
    k_box: int,
    orders: Iterable[str],
    depths: Optional[Iterable[int]],
    tie_bandwidth_values: Optional[Iterable[int]],
    exclude_trivial_depths: bool = True,
) -> Tuple[List[CandidateSpec], List[Dict[str, Any]]]:
    generated: List[CandidateSpec] = []
    skipped: List[Dict[str, Any]] = []
    k = int(np.log2(n2))
    if depths is None:
        depth_range = list(range(1, k)) if exclude_trivial_depths else list(range(0, k + 1))
    else:
        depth_range = list(depths)
    for order in orders:
        for s in depth_range:
            if order == "pyr_maj":
                # Section 5.4
                apex_len = n2 // (2 ** s)
                if apex_len < 2:
                    skipped.append({"order": order, "s": s, "reason": "apex_len < 2"})
                    continue
                boxes_used = n2 - apex_len
                if boxes_used > k_box:
                    skipped.append({"order": order, "s": s, "reason": f"requires {boxes_used} boxes, k_box={k_box}"})
                    continue
                majority_len = apex_len
            elif order == "maj_pyr":
                # Section 5.5
                block_count = 2 ** s
                block_size = n2 // block_count
                if block_size % 2 != 0:
                    skipped.append({"order": order, "s": s, "reason": "majority_len not even"})
                    continue
                boxes_used = block_count - 1
                if boxes_used > k_box:
                    skipped.append({"order": order, "s": s, "reason": f"requires {boxes_used} boxes, k_box={k_box}"})
                    continue
                majority_len = block_size
            else:
                skipped.append({"order": order, "s": s, "reason": "unknown order"})
                continue
            # Tie bandwidths
            b_values = list(range(0, majority_len // 2)) if tie_bandwidth_values is None else [b for b in tie_bandwidth_values if 0 <= b < majority_len // 2]
            if not b_values:
                skipped.append({"order": order, "s": s, "reason": "no valid tie_bandwidth"})
                continue
            for b in b_values:
                fam = "hybrid_vertical_pyr_maj" if order == "pyr_maj" else "hybrid_vertical_maj_pyr"
                legacy = CandidateSpec(
                    candidate_name="",
                    family=fam,
                    n2=n2,
                    k_box=k_box,
                    order=order,
                    depth_s=s,
                    majority_len=majority_len,
                    tie_bandwidth=b,
                    boxes_used=boxes_used,
                )
                try:
                    canonical_spec = _legacy_candidate_to_hybrid_spec(legacy)
                    generated.append(
                        CandidateSpec(
                            **{
                                **asdict(legacy),
                                "candidate_name": candidate_name(canonical_spec),
                                "boxes_used": required_boxes(canonical_spec),
                            }
                        )
                    )
                except ValueError as exc:
                    skipped.append(
                        {
                            "order": order,
                            "s": s,
                            "tie_bandwidth": b,
                            "reason": str(exc),
                        }
                    )
    return generated, skipped


def run_optimizer(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize deterministic hybrid-family parameters over explicit sweeps.")
    parser.add_argument("--family", choices=["horizontal", "vertical", "all"], default="horizontal")
    parser.add_argument("--evaluation-mode", choices=["exhaustive", "monte_carlo"], default="exhaustive")
    parser.add_argument("--n2", type=int, default=8)
    parser.add_argument("--k-box", type=int, default=1)
    parser.add_argument("--p-rules", type=float, nargs="+", default=DEFAULT_P_RULES)
    parser.add_argument("--box-limits", type=int, nargs="+", default=DEFAULT_BOX_LIMITS)
    parser.add_argument("--subset-policy", choices=["up_to", "exact"], default="up_to")
    parser.add_argument("--n-train-samples", type=int, default=20_000)
    parser.add_argument("--n-eval-samples", type=int, default=50_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vertical-orders", nargs="+", choices=["pyr_maj", "maj_pyr"], default=["pyr_maj", "maj_pyr"])
    parser.add_argument("--vertical-depths", nargs="+", type=int, default=None)
    parser.add_argument("--tie-bandwidth-values", nargs="+", type=int, default=None)
    parser.add_argument("--split-values", nargs="+", type=int, default=None)
    args = parser.parse_args(argv)

    n2 = int(args.n2)
    k_box = int(args.k_box)
    p_rules = [float(v) for v in args.p_rules]
    box_limits = [int(v) for v in args.box_limits]
    subset_policy = str(args.subset_policy)
    evaluation_mode = str(args.evaluation_mode)
    n_train_samples = int(args.n_train_samples)
    n_eval_samples = int(args.n_eval_samples)
    confidence = float(args.confidence)
    seed = int(args.seed)
    output_dir = Path(args.output_dir)

    # Horizontal sweep params
    tie_values = list(range(0, n2 + 1)) if args.tie_bandwidth_values is None else args.tie_bandwidth_values
    split_values = _default_splits(n2) if args.split_values is None else args.split_values

    # Vertical sweep params
    vertical_orders = args.vertical_orders
    vertical_depths = args.vertical_depths
    vertical_tie_values = args.tie_bandwidth_values
    exclude_trivial_vertical_depths = True

    if args.quick:
        n2 = 4
        k_box = 3
        p_rules = [0.8, 0.9]
        box_limits = [1, 3]
        tie_values = [0, 1]
        split_values = [2]
        vertical_orders = ["pyr_maj", "maj_pyr"]
        vertical_depths = [1]
        vertical_tie_values = [0, 1]
        if evaluation_mode == "monte_carlo":
            n_train_samples = min(n_train_samples, 2_000)
            n_eval_samples = min(n_eval_samples, 5_000)

    _ensure_output_dir(output_dir, overwrite=bool(args.overwrite))
    fixture_dir = output_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: List[CandidateSpec] = []
    all_skipped: List[Dict[str, Any]] = []
    built: List[CandidateBuildResult] = []

    if args.family in ("horizontal", "all"):
        candidates, skipped = enumerate_horizontal_candidate_specs(
            n2=n2,
            k_box=k_box,
            tie_bandwidth_values=tie_values,
            split_values=split_values,
        )
        all_candidates.extend(candidates)
        all_skipped.extend(skipped)
        built.extend([build_candidate_fixture(spec, fixture_dir) for spec in candidates])

    if args.family in ("vertical", "all"):
        candidates, skipped = enumerate_vertical_candidate_specs(
            n2=n2,
            k_box=k_box,
            orders=vertical_orders,
            depths=vertical_depths,
            tie_bandwidth_values=vertical_tie_values,
            exclude_trivial_depths=exclude_trivial_vertical_depths,
        )
        all_candidates.extend(candidates)
        all_skipped.extend(skipped)
        built.extend([build_candidate_fixture(spec, fixture_dir) for spec in candidates])

    if not built:
        raise ValueError("No valid candidates generated for the requested settings.")

    rows: List[Dict[str, Any]] = []
    for entry in built:
        for p_rule in p_rules:
            for box_limit in box_limits:
                row = {
                    "candidate_name": entry.spec.candidate_name,
                    "family": entry.spec.family,
                    "n2": entry.spec.n2,
                    "k_box": entry.spec.k_box,
                    "split": entry.spec.split,
                    "majority_side": entry.spec.majority_side,
                    "pyramid_side": entry.spec.pyramid_side,
                    "tie_bandwidth": entry.spec.tie_bandwidth,
                    "order": entry.spec.order,
                    "depth_s": entry.spec.depth_s,
                    "majority_len": entry.spec.majority_len,
                    "boxes_used": entry.spec.boxes_used,
                    "p_rule": float(p_rule),
                    "box_limit": int(box_limit),
                    "subset_policy": subset_policy,
                    "success": None,
                    "evaluation_mode": evaluation_mode,
                    "n_train_samples": n_train_samples if evaluation_mode == "monte_carlo" else None,
                    "n_eval_samples": n_eval_samples if evaluation_mode == "monte_carlo" else None,
                    "confidence": confidence if evaluation_mode == "monte_carlo" else None,
                    "seed": seed if evaluation_mode == "monte_carlo" else None,
                    "ci_low": None,
                    "ci_high": None,
                    "stderr": None,
                    "runtime_seconds": None,
                    "best_for_condition": False,
                    "fixture_file": entry.fixture_file,
                }
                # Evaluate
                t0 = time.perf_counter()
                if evaluation_mode == "exhaustive":
                    success = evaluate_strategy_exhaustive(
                        entry.strategy,
                        p_rule=float(p_rule),
                        box_limit=int(box_limit),
                        subset_policy=subset_policy,
                    )
                    row["success"] = float(success)
                else:
                    mc_result = evaluate_strategy_mc(
                        entry.strategy,
                        p_rule=float(p_rule),
                        box_limit=int(box_limit),
                        subset_policy=subset_policy,
                        n_train_samples=n_train_samples,
                        n_eval_samples=n_eval_samples,
                        confidence=confidence,
                        seed=seed,
                    )
                    row["success"] = float(mc_result.success)
                    row["ci_low"] = float(mc_result.ci_low)
                    row["ci_high"] = float(mc_result.ci_high)
                    row["stderr"] = float(mc_result.stderr)
                row["runtime_seconds"] = float(time.perf_counter() - t0)
                rows.append(row)

    _mark_best_per_condition(rows)

    args_dict = {
        "family": args.family,
        "evaluation_mode": evaluation_mode,
        "n2": n2,
        "k_box": k_box,
        "p_rules": p_rules,
        "box_limits": box_limits,
        "subset_policy": subset_policy,
        "n_train_samples": n_train_samples if evaluation_mode == "monte_carlo" else None,
        "n_eval_samples": n_eval_samples if evaluation_mode == "monte_carlo" else None,
        "confidence": confidence if evaluation_mode == "monte_carlo" else None,
        "seed": seed if evaluation_mode == "monte_carlo" else None,
        "output_dir": str(output_dir),
        "quick": bool(args.quick),
        "overwrite": bool(args.overwrite),
        "vertical_orders": vertical_orders,
        "vertical_depths": vertical_depths,
        "horizontal_split_values": split_values,
        "tie_bandwidth_values": tie_values,
        "exclude_trivial_vertical_depths": True,
    }
    _write_candidates_manifest(output_dir, built, all_skipped, args_dict)
    _write_results(output_dir, rows, args_dict, skipped_count=len(all_skipped))
    copied = _copy_best_fixtures(output_dir, rows)

    print(
        f"Generated {len(all_candidates)} candidates, skipped {len(all_skipped)} invalid configs, "
        f"evaluated {len(rows)} rows. Output: {output_dir}. Best fixtures copied: {len(copied)}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_optimizer(argv)


if __name__ == "__main__":
    raise SystemExit(main())
