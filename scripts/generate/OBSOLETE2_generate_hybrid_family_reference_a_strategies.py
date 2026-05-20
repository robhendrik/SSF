"""Generate deterministic A-side hybrid-family reference fixtures."""

from __future__ import annotations

"""DEPRECATED: use scripts/generate/generate_a_strategy_fixtures.py. This script is kept temporarily for migration only."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

BitExpr = Callable[[int, int], int]


def _bit(field: int, i: int) -> int:
    return (field >> i) & 1


def _out_bit(out_a: int, d: int) -> int:
    return (out_a >> d) & 1


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _majority_vote(bits: Iterable[int], tie_bandwidth: int, c_tie: int = 0) -> int:
    seq = list(bits)
    ones = int(sum(seq))
    zeros = len(seq) - ones
    if abs(ones - zeros) <= tie_bandwidth:
        return c_tie
    return 1 if ones > zeros else 0


def _majority_decision_or_tie(bits: Iterable[int], tie_bandwidth: int) -> Optional[int]:
    seq = list(bits)
    ones = int(sum(seq))
    zeros = len(seq) - ones
    if abs(ones - zeros) <= tie_bandwidth:
        return None
    return 1 if ones > zeros else 0


def pyramid_correctness_probability(s: int, p_rule: float) -> float:
    if s == 0:
        return 1.0
    return float((1.0 + (2.0 * p_rule - 1.0) ** s) / 2.0)


def majority_correct_probability(n: int, tie_bandwidth: int, c_tie: int = 0) -> float:
    del c_tie
    denom = float(2**n)
    total = 0.0
    for t in range(n + 1):
        prob_weight = math.comb(n, t) / denom
        d = abs(2 * t - n)
        if d <= tie_bandwidth:
            success_given_weight = 0.5
        elif t > n / 2:
            success_given_weight = t / n
        else:
            success_given_weight = (n - t) / n
        total += prob_weight * success_given_weight
    return float(total)


def _allocate_tables(n2: int, k_box: int) -> List[np.ndarray]:
    rows = 2**n2
    return [np.zeros((rows, 2**d), dtype=np.uint8) for d in range(k_box)]


def _build_pyramid_layers(
    n2: int,
    f_tables: List[np.ndarray],
    expressions: Sequence[BitExpr],
    start_box: int,
    max_layers: Optional[int] = None,
) -> tuple[List[BitExpr], int]:
    if len(expressions) == 0:
        raise ValueError("Pyramid requires at least one input expression.")
    if not _is_power_of_two(len(expressions)):
        raise ValueError("Pyramid input length must be a power of two.")

    rows = 2**n2
    current = list(expressions)
    box = int(start_box)
    used = 0
    built_layers = 0
    while len(current) > 1:
        if max_layers is not None and built_layers >= max_layers:
            break

        next_level: List[BitExpr] = []
        for idx in range(0, len(current), 2):
            if box >= len(f_tables):
                raise ValueError("Insufficient k_box for requested pyramid layers.")

            left_expr = current[idx]
            right_expr = current[idx + 1]
            box_idx = box

            def _new_expr(field: int, out_a: int, left: BitExpr = left_expr, right: BitExpr = right_expr, d: int = box_idx) -> int:
                return left(field, out_a) ^ right(field, out_a) ^ _out_bit(out_a, d)

            table = f_tables[box_idx]
            for field in range(rows):
                row = table[field]
                for prefix in range(2**box_idx):
                    row[prefix] = int(_new_expr(field, prefix))

            next_level.append(_new_expr)
            box += 1
            used += 1

        current = next_level
        built_layers += 1

    return current, used


def _horizontal_generic(
    n2: int,
    k_box: int,
    left: List[int],
    right: List[int],
    tie_bandwidth: int,
    c_tie: int,
    majority_side: str,
) -> Dict[str, Any]:
    del c_tie
    rows = 2**n2
    cols = 2**k_box
    if majority_side == "left":
        majority_idxs = list(left)
        pyramid_idxs = list(right)
    elif majority_side == "right":
        majority_idxs = list(right)
        pyramid_idxs = list(left)
    else:
        raise ValueError("majority_side must be 'left' or 'right'.")

    majority_len = len(majority_idxs)
    pyramid_len = len(pyramid_idxs)
    if majority_len % 2 != 0:
        raise ValueError("Horizontal configuration is illegal: majority side length must be even.")
    if not _is_power_of_two(pyramid_len):
        raise ValueError("Horizontal configuration is illegal: pyramid side length must be a power of two.")

    boxes_used = pyramid_len - 1
    if boxes_used > k_box:
        raise ValueError(
            f"Horizontal configuration requires {boxes_used} boxes but k_box={k_box}."
        )

    f_tables = _allocate_tables(n2, k_box)
    comm = np.zeros((rows, cols), dtype=np.uint8)

    leaf_exprs: List[BitExpr] = []
    for idx in pyramid_idxs:
        leaf_exprs.append(lambda field, out_a, bit_idx=idx: _bit(field, bit_idx))

    [final_expr], used = _build_pyramid_layers(n2, f_tables, leaf_exprs, start_box=0)
    if used != boxes_used:
        raise ValueError("Unexpected horizontal pyramid layer count.")

    for field in range(rows):
        maj_decision = _majority_decision_or_tie((_bit(field, i) for i in majority_idxs), tie_bandwidth)
        for out_a in range(cols):
            comm[field, out_a] = int(maj_decision) if maj_decision is not None else int(final_expr(field, out_a))

    return {"f_tables": f_tables, "comm_table": comm, "boxes_used": boxes_used}


def _horizontal_majority_left_pyramid_right(
    n2: int,
    k_box: int,
    left: List[int],
    right: List[int],
    tie_bandwidth: int,
    c_tie: int,
) -> Dict[str, Any]:
    return _horizontal_generic(n2, k_box, left, right, tie_bandwidth, c_tie, majority_side="left")


def _horizontal_pyramid_left_majority_right(
    n2: int,
    k_box: int,
    left: List[int],
    right: List[int],
    tie_bandwidth: int,
    c_tie: int,
) -> Dict[str, Any]:
    return _horizontal_generic(n2, k_box, left, right, tie_bandwidth, c_tie, majority_side="right")


def _build_vertical_maj_pyr(n2: int, k_box: int, s: int, tie_bandwidth: int, c_tie: int) -> Dict[str, Any]:
    if not _is_power_of_two(n2):
        raise ValueError("Vertical maj/pyr requires n2 to be a power of two.")
    max_depth = int(math.log2(n2))
    if not (0 < s < max_depth):
        raise ValueError(f"Vertical maj/pyr requires 0 < s < log2(n2); got s={s}, n2={n2}.")

    block_count = 2**s
    block_size = n2 // block_count
    if block_size % 2 != 0:
        raise ValueError("Vertical maj/pyr requires an even block size.")

    boxes_used = block_count - 1
    if boxes_used > k_box:
        raise ValueError(f"Vertical maj/pyr requires {boxes_used} boxes but k_box={k_box}.")

    f_tables = _allocate_tables(n2, k_box)
    rows = 2**n2
    cols = 2**k_box
    comm = np.zeros((rows, cols), dtype=np.uint8)

    y_exprs: List[BitExpr] = []
    for block in range(block_count):
        start = block * block_size
        stop = start + block_size

        def _block_expr(field: int, out_a: int, lo=start, hi=stop) -> int:
            del out_a
            return _majority_vote((_bit(field, i) for i in range(lo, hi)), tie_bandwidth, c_tie)

        y_exprs.append(_block_expr)

    [final_expr], used = _build_pyramid_layers(n2, f_tables, y_exprs, start_box=0)
    if used != boxes_used:
        raise ValueError("Unexpected vertical maj/pyr pyramid layer count.")

    for field in range(rows):
        for out_a in range(cols):
            comm[field, out_a] = int(final_expr(field, out_a))

    return {"f_tables": f_tables, "comm_table": comm, "boxes_used": boxes_used}


def _build_vertical_pyr_maj(n2: int, k_box: int, s: int, tie_bandwidth: int, c_tie: int) -> Dict[str, Any]:
    if not _is_power_of_two(n2):
        raise ValueError("Vertical pyr/maj requires n2 to be a power of two.")
    max_depth = int(math.log2(n2))
    if not (0 < s < max_depth):
        raise ValueError(f"Vertical pyr/maj requires 0 < s < log2(n2); got s={s}, n2={n2}.")

    apex_len = n2 // (2**s)
    if apex_len % 2 != 0:
        raise ValueError("Vertical pyr/maj requires an even apex length.")

    boxes_used = n2 - apex_len
    if boxes_used > k_box:
        raise ValueError(f"Vertical pyr/maj requires {boxes_used} boxes but k_box={k_box}.")

    f_tables = _allocate_tables(n2, k_box)
    rows = 2**n2
    cols = 2**k_box
    comm = np.zeros((rows, cols), dtype=np.uint8)

    leaves: List[BitExpr] = []
    for idx in range(n2):
        leaves.append(lambda field, out_a, bit_idx=idx: _bit(field, bit_idx))

    apex_exprs, used = _build_pyramid_layers(n2, f_tables, leaves, start_box=0, max_layers=s)
    if used != boxes_used:
        raise ValueError("Unexpected vertical pyr/maj pyramid layer count.")

    for field in range(rows):
        for out_a in range(cols):
            comm[field, out_a] = _majority_vote((expr(field, out_a) for expr in apex_exprs), tie_bandwidth, c_tie)

    return {"f_tables": f_tables, "comm_table": comm, "boxes_used": boxes_used}


def _analytical_horizontal(majority_len: int, pyramid_depth: int, tie_bandwidth: int, p_rule: float) -> float:
    left_majority = majority_correct_probability(majority_len, tie_bandwidth, c_tie=0)
    right_pyramid = pyramid_correctness_probability(pyramid_depth, p_rule)
    return float(0.5 * left_majority + 0.5 * right_pyramid)


def _analytical_vertical(s: int, tie_bandwidth: int, p_rule: float) -> float:
    maj2 = majority_correct_probability(2, tie_bandwidth, c_tie=0)
    pyr = pyramid_correctness_probability(s, p_rule)
    return float(0.5 * maj2 + 0.5 * pyr)


def _format_analytical(values: Dict[float, float]) -> Dict[str, float]:
    return {f"{p:.2f}": float(v) for p, v in values.items()}


def _write_fixture(output_dir: Path, name: str, n2: int, k_box: int, f_tables: List[np.ndarray], comm_table: np.ndarray) -> str:
    file_name = f"{name}.npz"
    payload: Dict[str, np.ndarray] = {
        "n2": np.array(n2, dtype=np.int64),
        "k_box": np.array(k_box, dtype=np.int64),
        "comm_table": comm_table.astype(np.uint8, copy=False),
    }
    for d, table in enumerate(f_tables):
        payload[f"f_table_{d}"] = table.astype(np.uint8, copy=False)
    np.savez(output_dir / file_name, **payload)
    return file_name


def _dense_fixture_allowed(n2: int, k_box: int, max_n2_for_dense_fixtures: int) -> tuple[bool, Optional[str]]:
    if n2 > max_n2_for_dense_fixtures:
        return False, f"n2={n2} exceeds max dense threshold {max_n2_for_dense_fixtures}"
    comm_cells = (2**n2) * (2**k_box)
    if comm_cells > 16_000_000:
        return False, f"comm_table too large ({2**n2}x{2**k_box})"
    return True, None


def _make_manifest_entry(
    *,
    name: str,
    family: str,
    n2: int,
    k_box: int,
    boxes_used: int,
    tie_bandwidth: int,
    c_tie: int,
    file_name: str,
    analytical_success: Dict[str, float],
    formula_notes: str,
    split_left: Optional[int] = None,
    split_right: Optional[int] = None,
    majority_side: Optional[str] = None,
    pyramid_side: Optional[str] = None,
    order: Optional[str] = None,
    depth_s: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "n2": n2,
        "k_box": k_box,
        "boxes_used": boxes_used,
        "unused_boxes": k_box - boxes_used,
        "split_left": split_left,
        "split_right": split_right,
        "majority_side": majority_side,
        "pyramid_side": pyramid_side,
        "order": order,
        "depth_s": depth_s,
        "tie_bandwidth": tie_bandwidth,
        "c_tie": c_tie,
        "legal_config": True,
        "file": file_name,
        "analytical_success": analytical_success,
        "formula_notes": formula_notes,
    }


def _legacy_specs() -> List[Dict[str, Any]]:
    return [
        {
            "family": "horizontal",
            "majority_side": "left",
            "n2": 4,
            "k_box": 3,
            "split_left": 2,
            "split_right": 2,
            "tie_bandwidth": 0,
        },
        {
            "family": "horizontal",
            "majority_side": "right",
            "n2": 4,
            "k_box": 3,
            "split_left": 2,
            "split_right": 2,
            "tie_bandwidth": 0,
        },
        {"family": "vertical", "order": "pyr_maj", "n2": 4, "k_box": 3, "s": 1, "tie_bandwidth": 0},
        {"family": "vertical", "order": "maj_pyr", "n2": 4, "k_box": 3, "s": 1, "tie_bandwidth": 0},
        {
            "family": "horizontal",
            "majority_side": "left",
            "n2": 8,
            "k_box": 3,
            "split_left": 4,
            "split_right": 4,
            "tie_bandwidth": 0,
        },
        {
            "family": "horizontal",
            "majority_side": "right",
            "n2": 8,
            "k_box": 3,
            "split_left": 4,
            "split_right": 4,
            "tie_bandwidth": 0,
        },
        {"family": "vertical", "order": "maj_pyr", "n2": 8, "k_box": 3, "s": 2, "tie_bandwidth": 0},
        {
            "family": "horizontal",
            "majority_side": "left",
            "n2": 8,
            "k_box": 1,
            "split_left": 6,
            "split_right": 2,
            "tie_bandwidth": 2,
        },
        {
            "family": "horizontal",
            "majority_side": "left",
            "n2": 4,
            "k_box": 1,
            "split_left": 2,
            "split_right": 2,
            "tie_bandwidth": 0,
        },
        {
            "family": "horizontal",
            "majority_side": "right",
            "n2": 4,
            "k_box": 1,
            "split_left": 2,
            "split_right": 2,
            "tie_bandwidth": 0,
        },
        {"family": "vertical", "order": "maj_pyr", "n2": 4, "k_box": 1, "s": 1, "tie_bandwidth": 0},
        {
            "family": "horizontal",
            "majority_side": "left",
            "n2": 8,
            "k_box": 7,
            "split_left": 4,
            "split_right": 4,
            "tie_bandwidth": 0,
        },
        {
            "family": "horizontal",
            "majority_side": "right",
            "n2": 8,
            "k_box": 7,
            "split_left": 4,
            "split_right": 4,
            "tie_bandwidth": 0,
        },
        {"family": "vertical", "order": "maj_pyr", "n2": 8, "k_box": 7, "s": 2, "tie_bandwidth": 0},
        {
            "family": "horizontal",
            "majority_side": "left",
            "n2": 8,
            "k_box": 7,
            "split_left": 6,
            "split_right": 2,
            "tie_bandwidth": 2,
        },
        {"family": "vertical", "order": "pyr_maj", "n2": 8, "k_box": 7, "s": 2, "tie_bandwidth": 0},
    ]


def _iter_all_legal_specs(n2_values: Sequence[int], k_box_values: Sequence[int]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for n2 in sorted(set(int(v) for v in n2_values)):
        if not _is_power_of_two(n2):
            continue
        max_depth = int(math.log2(n2))
        for k_box in sorted(set(int(v) for v in k_box_values)):
            for split_left in range(1, n2):
                split_right = n2 - split_left
                for majority_side in ("left", "right"):
                    majority_len = split_left if majority_side == "left" else split_right
                    pyramid_len = split_right if majority_side == "left" else split_left
                    if majority_len % 2 != 0 or not _is_power_of_two(pyramid_len):
                        continue
                    boxes_used = pyramid_len - 1
                    if boxes_used > k_box:
                        continue
                    max_b = max(0, majority_len // 2)
                    for tie_bandwidth in range(max_b):
                        specs.append(
                            {
                                "family": "horizontal",
                                "majority_side": majority_side,
                                "n2": n2,
                                "k_box": k_box,
                                "split_left": split_left,
                                "split_right": split_right,
                                "tie_bandwidth": tie_bandwidth,
                            }
                        )

            for s in range(1, max_depth):
                block_count = 2**s
                block_size = n2 // block_count
                if block_size % 2 == 0 and block_count - 1 <= k_box:
                    max_b = max(0, block_size // 2)
                    for tie_bandwidth in range(max_b):
                        specs.append(
                            {
                                "family": "vertical",
                                "order": "maj_pyr",
                                "n2": n2,
                                "k_box": k_box,
                                "s": s,
                                "tie_bandwidth": tie_bandwidth,
                            }
                        )

                apex_len = n2 // (2**s)
                if apex_len % 2 == 0 and (n2 - apex_len) <= k_box:
                    max_b = max(0, apex_len // 2)
                    for tie_bandwidth in range(max_b):
                        specs.append(
                            {
                                "family": "vertical",
                                "order": "pyr_maj",
                                "n2": n2,
                                "k_box": k_box,
                                "s": s,
                                "tie_bandwidth": tie_bandwidth,
                            }
                        )
    return specs


def _horizontal_name(majority_side: str, n2: int, k_box: int, split_left: int, split_right: int, tie_bandwidth: int) -> str:
    if majority_side == "left":
        return f"hybrid_horiz_majority_left_pyramid_right_n2_{n2}_k_{k_box}_split_{split_left}_{split_right}_b_{tie_bandwidth}"
    return f"hybrid_horiz_pyramid_left_majority_right_n2_{n2}_k_{k_box}_split_{split_left}_{split_right}_b_{tie_bandwidth}"


def _vertical_name(order: str, n2: int, k_box: int, s: int, tie_bandwidth: int) -> str:
    return f"hybrid_vert_{order}_n2_{n2}_k_{k_box}_s_{s}_b_{tie_bandwidth}"


def generate_hybrid_family_fixtures(
    output_dir: Path,
    p_rule_values: tuple[float, ...] = (0.5, 0.65, 0.77, 1.0),
    allow_required_box_override: bool = False,
    n2_values: Sequence[int] = (4, 8),
    k_box_values: Sequence[int] = (1, 3, 7),
    include_all_legal: bool = False,
    max_n2_for_dense_fixtures: int = 8,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    c_tie = 0

    base_specs = _iter_all_legal_specs(n2_values, k_box_values) if include_all_legal else _legacy_specs()

    if allow_required_box_override:
        base_specs.append(
            {
                "family": "vertical",
                "order": "pyr_maj",
                "n2": 8,
                "k_box": 6,
                "s": 2,
                "tie_bandwidth": 0,
            }
        )

    seen_names: set[str] = set()
    for spec in base_specs:
        n2 = int(spec["n2"])
        k_box = int(spec["k_box"])
        tie_bandwidth = int(spec.get("tie_bandwidth", 0))

        if not include_all_legal:
            if n2 not in {int(v) for v in n2_values}:
                continue
            if k_box not in {int(v) for v in k_box_values} and not (
                allow_required_box_override and spec.get("order") == "pyr_maj" and n2 == 8 and k_box == 6
            ):
                continue

        dense_ok, reason = _dense_fixture_allowed(n2, k_box, int(max_n2_for_dense_fixtures))
        if not dense_ok:
            skipped.append({"spec": spec, "reason": reason})
            continue

        try:
            if spec["family"] == "horizontal":
                split_left = int(spec["split_left"])
                split_right = int(spec["split_right"])
                left = list(range(split_left))
                right = list(range(split_left, split_left + split_right))
                majority_side = str(spec["majority_side"])

                if majority_side == "left":
                    built = _horizontal_majority_left_pyramid_right(n2, k_box, left, right, tie_bandwidth, c_tie)
                    majority_len = split_left
                    pyramid_len = split_right
                    formula_notes = f"Horizontal mix: left majority({majority_len},b={tie_bandwidth}) and right {pyramid_len}-bit pyramid fallback."
                    pyramid_side = "right"
                else:
                    built = _horizontal_pyramid_left_majority_right(n2, k_box, left, right, tie_bandwidth, c_tie)
                    majority_len = split_right
                    pyramid_len = split_left
                    formula_notes = f"Horizontal mix: right majority({majority_len},b={tie_bandwidth}) and left {pyramid_len}-bit pyramid fallback."
                    pyramid_side = "left"

                name = _horizontal_name(majority_side, n2, k_box, split_left, split_right, tie_bandwidth)
                if name in seen_names:
                    continue
                seen_names.add(name)

                file_name = _write_fixture(output_dir, name, n2, k_box, built["f_tables"], built["comm_table"])
                analytical = _format_analytical(
                    {
                        p: _analytical_horizontal(majority_len, int(math.log2(pyramid_len)), tie_bandwidth, p)
                        for p in p_rule_values
                    }
                )
                manifest.append(
                    _make_manifest_entry(
                        name=name,
                        family="hybrid_horizontal",
                        n2=n2,
                        k_box=k_box,
                        boxes_used=int(built["boxes_used"]),
                        tie_bandwidth=tie_bandwidth,
                        c_tie=c_tie,
                        file_name=file_name,
                        analytical_success=analytical,
                        formula_notes=formula_notes,
                        split_left=split_left,
                        split_right=split_right,
                        majority_side=majority_side,
                        pyramid_side=pyramid_side,
                    )
                )
                continue

            if spec["family"] == "vertical":
                order = str(spec["order"])
                s = int(spec["s"])
                if order == "maj_pyr":
                    built = _build_vertical_maj_pyr(n2, k_box, s, tie_bandwidth, c_tie)
                    formula_notes = "Vertical maj/pyr approximation: block majorities then generic pyramid merge."
                else:
                    built = _build_vertical_pyr_maj(n2, k_box, s, tie_bandwidth, c_tie)
                    formula_notes = "Vertical pyr/maj approximation: first s pyramid layers then majority over apex sequence."

                name = _vertical_name(order, n2, k_box, s, tie_bandwidth)
                if name in seen_names:
                    continue
                seen_names.add(name)

                file_name = _write_fixture(output_dir, name, n2, k_box, built["f_tables"], built["comm_table"])
                analytical = _format_analytical({p: _analytical_vertical(s, tie_bandwidth, p) for p in p_rule_values})
                manifest.append(
                    _make_manifest_entry(
                        name=name,
                        family="hybrid_vertical",
                        n2=n2,
                        k_box=k_box,
                        boxes_used=int(built["boxes_used"]),
                        tie_bandwidth=tie_bandwidth,
                        c_tie=c_tie,
                        file_name=file_name,
                        analytical_success=analytical,
                        formula_notes=formula_notes,
                        order=order,
                        depth_s=s,
                    )
                )
                continue

        except ValueError as exc:
            skipped.append({"spec": spec, "reason": str(exc)})
            continue

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if skipped:
        with (output_dir / "skipped.json").open("w", encoding="utf-8") as handle:
            json.dump(skipped, handle, indent=2)

    return manifest


# TODO(path-migration): keep legacy output path until all legacy consumers are removed.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "a_strategies_hybrid_family"
DEFAULT_P_RULE_VALUES = (0.5, 0.65, 0.77, 1.0)
DEFAULT_N2_VALUES = (4, 8)
DEFAULT_K_BOX_VALUES = (1, 3, 7)

__all__ = [
    "_build_vertical_maj_pyr",
    "_build_vertical_pyr_maj",
    "_horizontal_majority_left_pyramid_right",
    "_horizontal_pyramid_left_majority_right",
    "generate_hybrid_family_fixtures",
    "majority_correct_probability",
    "pyramid_correctness_probability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate A-side-only hybrid family reference fixtures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where .npz fixtures and manifest.json are written.",
    )
    parser.add_argument(
        "--allow-required-box-override",
        action="store_true",
        help="Allow adding required-box override fixture(s) outside requested k_box values.",
    )
    parser.add_argument(
        "--n2-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_N2_VALUES),
        help="n2 values to include (default: 4 8).",
    )
    parser.add_argument(
        "--k-box-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_BOX_VALUES),
        help="k_box values to include (default: 1 3 7).",
    )
    parser.add_argument(
        "--include-all-legal",
        action="store_true",
        help="Generate all legal horizontal/vertical hybrid configs for selected n2 and k_box values.",
    )
    parser.add_argument(
        "--max-n2-for-dense-fixtures",
        type=int,
        default=8,
        help="Skip dense fixture generation above this n2 threshold (default: 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_hybrid_family_fixtures(
        output_dir=args.output_dir,
        p_rule_values=DEFAULT_P_RULE_VALUES,
        allow_required_box_override=args.allow_required_box_override,
        n2_values=args.n2_values,
        k_box_values=args.k_box_values,
        include_all_legal=args.include_all_legal,
        max_n2_for_dense_fixtures=int(args.max_n2_for_dense_fixtures),
    )
    print(f"Generated {len(manifest)} hybrid-family A-side fixtures in {args.output_dir}")


if __name__ == "__main__":
    main()
