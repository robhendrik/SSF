from __future__ import annotations

"""DEPRECATED: use scripts/generate/generate_a_strategy_fixtures.py. This script is kept temporarily for migration only."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# TODO(path-migration): keep legacy output path until all legacy consumers are removed.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "a_strategies_reference"
P_RULE_VALUES = (0.50, 0.65, 0.77, 0.80, 0.90, 1.00)


def _bit(field: int, idx: int) -> int:
    return (field >> idx) & 1


def _is_power_of_two(value: int) -> bool:
    return value >= 1 and (value & (value - 1)) == 0


def _validate_pyramid_args(n2: int, k_box: int) -> None:
    if n2 < 2 or not _is_power_of_two(n2):
        raise ValueError("n2 must be a power of two and >= 2 for canonical pyramid.")
    if k_box < n2 - 1:
        raise ValueError("k_box must be >= n2-1 for canonical pyramid.")


def _generate_pyramid_measurements(n2: int) -> List[int]:
    """Return leaf-bit masks for canonical pyramid boxes in layer-first order."""
    masks: List[int] = []
    depth = int(math.log2(n2))
    for layer in range(depth):
        span = 1 << (layer + 1)
        n_boxes_layer = n2 // span
        for pos in range(n_boxes_layer):
            start = pos * span
            mask = 0
            for bit in range(start, start + span):
                mask |= (1 << bit)
            masks.append(mask)
    return masks


def _majority_comm_bit(field: int, n2: int) -> int:
    ones = field.bit_count()
    zeros = n2 - ones
    if ones > zeros:
        return 1
    return 0


def _majority_success_probability(n: int) -> float:
    total = 0.0
    denom = float(2**n)
    for t in range(n + 1):
        prob_weight = math.comb(n, t) / denom
        if t > n / 2:
            success_given_weight = t / n
        elif t < n / 2:
            success_given_weight = (n - t) / n
        else:
            # Tie rule returns 0. Against uniformly random queried bit this is 1/2 accurate.
            success_given_weight = 0.5
        total += prob_weight * success_given_weight
    return float(total)


def _pyramid_depth2_success_probability(p_rule: float) -> float:
    return float((1.0 + (2.0 * p_rule - 1.0) ** 2) / 2.0)


def build_f_tables_majority(n2: int, k_box: int) -> List[np.ndarray]:
    rows = 2**n2
    return [np.zeros((rows, 2**d), dtype=np.uint8) for d in range(k_box)]


def build_comm_majority(n2: int, k_box: int) -> np.ndarray:
    rows = 2**n2
    cols = 2**k_box
    comm = np.zeros((rows, cols), dtype=np.uint8)
    for field in range(rows):
        comm[field, :] = _majority_comm_bit(field, n2)
    return comm


def build_f_tables_pyramid(n2: int, k_box: int) -> List[np.ndarray]:
    _validate_pyramid_args(n2, k_box)

    # Preserve exact historical fixture behavior.
    if n2 == 4 and k_box == 3:
        rows = 2**n2
        f0 = np.zeros((rows, 1), dtype=np.uint8)
        f1 = np.zeros((rows, 2), dtype=np.uint8)
        f2 = np.zeros((rows, 4), dtype=np.uint8)

        for field in range(rows):
            x0 = _bit(field, 0)
            x1 = _bit(field, 1)
            x2 = _bit(field, 2)
            x3 = _bit(field, 3)

            f0[field, 0] = x0 ^ x1

            for a0 in range(2):
                f1[field, a0] = x2 ^ x3

            for prefix in range(4):
                a0 = prefix & 1
                a1 = (prefix >> 1) & 1
                f2[field, prefix] = (x0 ^ a0) ^ (x2 ^ a1)

        return [f0, f1, f2]

    rows = 2**n2
    f_tables = [np.zeros((rows, 2**d), dtype=np.uint8) for d in range(k_box)]
    measurement_masks = _generate_pyramid_measurements(n2)
    required_boxes = n2 - 1

    for d in range(required_boxes):
        mask = measurement_masks[d]
        table = f_tables[d]
        for field in range(rows):
            value = (field & mask).bit_count() & 1
            table[field, :] = value

    return f_tables


def build_comm_pyramid(n2: int, k_box: int) -> np.ndarray:
    _validate_pyramid_args(n2, k_box)

    # Preserve exact historical fixture behavior.
    if n2 == 4 and k_box == 3:
        rows = 2**n2
        cols = 2**k_box
        comm = np.zeros((rows, cols), dtype=np.uint8)

        for field in range(rows):
            x0 = _bit(field, 0)
            for out_a in range(cols):
                a0 = (out_a >> 0) & 1
                a2 = (out_a >> 2) & 1
                comm[field, out_a] = x0 ^ a0 ^ a2

        return comm

    rows = 2**n2
    cols = 2**k_box
    comm = np.zeros((rows, cols), dtype=np.uint8)
    apex_mask = (1 << n2) - 1
    for field in range(rows):
        comm_bit = (field & apex_mask).bit_count() & 1
        comm[field, :] = comm_bit
    return comm


def _analytical_success_majority(n2: int) -> Dict[str, float]:
    value = _majority_success_probability(n2)
    return {f"{p:.2f}": value for p in P_RULE_VALUES}


def _analytical_success_pyramid_depth2() -> Dict[str, float]:
    return {f"{p:.2f}": _pyramid_depth2_success_probability(p) for p in P_RULE_VALUES}


def _write_fixture(
    output_dir: Path,
    name: str,
    n2: int,
    k_box: int,
    f_tables: List[np.ndarray],
    comm_table: np.ndarray,
) -> str:
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


def _manifest_entry(
    name: str,
    family: str,
    n2: int,
    k_box: int,
    boxes_used: int,
    file_name: str,
    analytical_success: Dict[str, float],
    formula_notes: str,
) -> Dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "n2": n2,
        "k_box": k_box,
        "boxes_used": boxes_used,
        "unused_boxes": k_box - boxes_used,
        "file": file_name,
        "analytical_success": analytical_success,
        "formula_notes": formula_notes,
    }


def generate_reference_fixtures(output_dir: Path, overwrite: bool = False) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for npz_path in output_dir.glob("*.npz"):
            npz_path.unlink()
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()

    manifest: List[Dict[str, Any]] = []

    majority_configs = [(4, 3), (8, 3), (8, 1)]
    for n2, k_box in majority_configs:
        name = f"majority_n2_{n2}_k_{k_box}"
        f_tables = build_f_tables_majority(n2, k_box)
        comm = build_comm_majority(n2, k_box)
        file_name = _write_fixture(output_dir, name, n2, k_box, f_tables, comm)
        manifest.append(
            _manifest_entry(
                name=name,
                family="majority",
                n2=n2,
                k_box=k_box,
                boxes_used=0,
                file_name=file_name,
                analytical_success=_analytical_success_majority(n2),
                formula_notes="Exact majority baseline with tie->0; independent of p_rule.",
            )
        )

    pyr_name = "pyramid_n2_4_k_3"
    f_tables = build_f_tables_pyramid(4, 3)
    comm = build_comm_pyramid(4, 3)
    file_name = _write_fixture(output_dir, pyr_name, 4, 3, f_tables, comm)
    manifest.append(
        _manifest_entry(
            name=pyr_name,
            family="pyramid",
            n2=4,
            k_box=3,
            boxes_used=3,
            file_name=file_name,
            analytical_success=_analytical_success_pyramid_depth2(),
            formula_notes="Depth-2 pyramid composition: (1 + (2p-1)^2)/2.",
        )
    )

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate canonical baseline A-side reference fixtures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for .npz fixtures and manifest.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing .npz and manifest in output dir before generating.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = generate_reference_fixtures(args.output_dir, overwrite=args.overwrite)
    print(f"Generated {len(manifest)} reference fixtures in {args.output_dir}")


if __name__ == "__main__":
    main()

