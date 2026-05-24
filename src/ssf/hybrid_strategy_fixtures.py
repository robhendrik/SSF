"""Dense fixture construction and serialization for canonical HybridStrategySpec strategies."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .hybrid_strategy_spec import HybridStrategySpec, validate_spec, required_boxes, candidate_name


def _bit(x: int, i: int) -> int:
    return (x >> i) & 1

def _out_bit(out_a: int, d: int) -> int:
    return (out_a >> d) & 1

def _majority_or_tie(bits: List[int], tie_bandwidth: int, tie_value: int) -> int:
    n = len(bits)
    s = sum(bits)
    d = abs(2 * s - n)
    if d <= tie_bandwidth:
        return tie_value
    return 1 if s > n / 2 else 0


def _allocate_tables(n2: int, k_box: int):
    f_tables = [np.zeros((2 ** n2, 2 ** d), dtype=np.uint8) for d in range(k_box)]
    comm_table = np.zeros((2 ** n2, 2 ** k_box), dtype=np.uint8)
    return f_tables, comm_table

@dataclass(frozen=True)
class DenseStrategyFixture:
    spec: HybridStrategySpec
    n2: int
    k_box: int
    f_tables: List[np.ndarray]
    comm_table: np.ndarray
    boxes_used: int

    def as_strategy_dict(self) -> dict:
        return {
            "n2": self.n2,
            "k_box": self.k_box,
            "f_tables": self.f_tables,
            "comm": self.comm_table,
            "comm_table": self.comm_table,
        }


def build_strategy_fixture(spec: HybridStrategySpec) -> DenseStrategyFixture:
    spec = validate_spec(spec)
    n2 = spec.n2
    k_box = spec.k_box
    req_boxes = required_boxes(spec)
    if req_boxes > k_box:
        raise ValueError(f"Spec requires at least {req_boxes} boxes, got {k_box}.")
    f_tables, comm_table = _allocate_tables(n2, k_box)
    # Fill f_tables and comm_table according to spec.family
    if spec.family == "majority":
        # All f_tables are zero, comm_table is majority result (independent of out_a)
        for field in range(2 ** n2):
            bits = [_bit(field, i) for i in range(n2)]
            c = _majority_or_tie(bits, spec.tie_bandwidth, spec.tie_value)
            comm_table[field, :] = c
        boxes_used = 0
    elif spec.family == "pyramid":
        _build_pyramid_dense(spec, f_tables, comm_table)
        boxes_used = n2 - 1
    elif spec.family == "horizontal":
        _build_horizontal_dense(spec, f_tables, comm_table)
        boxes_used = required_boxes(spec)
    elif spec.family == "vertical":
        _build_vertical_dense(spec, f_tables, comm_table)
        boxes_used = required_boxes(spec)
    else:
        raise ValueError(f"Unknown family: {spec.family}")
    # Zero out any unused f_tables
    for d in range(boxes_used, k_box):
        f_tables[d][:] = 0
    return DenseStrategyFixture(spec, n2, k_box, f_tables, comm_table, boxes_used)


def write_strategy_fixture(spec: HybridStrategySpec, output_dir: Path) -> Path:
    fixture = build_strategy_fixture(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = candidate_name(spec) + ".npz"
    out_path = output_dir / fname
    arrays = {f"f_table_{i}": arr for i, arr in enumerate(fixture.f_tables)}
    arrays["comm_table"] = fixture.comm_table
    arrays["n2"] = fixture.n2
    arrays["k_box"] = fixture.k_box
    np.savez_compressed(out_path, **arrays)
    return out_path

# --- Internal builders ---

def _build_pyramid_dense(spec, f_tables, comm_table):
    n2 = spec.n2
    k_box = spec.k_box
    # Canonical breadth-first pyramid box assignment
    for field in range(2 ** n2):
        field_bits = [_bit(field, i) for i in range(n2)]
        for out_a in range(2 ** k_box):
            units = field_bits[:]
            # First (n2-1) boxes form the pyramid to a single bit
            for d in range(n2 - 1):
                left = units[_pyramid_left(d)]
                right = units[_pyramid_right(d)]
                a_d = left ^ right
                out_a_prefix = out_a & ((1 << d) - 1)
                f_tables[d][field, out_a_prefix] = a_d
                o_A_d = _out_bit(out_a, d)
                units.append(left ^ o_A_d)
            comm_table[field, out_a] = units[-1]


def _pyramid_left(d):
    return 2 * d

def _pyramid_right(d):
    return 2 * d + 1

def _build_horizontal_dense(spec, f_tables, comm_table):
    n2 = spec.n2
    k_box = spec.k_box
    split = spec.split
    order = spec.order
    tie_bandwidth = spec.tie_bandwidth
    tie_value = spec.tie_value
    if order == "maj_pyr":
        majority_idx = list(range(split))
        pyramid_idx = list(range(split, n2))
    else:
        pyramid_idx = list(range(split))
        majority_idx = list(range(split, n2))
    pyramid_len = len(pyramid_idx)
    pyramid_boxes = pyramid_len - 1 if pyramid_len > 1 else 0
    # Build local pyramid for the branch
    for field in range(2 ** n2):
        field_bits = [_bit(field, i) for i in range(n2)]
        maj_bits = [field_bits[i] for i in majority_idx]
        pyr_bits = [field_bits[i] for i in pyramid_idx]
        for out_a in range(2 ** k_box):
            # Compute majority result
            s = sum(maj_bits)
            n_maj = len(maj_bits)
            is_tie = abs(2 * s - n_maj) <= tie_bandwidth
            maj = 1 if s > n_maj / 2 else 0
            # Compute pyramid apex if needed
            if pyramid_len > 1:
                # Build local units for pyramid branch
                units = [pyr_bits[i] for i in range(pyramid_len)]
                for d in range(pyramid_boxes):
                    left = units[_pyramid_left(d)]
                    right = units[_pyramid_right(d)]
                    a_d = left ^ right
                    out_a_prefix = out_a & ((1 << d) - 1)
                    f_tables[d][field, out_a_prefix] = a_d
                    o_A_d = _out_bit(out_a, d)
                    units.append(left ^ o_A_d)
                pyramid_apex = units[-1]
            else:
                pyramid_apex = pyr_bits[0] if pyramid_len == 1 else 0
            # comm logic
            comm_table[field, out_a] = maj if not is_tie else pyramid_apex
    # Zero out unused f_tables
    for d in range(pyramid_boxes, k_box):
        f_tables[d][:] = 0

def _build_pyramid_layers_dense(initial_values, start_box, max_layers, field, out_a, f_tables):
    """
    Build up to max_layers of a canonical pyramid, breadth-first, left-to-right.
    Returns the resulting apex values and the next box index.
    """
    current_values = initial_values[:]
    next_box = start_box
    for layer in range(max_layers):
        layer_outputs = []
        for i in range(0, len(current_values), 2):
            left = current_values[i]
            right = current_values[i + 1]
            a_d = left ^ right
            out_a_prefix = out_a & ((1 << next_box) - 1)
            f_tables[next_box][field, out_a_prefix] = a_d
            o_A_d = _out_bit(out_a, next_box)
            layer_outputs.append(left ^ o_A_d)
            next_box += 1
        current_values = layer_outputs
    return current_values, next_box

def _build_vertical_dense(spec, f_tables, comm_table):
    n2 = spec.n2
    k_box = spec.k_box
    depth_s = spec.depth_s
    order = spec.order
    tie_bandwidth = spec.tie_bandwidth
    tie_value = spec.tie_value

    if order == "maj_pyr":
        # Existing maj_pyr logic remains unchanged
        block_count = 2 ** depth_s
        block_size = n2 // block_count
        for field in range(2 ** n2):
            field_bits = [_bit(field, i) for i in range(n2)]
            block_outs = []
            for b in range(block_count):
                block = field_bits[b * block_size : (b + 1) * block_size]
                block_outs.append(_majority_or_tie(block, tie_bandwidth, tie_value))
            for out_a in range(2 ** k_box):
                units = block_outs[:]
                for d in range(block_count - 1):
                    left = units[_pyramid_left(d)]
                    right = units[_pyramid_right(d)]
                    a_d = left ^ right
                    out_a_prefix = out_a & ((1 << d) - 1)
                    f_tables[d][field, out_a_prefix] = a_d
                    o_A_d = _out_bit(out_a, d)
                    units.append(left ^ o_A_d)
                comm_table[field, out_a] = units[-1]
        # Zero out unused f_tables
        for d in range(block_count - 1, k_box):
            f_tables[d][:] = 0
    else:  # pyr_maj
        apex_len = n2 // (2 ** depth_s)
        boxes_used = n2 - apex_len
        for field in range(2 ** n2):
            field_bits = [_bit(field, i) for i in range(n2)]
            for out_a in range(2 ** k_box):
                apex_values, next_box = _build_pyramid_layers_dense(
                    initial_values=field_bits,
                    start_box=0,
                    max_layers=depth_s,
                    field=field,
                    out_a=out_a,
                    f_tables=f_tables,
                )
                assert len(apex_values) == apex_len
                comm_table[field, out_a] = _majority_or_tie(apex_values, tie_bandwidth, tie_value)
        for d in range(boxes_used, k_box):
            f_tables[d][:] = 0
