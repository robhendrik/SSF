from __future__ import annotations

from math import comb
from typing import Iterable
import warnings

import numpy as np
from numba import njit


WARNING_WORK_UNITS = 50_000_000


def correct_b_output(out_a: int, m_a: int, m_b: int) -> int:
    """Return the rule-correct Bob output for one binary PR-style box.

    Convention used by SSF:
        out_a XOR out_b = m_a AND m_b

    Therefore:
        out_b = out_a XOR (m_a AND m_b)

    Alice outcomes are locally random and are enumerated by the evaluator.
    Bob outcomes are weighted by p_rule relative to this rule-correct value.
    """

    return (int(out_a) ^ (int(m_a) & int(m_b))) & 1


def _subset_masks(k_box: int, box_limit: int, subset_policy: str) -> np.ndarray:
    masks: list[int] = []
    for mask in range(1 << k_box):
        bits = mask.bit_count()
        if subset_policy == "up_to":
            if bits <= box_limit:
                masks.append(mask)
        elif subset_policy == "exact":
            if bits == box_limit:
                masks.append(mask)
        else:
            raise ValueError("subset_policy must be 'up_to' or 'exact'.")
    return np.asarray(masks, dtype=np.int64)


def _pack_f_tables(f_tables: Iterable[np.ndarray], n_fields: int, k_box: int) -> np.ndarray:
    max_prefix_cols = 1 << max(k_box - 1, 0)
    packed = np.zeros((k_box, n_fields, max_prefix_cols), dtype=np.uint8)

    tables = list(f_tables)
    if len(tables) < k_box:
        raise ValueError("f_tables must contain at least k_box tables.")

    for j in range(k_box):
        table = np.asarray(tables[j], dtype=np.uint8)
        expected_cols = 1 << j
        if table.shape != (n_fields, expected_cols):
            raise ValueError(
                f"Invalid f_tables[{j}] shape: {table.shape}, expected {(n_fields, expected_cols)}"
            )
        packed[j, :, :expected_cols] = table

    return packed


def _estimate_work_units(n2: int, k_box: int, eff_limit: int, subset_policy: str) -> int:
    n_fields = 1 << n2
    outa_count = 1 << k_box

    if subset_policy == "up_to":
        subset_count = sum(comb(k_box, r) for r in range(eff_limit + 1))
    elif subset_policy == "exact":
        subset_count = comb(k_box, eff_limit)
    else:
        raise ValueError("subset_policy must be 'up_to' or 'exact'.")

    max_outb_count = 1 << eff_limit
    max_meas_count = 1 << eff_limit
    guns = n2

    return int(guns * subset_count * n_fields * outa_count * max_outb_count * max_meas_count)


def _validate_inputs(
    strategy: dict,
    p_rule: float,
    box_limit: int | None,
    subset_policy: str,
) -> tuple[int, int, np.ndarray, np.ndarray, int]:
    if "n2" not in strategy or "k_box" not in strategy or "f_tables" not in strategy:
        raise ValueError("Strategy must contain n2, k_box, and f_tables.")

    n2 = int(strategy["n2"])
    k_box = int(strategy["k_box"])

    if n2 <= 0:
        raise ValueError("n2 must be positive.")
    if k_box <= 0:
        raise ValueError("k_box must be positive.")

    p = float(p_rule)
    if not 0.0 <= p <= 1.0:
        raise ValueError("p_rule must be in [0, 1].")

    if subset_policy not in ("up_to", "exact"):
        raise ValueError("subset_policy must be 'up_to' or 'exact'.")

    eff_limit = k_box if box_limit is None else int(box_limit)
    if eff_limit < 0 or eff_limit > k_box:
        raise ValueError("box_limit must be in [0, k_box] or None.")

    n_fields = 1 << n2

    comm_arr = strategy.get("comm")
    if comm_arr is None:
        comm_arr = strategy.get("comm_table")
    if comm_arr is None:
        raise ValueError("Strategy must contain 'comm' or legacy 'comm_table'.")

    comm = np.asarray(comm_arr, dtype=np.uint8)
    expected_comm_shape = (n_fields, 1 << k_box)
    if comm.shape != expected_comm_shape:
        raise ValueError(f"Invalid comm shape: {comm.shape}, expected {expected_comm_shape}")

    f_tables = _pack_f_tables(strategy["f_tables"], n_fields, k_box)

    if np.any((comm != 0) & (comm != 1)):
        raise ValueError("comm must be binary.")
    if np.any((f_tables != 0) & (f_tables != 1)):
        raise ValueError("f_tables entries must be binary.")

    return n2, k_box, f_tables, comm, eff_limit


def _evaluate_kernel_python(
    n2: int,
    k_box: int,
    f_tables: np.ndarray,
    comm: np.ndarray,
    p_rule: float,
    subset_masks: np.ndarray,
) -> float:
    n_fields = 1 << n2
    outa_count = 1 << k_box
    base_weight = 1.0 / (n_fields * outa_count)

    total_best = 0.0

    for gun in range(n2):
        best_for_gun = 0.0

        for subset_mask_raw in subset_masks:
            subset_mask = int(subset_mask_raw)
            active = [j for j in range(k_box) if (subset_mask >> j) & 1]
            r = len(active)
            meas_count = 1 << r
            outb_count = 1 << r
            visible_count = 2 * outb_count  # comm bit times active outB pattern

            best_subset = 0.0

            for meas_mask in range(meas_count):
                m_b_values = [((meas_mask >> t) & 1) for t in range(r)]
                mass = np.zeros((visible_count, 2), dtype=np.float64)

                for field in range(n_fields):
                    target = (field >> gun) & 1

                    for out_a in range(outa_count):
                        comm_bit = int(comm[field, out_a])

                        m_a_vals = [0] * k_box
                        for j in range(k_box):
                            prefix = 0 if j == 0 else (out_a & ((1 << j) - 1))
                            m_a_vals[j] = int(f_tables[j, field, prefix])

                        for out_b_mask in range(outb_count):
                            weight = base_weight

                            for t, box_idx in enumerate(active):
                                out_a_j = (out_a >> box_idx) & 1
                                out_b_j = (out_b_mask >> t) & 1
                                corr = correct_b_output(out_a_j, m_a_vals[box_idx], m_b_values[t])
                                weight *= p_rule if out_b_j == corr else (1.0 - p_rule)

                            visible = comm_bit * outb_count + out_b_mask
                            mass[visible, target] += weight

                success = float(np.maximum(mass[:, 0], mass[:, 1]).sum())
                if success > best_subset:
                    best_subset = success

            if best_subset > best_for_gun:
                best_for_gun = best_subset

        total_best += best_for_gun

    return total_best / n2

@njit(cache=True)
def _evaluate_kernel_numba(
    n2: int,
    k_box: int,
    f_tables: np.ndarray,
    comm: np.ndarray,
    p_rule: float,
    subset_masks: np.ndarray,
) -> float:
        n_fields = 1 << n2
        outa_count = 1 << k_box
        base_weight = 1.0 / (n_fields * outa_count)

        total_best = 0.0

        for gun in range(n2):
            best_for_gun = 0.0

            for subset_i in range(subset_masks.shape[0]):
                subset_mask = int(subset_masks[subset_i])

                active_idx = np.empty(k_box, dtype=np.int64)
                r = 0
                for j in range(k_box):
                    if (subset_mask >> j) & 1:
                        active_idx[r] = j
                        r += 1

                meas_count = 1 << r
                outb_count = 1 << r
                visible_count = 2 * outb_count

                best_subset = 0.0

                for meas_mask in range(meas_count):
                    m_b_values = np.empty(r, dtype=np.int64)
                    for t in range(r):
                        m_b_values[t] = (meas_mask >> t) & 1

                    mass = np.zeros((visible_count, 2), dtype=np.float64)

                    for field in range(n_fields):
                        target = (field >> gun) & 1

                        for out_a in range(outa_count):
                            comm_bit = int(comm[field, out_a])

                            m_a_vals = np.empty(k_box, dtype=np.int64)
                            for j in range(k_box):
                                if j == 0:
                                    prefix = 0
                                else:
                                    prefix = out_a & ((1 << j) - 1)
                                m_a_vals[j] = int(f_tables[j, field, prefix])

                            for out_b_mask in range(outb_count):
                                weight = base_weight

                                for t in range(r):
                                    box_idx = int(active_idx[t])
                                    out_a_j = (out_a >> box_idx) & 1
                                    out_b_j = (out_b_mask >> t) & 1
                                    corr = (out_a_j ^ (int(m_a_vals[box_idx]) & int(m_b_values[t]))) & 1
                                    if out_b_j == corr:
                                        weight *= p_rule
                                    else:
                                        weight *= 1.0 - p_rule

                                visible = comm_bit * outb_count + out_b_mask
                                mass[visible, target] += weight

                    success = 0.0
                    for v in range(visible_count):
                        if mass[v, 0] >= mass[v, 1]:
                            success += mass[v, 0]
                        else:
                            success += mass[v, 1]

                    if success > best_subset:
                        best_subset = success

                if best_subset > best_for_gun:
                    best_for_gun = best_subset

            total_best += best_for_gun

        return total_best / n2


def evaluate_strategy_exhaustive(
    strategy: dict,
    p_rule: float,
    box_limit: int | None = None,
    subset_policy: str = "up_to",
) -> float:
    """Evaluate the best exhaustive restricted Bob response for a fixed Alice strategy.

    Bob always observes the gun index and Alice's communication bit. For each gun,
    Bob may additionally use a subset of Bob-side box outputs controlled by
    ``box_limit`` and ``subset_policy``. For every allowed subset, all binary Bob
    measurement inputs on that subset are tried, and the final shoot decision is
    solved by weighted majority over Bob-visible states.

    No strategy names, manifest expected scores, or reference-strategy shortcuts
    are used here. The result depends only on the lookup tables and p_rule.
    """

    n2, k_box, f_tables, comm, eff_limit = _validate_inputs(
        strategy, p_rule, box_limit, subset_policy
    )

    work_units = _estimate_work_units(n2, k_box, eff_limit, subset_policy)
    if work_units > WARNING_WORK_UNITS:
        warnings.warn(
            f"Exhaustive Bob evaluation may be expensive: n2={n2}, k_box={k_box}, "
            f"box_limit={box_limit}, estimated_work_units={work_units}.",
            RuntimeWarning,
            stacklevel=2,
        )

    subsets = _subset_masks(k_box, eff_limit, subset_policy)
    if subsets.size == 0:
        raise ValueError("No valid subsets for the given box_limit/subset_policy.")

    score = _evaluate_kernel_numba(n2, k_box, f_tables, comm, float(p_rule), subsets)

    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    return float(score)
