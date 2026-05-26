"""Dense Monte Carlo Bob-side evaluator for canonical dense strategy fixtures.

Architectural role:
- probabilistic dense backend used by strategy_evaluation dispatch
- approximates exhaustive backend when exact evaluation is expensive

Invariants:
- same visible-state decision structure as exhaustive backend
- two-phase train/eval sampling with reproducible RNG seed

Failure behavior:
- raises ValueError for malformed fixtures or invalid sampling parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MonteCarloBobResult:
    """Normalized dense-MC result payload consumed by evaluation dispatcher/cache."""

    success: float
    ci_low: float
    ci_high: float
    stderr: float
    n_train_samples: int
    n_eval_samples: int
    confidence: float
    seed: int
    p_rule: float
    box_limit: int | None
    subset_policy: str


def _correct_b_output(out_a: int, m_a: int, m_b: int) -> int:
    return (int(out_a) ^ (int(m_a) & int(m_b))) & 1


def _subset_masks(k_box: int, box_limit: int, subset_policy: str) -> list[int]:
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
    return masks


def _z_value(confidence: float) -> float:
    z_lookup = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    return z_lookup.get(confidence, 1.959963984540054)


def _pack_f_tables(f_tables: Iterable[np.ndarray], n_fields: int, k_box: int) -> np.ndarray:
    max_prefix_cols = 1 << max(k_box - 1, 0)
    packed = np.zeros((k_box, n_fields, max_prefix_cols), dtype=np.uint8)

    tables = list(f_tables)
    if len(tables) < k_box:
        raise ValueError("f_tables must contain at least k_box tables.")

    for d in range(k_box):
        table = np.asarray(tables[d], dtype=np.uint8)
        expected_shape = (n_fields, 1 << d)
        if table.shape != expected_shape:
            raise ValueError(
                f"Invalid f_tables[{d}] shape: {table.shape}, expected {expected_shape}"
            )
        packed[d, :, : (1 << d)] = table

    return packed


def _validate_and_pack(
    strategy: dict,
    p_rule: float,
    box_limit: int | None,
    subset_policy: str,
    n_train_samples: int,
    n_eval_samples: int,
    confidence: float,
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
    if p < 0.0 or p > 1.0:
        raise ValueError("p_rule must be in [0, 1].")

    if subset_policy not in ("up_to", "exact"):
        raise ValueError("subset_policy must be 'up_to' or 'exact'.")

    eff_limit = k_box if box_limit is None else int(box_limit)
    if eff_limit < 0 or eff_limit > k_box:
        raise ValueError("box_limit must be in [0, k_box] or None.")

    if int(n_train_samples) <= 0:
        raise ValueError("n_train_samples must be positive.")
    if int(n_eval_samples) <= 0:
        raise ValueError("n_eval_samples must be positive.")
    if not (0.0 < float(confidence) < 1.0):
        raise ValueError("confidence must be in (0, 1).")

    n_fields = 1 << n2

    comm_arr = strategy.get("comm")
    if comm_arr is None:
        comm_arr = strategy.get("comm_table")
    if comm_arr is None:
        raise ValueError("Strategy must contain 'comm' or 'comm_table'.")

    comm = np.asarray(comm_arr, dtype=np.uint8)
    expected_comm_shape = (n_fields, 1 << k_box)
    if comm.shape != expected_comm_shape:
        raise ValueError(f"Invalid comm shape: {comm.shape}, expected {expected_comm_shape}")

    f_tables = _pack_f_tables(strategy["f_tables"], n_fields, k_box)

    if np.any((comm != 0) & (comm != 1)):
        raise ValueError("comm entries must be binary.")
    if np.any((f_tables != 0) & (f_tables != 1)):
        raise ValueError("f_tables entries must be binary.")

    return n2, k_box, f_tables, comm, eff_limit


def _prepare_sample_views(
    fields: np.ndarray,
    out_as: np.ndarray,
    gun: int,
    k_box: int,
    f_tables: np.ndarray,
    comm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = ((fields >> gun) & 1).astype(np.int64)
    comm_bits = comm[fields, out_as].astype(np.int64)

    m_a_vals = np.empty((fields.shape[0], k_box), dtype=np.int64)
    out_a_bits = np.empty((fields.shape[0], k_box), dtype=np.int64)

    for j in range(k_box):
        if j == 0:
            prefixes = np.zeros(fields.shape[0], dtype=np.int64)
        else:
            prefixes = out_as & ((1 << j) - 1)

        m_a_vals[:, j] = f_tables[j, fields, prefixes].astype(np.int64)
        out_a_bits[:, j] = ((out_as >> j) & 1).astype(np.int64)

    return targets, comm_bits, m_a_vals, out_a_bits


def _sample_visible_indices(
    rng: np.random.Generator | None,
    comm_bits: np.ndarray,
    m_a_vals: np.ndarray,
    out_a_bits: np.ndarray,
    active: list[int],
    meas_mask: int,
    p_rule: float,
    noise_uniforms: np.ndarray | None = None,
) -> np.ndarray:
    n_samples = comm_bits.shape[0]
    r = len(active)
    outb_count = 1 << r

    if r == 0:
        return comm_bits

    out_b_bits = np.empty((n_samples, r), dtype=np.int64)

    for t, box_idx in enumerate(active):
        m_b_j = (meas_mask >> t) & 1
        correct = out_a_bits[:, box_idx] ^ (m_a_vals[:, box_idx] & m_b_j)
        if noise_uniforms is None:
            if rng is None:
                raise ValueError("rng is required when noise_uniforms is not provided.")
            flips = rng.random(n_samples) >= p_rule
        else:
            flips = noise_uniforms[:, box_idx] >= p_rule
        out_b_bits[:, t] = np.where(flips, 1 - correct, correct)

    out_b_pattern = np.zeros(n_samples, dtype=np.int64)
    for t in range(r):
        out_b_pattern |= out_b_bits[:, t] << t

    return comm_bits * outb_count + out_b_pattern


def _score_candidate_by_counts(
    rng: np.random.Generator | None,
    targets: np.ndarray,
    comm_bits: np.ndarray,
    m_a_vals: np.ndarray,
    out_a_bits: np.ndarray,
    active: list[int],
    meas_mask: int,
    p_rule: float,
    noise_uniforms: np.ndarray | None = None,
) -> float:
    visible = _sample_visible_indices(
        rng,
        comm_bits,
        m_a_vals,
        out_a_bits,
        active,
        meas_mask,
        p_rule,
        noise_uniforms,
    )

    visible_count = 2 * (1 << len(active))
    packed = visible * 2 + targets
    counts = np.bincount(packed, minlength=visible_count * 2).reshape(visible_count, 2)

    best_counts = np.maximum(counts[:, 0], counts[:, 1])
    return float(best_counts.sum()) / float(targets.shape[0])


def evaluate_strategy_mc(
    strategy: dict,
    p_rule: float,
    box_limit: int | None = None,
    subset_policy: str = "up_to",
    n_train_samples: int = 20_000,
    n_eval_samples: int = 50_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> MonteCarloBobResult:
    """Estimate Bob success via canonical dense Monte Carlo train/eval procedure."""
    n2, k_box, f_tables, comm, eff_limit = _validate_and_pack(
        strategy,
        p_rule,
        box_limit,
        subset_policy,
        n_train_samples,
        n_eval_samples,
        confidence,
    )

    subset_masks = _subset_masks(k_box, eff_limit, subset_policy)
    if not subset_masks:
        raise ValueError("No valid subsets for given box_limit/subset_policy.")

    rng = np.random.default_rng(seed)

    selected_subset_mask_by_gun: list[int] = []
    selected_meas_mask_by_gun: list[int] = []

    n_fields = 1 << n2
    outa_count = 1 << k_box

    train_fields = rng.integers(0, n_fields, size=n_train_samples, dtype=np.int64)
    train_out_as = rng.integers(0, outa_count, size=n_train_samples, dtype=np.int64)

    for gun in range(n2):
        targets, comm_bits, m_a_vals, out_a_bits = _prepare_sample_views(
            train_fields, train_out_as, gun, k_box, f_tables, comm
        )
        train_noise_uniforms = rng.random((n_train_samples, k_box))

        best_score = -1.0
        best_subset_mask = 0
        best_meas_mask = 0

        for subset_mask in subset_masks:
            active = [j for j in range(k_box) if (subset_mask >> j) & 1]
            r = len(active)

            for meas_mask in range(1 << r):
                score = _score_candidate_by_counts(
                    None,
                    targets,
                    comm_bits,
                    m_a_vals,
                    out_a_bits,
                    active,
                    meas_mask,
                    float(p_rule),
                    train_noise_uniforms,
                )
                if score > best_score:
                    best_score = score
                    best_subset_mask = subset_mask
                    best_meas_mask = meas_mask

        selected_subset_mask_by_gun.append(best_subset_mask)
        selected_meas_mask_by_gun.append(best_meas_mask)

    eval_fields = rng.integers(0, n_fields, size=n_eval_samples, dtype=np.int64)
    eval_out_as = rng.integers(0, outa_count, size=n_eval_samples, dtype=np.int64)

    total_success = 0.0

    for gun in range(n2):
        subset_mask = selected_subset_mask_by_gun[gun]
        meas_mask = selected_meas_mask_by_gun[gun]
        active = [j for j in range(k_box) if (subset_mask >> j) & 1]

        targets, comm_bits, m_a_vals, out_a_bits = _prepare_sample_views(
            eval_fields, eval_out_as, gun, k_box, f_tables, comm
        )
        eval_noise_uniforms = rng.random((n_eval_samples, k_box))
        gun_success = _score_candidate_by_counts(
            None,
            targets,
            comm_bits,
            m_a_vals,
            out_a_bits,
            active,
            meas_mask,
            float(p_rule),
            eval_noise_uniforms,
        )
        total_success += gun_success

    success = total_success / n2

    n_total = float(n_eval_samples * n2)
    stderr = float(np.sqrt(max(success * (1.0 - success), 0.0) / n_total))
    z = _z_value(float(confidence))
    ci_low = max(0.0, success - z * stderr)
    ci_high = min(1.0, success + z * stderr)

    return MonteCarloBobResult(
        success=float(success),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        stderr=float(stderr),
        n_train_samples=int(n_train_samples),
        n_eval_samples=int(n_eval_samples),
        confidence=float(confidence),
        seed=int(seed),
        p_rule=float(p_rule),
        box_limit=box_limit,
        subset_policy=subset_policy,
    )
