"""Procedural Monte Carlo Bob evaluator for canonical procedural strategies.

Architectural role:
- procedural backend used by strategy_evaluation dispatch
- mirrors dense MC decision/search flow without dense fixture tables

Invariants:
- strategy must satisfy canonical measure_a/comm protocol
- sampling and subset-policy semantics match dense MC backend

Failure behavior:
- raises ValueError for invalid parameters or non-binary procedural outputs
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProceduralBobResult:
    """Normalized procedural-MC result payload consumed by dispatcher/cache."""

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
    elapsed_seconds: float | None = None


# ---------------------------------------------------------------------------
# ProceduralAliceStrategy protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ProceduralAliceStrategy(Protocol):
    """Canonical procedural interface used by procedural MC evaluator."""

    n2: int
    k_box: int

    def measure_a(self, box_idx: int, field: int, out_a_prefix: int) -> int:
        """Return the measurement output bit for this box.

        Args:
            box_idx:      Index of the PR box (0 <= box_idx < k_box).
            field:        Alice's field integer (0 <= field < 2**n2).
            out_a_prefix: Integer encoding the outputs of all preceding boxes
                          (bits 0..box_idx-1 of out_a).

        Returns:
            Binary int (0 or 1).
        """
        ...

    def comm(self, field: int, out_a: int) -> int:
        """Return the communication bit Alice sends to Bob.

        Args:
            field:  Alice's field integer (0 <= field < 2**n2).
            out_a:  Alice's full PR output integer (0 <= out_a < 2**k_box).

        Returns:
            Binary int (0 or 1).
        """
        ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _z_value(confidence: float) -> float:
    z_lookup = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    return z_lookup.get(confidence, 1.959963984540054)


def _subset_masks(k_box: int, box_limit: int, subset_policy: str) -> list[int]:
    masks: list[int] = []
    for mask in range(1 << k_box):
        bits = bin(mask).count("1")
        if subset_policy == "up_to":
            if bits <= box_limit:
                masks.append(mask)
        else:  # "exact"
            if bits == box_limit:
                masks.append(mask)
    return masks


def _validate_inputs(
    strategy: ProceduralAliceStrategy,
    p_rule: float,
    box_limit: int | None,
    subset_policy: str,
    n_train_samples: int,
    n_eval_samples: int,
    confidence: float,
) -> tuple[int, int, int]:
    """Validate and return (n2, k_box, eff_limit)."""
    n2 = int(strategy.n2)
    k_box = int(strategy.k_box)

    if n2 <= 0:
        raise ValueError("n2 must be positive.")
    if k_box <= 0:
        raise ValueError("k_box must be positive.")

    if float(p_rule) < 0.0 or float(p_rule) > 1.0:
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

    return n2, k_box, eff_limit


def _build_sample_arrays_v2(
    strategy: ProceduralAliceStrategy,
    fields: np.ndarray,
    out_as: np.ndarray,
    n_samples: int,
    k_box: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute out_a_bits, m_a_vals, and comm_bits for all samples.

    Args:
        strategy:   ProceduralAliceStrategy instance
        fields:     (n_samples,) sampled field integers
        out_as:     (n_samples,) sampled out_a integers (uniform in [0, 2^k_box))
        n_samples:  number of samples
        k_box:      number of PR boxes

    Returns:
        out_a_bits: (n_samples, k_box) int64
        m_a_vals:   (n_samples, k_box) int64
        comm_bits:  (n_samples,) int64
    """
    out_a_bits = np.empty((n_samples, k_box), dtype=np.int64)
    m_a_vals = np.empty((n_samples, k_box), dtype=np.int64)

    # Extract out_a bits
    for j in range(k_box):
        out_a_bits[:, j] = (out_as >> j) & 1

    # Compute m_a_vals using sequential prefix
    for j in range(k_box):
        if j == 0:
            prefix_arr = np.zeros(n_samples, dtype=np.int64)
        else:
            prefix_arr = out_as & ((1 << j) - 1)

        m_a_col = np.empty(n_samples, dtype=np.int64)
        for i in range(n_samples):
            m_a_col[i] = strategy.measure_a(j, int(fields[i]), int(prefix_arr[i]))

        if not np.all((m_a_col == 0) | (m_a_col == 1)):
            raise ValueError(
                f"measure_a returned non-binary value for box_idx={j}"
            )
        m_a_vals[:, j] = m_a_col

    # Compute comm bits
    comm_bits = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        c = strategy.comm(int(fields[i]), int(out_as[i]))
        if c not in (0, 1):
            raise ValueError(
                f"comm returned non-binary value for field={fields[i]}, out_a={out_as[i]}"
            )
        comm_bits[i] = c

    return out_a_bits, m_a_vals, comm_bits


def _score_candidate_by_counts(
    targets: np.ndarray,
    comm_bits: np.ndarray,
    m_a_vals: np.ndarray,
    out_a_bits: np.ndarray,
    active: list[int],
    meas_mask: int,
    p_rule: float,
    noise_uniforms: np.ndarray,
) -> float:
    """Score a (subset, meas_mask) candidate using pre-drawn noise uniforms."""
    n_samples = targets.shape[0]
    r = len(active)
    out_b_count = 1 << r

    if r == 0:
        visible = comm_bits.copy()
    else:
        out_b_bits = np.empty((n_samples, r), dtype=np.int64)
        for t, box_idx in enumerate(active):
            m_b_j = (meas_mask >> t) & 1
            correct = out_a_bits[:, box_idx] ^ (m_a_vals[:, box_idx] & m_b_j)
            flips = noise_uniforms[:, box_idx] >= p_rule
            out_b_bits[:, t] = np.where(flips, 1 - correct, correct)

        out_b_pattern = np.zeros(n_samples, dtype=np.int64)
        for t in range(r):
            out_b_pattern |= out_b_bits[:, t] << t

        visible = comm_bits * out_b_count + out_b_pattern

    visible_count = 2 * out_b_count
    packed = visible * 2 + targets
    counts = np.bincount(packed, minlength=visible_count * 2).reshape(visible_count, 2)
    best_counts = np.maximum(counts[:, 0], counts[:, 1])
    return float(best_counts.sum()) / float(n_samples)


# ---------------------------------------------------------------------------
# Public evaluator
# ---------------------------------------------------------------------------

def evaluate_strategy_mc_procedural(
    strategy: ProceduralAliceStrategy,
    p_rule: float,
    box_limit: int | None = None,
    subset_policy: str = "up_to",
    n_train_samples: int = 20_000,
    n_eval_samples: int = 50_000,
    confidence: float = 0.95,
    seed: int = 0,
    record_time: bool = False,
) -> ProceduralBobResult:
    """Estimate Bob success by Monte Carlo against canonical procedural strategy API."""
    start_time = time.perf_counter() if record_time else None

    n2, k_box, eff_limit = _validate_inputs(
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

    n_fields = 1 << n2
    outa_count = 1 << k_box

    # --- Training phase ---
    train_fields = rng.integers(0, n_fields, size=n_train_samples, dtype=np.int64)
    train_out_as = rng.integers(0, outa_count, size=n_train_samples, dtype=np.int64)

    train_out_a_bits, train_m_a_vals, train_comm_bits = _build_sample_arrays_v2(
        strategy, train_fields, train_out_as, n_train_samples, k_box
    )
    train_noise_uniforms = rng.random((n_train_samples, k_box))

    selected_subset_mask_by_gun: list[int] = []
    selected_meas_mask_by_gun: list[int] = []

    for gun in range(n2):
        targets = ((train_fields >> gun) & 1).astype(np.int64)

        best_score = -1.0
        best_subset_mask = 0
        best_meas_mask = 0

        for subset_mask in subset_masks:
            active = [j for j in range(k_box) if (subset_mask >> j) & 1]
            r = len(active)

            for meas_mask in range(1 << r):
                score = _score_candidate_by_counts(
                    targets,
                    train_comm_bits,
                    train_m_a_vals,
                    train_out_a_bits,
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

    # --- Evaluation phase ---
    eval_fields = rng.integers(0, n_fields, size=n_eval_samples, dtype=np.int64)
    eval_out_as = rng.integers(0, outa_count, size=n_eval_samples, dtype=np.int64)

    eval_out_a_bits, eval_m_a_vals, eval_comm_bits = _build_sample_arrays_v2(
        strategy, eval_fields, eval_out_as, n_eval_samples, k_box
    )

    total_success = 0.0

    for gun in range(n2):
        subset_mask = selected_subset_mask_by_gun[gun]
        meas_mask = selected_meas_mask_by_gun[gun]
        active = [j for j in range(k_box) if (subset_mask >> j) & 1]

        targets = ((eval_fields >> gun) & 1).astype(np.int64)
        eval_noise_uniforms = rng.random((n_eval_samples, k_box))

        gun_success = _score_candidate_by_counts(
            targets,
            eval_comm_bits,
            eval_m_a_vals,
            eval_out_a_bits,
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

    elapsed = (time.perf_counter() - start_time) if record_time and start_time is not None else None

    return ProceduralBobResult(
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
        elapsed_seconds=float(elapsed) if elapsed is not None else None,
    )
