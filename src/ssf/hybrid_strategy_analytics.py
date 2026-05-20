"""
Closed-form analytical scoring for HybridStrategySpec.

All formulas are derived from first principles and return exact probabilities
without Monte Carlo sampling, fixture construction, or evaluator calls.

Vertical assumption note
------------------------
The vertical closed-form formulas assume that majority-block correctness and
pyramid routing errors combine independently through XOR double-error
cancellation.  Specifically:
  - For vertical maj_pyr: each majority block is assumed to produce an
    independent correct/incorrect bit; the pyramid routes one block's output to
    the final output, and an error in both the majority and pyramid layers
    cancels via XOR, giving success = Pm*Pp + (1-Pm)*(1-Pp).
  - For vertical pyr_maj: the pyramid selects one apex input; the apex majority
    then decides; the same XOR cancellation assumption applies.
These formulas are marked exact=True for now pending empirical validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .hybrid_strategy_spec import HybridStrategySpec, validate_spec


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalyticalResult:
    success: float
    method: str
    exact: bool
    notes: str


# ---------------------------------------------------------------------------
# Majority helpers
# ---------------------------------------------------------------------------

def majority_tie_probability(n: int, tie_bandwidth: int) -> float:
    """Return the probability that a uniform random n-bit string falls in a tie.

    A tie occurs when the Hamming weight t satisfies |2t - n| <= tie_bandwidth.
    For tie_bandwidth=0 and odd n this is 0; for even n it equals C(n, n//2)/2**n.
    """
    total = 1 << n  # 2**n
    tie_count = 0
    for t in range(n + 1):
        if abs(2 * t - n) <= tie_bandwidth:
            tie_count += math.comb(n, t)
    return tie_count / total


def majority_success_probability(n: int, tie_bandwidth: int) -> float:
    """Return the A-side win probability for a majority-vote strategy over n bits.

    For each Hamming weight t:
      - If the weight falls within tie_bandwidth of n/2: success = 0.5
      - If t > n/2: success = t/n  (majority vote is correct)
      - If t < n/2: success = (n-t)/n  (majority vote takes the complement)
    """
    total = 1 << n
    p_success = 0.0
    for t in range(n + 1):
        prob_weight = math.comb(n, t) / total
        if abs(2 * t - n) <= tie_bandwidth:
            p_success += prob_weight * 0.5
        elif t > n / 2:
            p_success += prob_weight * (t / n)
        else:
            p_success += prob_weight * ((n - t) / n)
    return p_success


def majority_success_probability_given_not_tie(n: int, tie_bandwidth: int) -> float:
    """Return majority success conditioned on being outside the tie region.

    This is P(success | not tie). Tie states are excluded from both numerator
    and denominator. If tie_probability is 1, the conditional is undefined.
    """
    tie_probability = majority_tie_probability(n, tie_bandwidth)
    if tie_probability == 1.0:
        raise ValueError("majority_success_probability_given_not_tie undefined when tie_probability == 1.")

    total = 1 << n
    p_success_notie = 0.0
    for t in range(n + 1):
        if abs(2 * t - n) <= tie_bandwidth:
            continue
        prob_weight = math.comb(n, t) / total
        if t > n / 2:
            p_success_notie += prob_weight * (t / n)
        else:
            p_success_notie += prob_weight * ((n - t) / n)

    return p_success_notie / (1.0 - tie_probability)


# ---------------------------------------------------------------------------
# Pyramid helper
# ---------------------------------------------------------------------------

def pyramid_success_probability(depth: int, p_rule: float) -> float:
    """Return the A-side win probability for a pyramid strategy.

    Uses the closed form: (1 + (2*p_rule - 1)**depth) / 2.

    Args:
        depth:  log2(n2), i.e. the number of pyramid layers.
        p_rule: PR-box rule probability in [0, 1].
    """
    return (1.0 + (2.0 * p_rule - 1.0) ** depth) / 2.0


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def analytical_win_rate(spec: HybridStrategySpec, p_rule: float) -> AnalyticalResult:
    """Compute the closed-form analytical win rate for a strategy spec.

    Args:
        spec:   A HybridStrategySpec; must pass validate_spec.
        p_rule: PR-box rule probability; must be in [0, 1].

    Returns:
        AnalyticalResult with success probability, method tag, exactness flag,
        and assumption notes.

    Raises:
        ValueError: if p_rule is out of [0, 1] or spec fails validate_spec.
    """
    if not (0.0 <= p_rule <= 1.0):
        raise ValueError(f"p_rule must be in [0, 1], got {p_rule}.")

    validate_spec(spec)

    if spec.family == "majority":
        assert spec.tie_bandwidth is not None
        success = majority_success_probability(spec.n2, spec.tie_bandwidth)
        return AnalyticalResult(
            success=success,
            method="closed_form",
            exact=True,
            notes="Pure majority over n2 bits with tie_bandwidth resolution.",
        )

    if spec.family == "pyramid":
        depth = int(round(math.log2(spec.n2)))
        success = pyramid_success_probability(depth, p_rule)
        return AnalyticalResult(
            success=success,
            method="closed_form",
            exact=True,
            notes="Pure pyramid of depth log2(n2).",
        )

    if spec.family == "horizontal":
        assert spec.split is not None
        assert spec.order is not None
        assert spec.tie_bandwidth is not None

        if spec.order == "maj_pyr":
            majority_len = spec.split
            pyramid_len = spec.n2 - spec.split
        else:
            pyramid_len = spec.split
            majority_len = spec.n2 - spec.split

        q = majority_len / spec.n2
        t = majority_tie_probability(majority_len, spec.tie_bandwidth)
        pm_notie = majority_success_probability_given_not_tie(majority_len, spec.tie_bandwidth)
        pp = pyramid_success_probability(int(round(math.log2(pyramid_len))), p_rule)

        success = (
            q * ((1.0 - t) * pm_notie + t * 0.5)
            + (1.0 - q) * ((1.0 - t) * 0.5 + t * pp)
        )
        return AnalyticalResult(
            success=success,
            method="closed_form",
            exact=True,
            notes=(
                f"Horizontal hybrid: majority_len={majority_len}, "
                f"pyramid_len={pyramid_len}, order={spec.order}."
            ),
        )

    if spec.family == "vertical":
        assert spec.depth_s is not None
        assert spec.order is not None
        assert spec.tie_bandwidth is not None

        if spec.order == "maj_pyr":
            block_count = 2 ** spec.depth_s
            block_size = spec.n2 // block_count
            pm = majority_success_probability(block_size, spec.tie_bandwidth)
            pp = pyramid_success_probability(spec.depth_s, p_rule)
            success = pm * pp + (1.0 - pm) * (1.0 - pp)
            return AnalyticalResult(
                success=success,
                method="closed_form",
                exact=True,
                notes=(
                    "Closed-form vertical maj_pyr assumes majority-block correctness "
                    "and pyramid routing errors combine through XOR double-error cancellation."
                ),
            )

        else:  # pyr_maj
            apex_len = spec.n2 // (2 ** spec.depth_s)
            pp = pyramid_success_probability(spec.depth_s, p_rule)
            pm = majority_success_probability(apex_len, spec.tie_bandwidth)
            success = pm * pp + (1.0 - pm) * (1.0 - pp)
            return AnalyticalResult(
                success=success,
                method="closed_form",
                exact=True,
                notes=(
                    "Closed-form vertical pyr_maj assumes apex majority correctness "
                    "and pyramid path errors combine through XOR double-error cancellation."
                ),
            )

    raise ValueError(f"Unsupported family: {spec.family!r}")
