"""
Plot closed-form analytical win-rate curves for representative hybrid strategy families.

Produces one figure per n2 value (4, 8, 16) with up to 9 curves:
  1. pure majority
  2. pure pyramid
  3. fixed horizontal hybrid (majority = n2-2, pyramid = 2, order=maj_pyr)
  4-6. sampled diverse horizontal hybrids (from enumerated valid pool)
  7-9. sampled diverse vertical hybrids (from enumerated valid pool)

All specs are validated through validate_spec() before use.
All win rates are computed via analytical_win_rate() — no MC, no fixtures.

Note on n2=4:
  The valid horizontal/vertical parameter space for n2=4 is very small
  (only 2 horizontal and 2 vertical distinct analytical specs exist).
  For n2=4 the figure will contain fewer than 9 curves; a warning is printed.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Resolve src/ on the path so this script works without installation.
SSF_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SSF_DIR / "src"))

from ssf.hybrid_strategy_spec import (
    HybridStrategySpec,
    candidate_name,
    is_power_of_two,
    required_boxes,
    validate_spec,
    even_tie_values,
)
from ssf.hybrid_strategy_analytics import analytical_win_rate

DEFAULT_OUTPUT_DIR = SSF_DIR / "results" / "analytics_curves"
DEFAULT_DPI = 150
P_VALUES = [round(0.50 + i * 0.01, 2) for i in range(51)]  # 0.50 .. 1.00


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------

def build_fixed_specs(n2: int) -> list[HybridStrategySpec]:
    """Return the three fixed curves: majority, pyramid, fixed horizontal."""
    # Majority (no PR boxes used)
    maj = HybridStrategySpec(family="majority", n2=n2, k_box=n2 - 1, tie_bandwidth=0)
    validate_spec(maj)

    # Pyramid (requires power-of-two n2)
    pyr = HybridStrategySpec(family="pyramid", n2=n2, k_box=n2 - 1)
    validate_spec(pyr)

    # Fixed horizontal: majority side = n2-2, pyramid side = 2
    fixed_horiz = HybridStrategySpec(
        family="horizontal",
        n2=n2,
        k_box=n2 - 1,
        split=n2 - 2,
        order="maj_pyr",
        tie_bandwidth=0,
    )
    validate_spec(fixed_horiz)

    return [maj, pyr, fixed_horiz]


def _enumerate_horizontal_pool(n2: int) -> list[HybridStrategySpec]:
    """Enumerate all analytically distinct horizontal specs for this n2, using only even tie_bandwidth."""
    pool: list[HybridStrategySpec] = []
    for split in range(1, n2):
        for order in ("maj_pyr", "pyr_maj"):
            if order == "maj_pyr":
                majority_len = split
                pyramid_len = n2 - split
            else:
                pyramid_len = split
                majority_len = n2 - split

            if majority_len % 2 != 0:
                continue
            if not is_power_of_two(pyramid_len):
                continue
            if majority_len < 2:
                continue

            for bw in even_tie_values(majority_len):
                req = required_boxes(
                    HybridStrategySpec(
                        family="horizontal",
                        n2=n2,
                        k_box=n2 - 1,
                        split=split,
                        order=order,
                        tie_bandwidth=bw,
                    )
                )
                spec = HybridStrategySpec(
                    family="horizontal",
                    n2=n2,
                    k_box=max(req, n2 - 1),
                    split=split,
                    order=order,
                    tie_bandwidth=bw,
                )
                validate_spec(spec)
                pool.append(spec)
    return pool


def _enumerate_vertical_pool(n2: int) -> list[HybridStrategySpec]:
    """Enumerate all analytically distinct vertical specs for this n2."""
    if not is_power_of_two(n2):
        return []

    pool: list[HybridStrategySpec] = []
    max_depth = int(round(math.log2(n2)))  # log2(n2)
    for depth_s in range(1, max_depth):
        for order in ("maj_pyr", "pyr_maj"):
            if order == "maj_pyr":
                block_count = 2 ** depth_s
                majority_len = n2 // block_count
            else:
                majority_len = n2 // (2 ** depth_s)

            if majority_len % 2 != 0:
                continue
            if majority_len < 2:
                continue

            for bw in even_tie_values(majority_len):
                req = required_boxes(
                    HybridStrategySpec(
                        family="vertical",
                        n2=n2,
                        k_box=n2 - 1,
                        depth_s=depth_s,
                        order=order,
                        tie_bandwidth=bw,
                    )
                )
                spec = HybridStrategySpec(
                    family="vertical",
                    n2=n2,
                    k_box=max(req, n2 - 1),
                    depth_s=depth_s,
                    order=order,
                    tie_bandwidth=bw,
                )
                validate_spec(spec)
                pool.append(spec)
    return pool


def _select_diverse(pool: list[HybridStrategySpec], count: int, exclude: list[HybridStrategySpec]) -> list[HybridStrategySpec]:
    """Select up to `count` specs from pool, skipping any in exclude.

    Diversity is ensured by deduplicating on (split, depth_s, order, tie_bandwidth).
    Returns fewer than `count` specs if the pool is smaller.
    """
    exclude_keys = {
        (s.split, s.depth_s, s.order, s.tie_bandwidth) for s in exclude
    }
    seen_keys: set[tuple] = set()
    diverse: list[HybridStrategySpec] = []
    shuffled = list(pool)
    random.shuffle(shuffled)
    for spec in shuffled:
        key = (spec.split, spec.depth_s, spec.order, spec.tie_bandwidth)
        if key in exclude_keys or key in seen_keys:
            continue
        seen_keys.add(key)
        diverse.append(spec)
        if len(diverse) == count:
            break
    return diverse


def random_horizontal_specs(n2: int, count: int, exclude: list[HybridStrategySpec]) -> list[HybridStrategySpec]:
    pool = _enumerate_horizontal_pool(n2)
    if not pool:
        raise RuntimeError(f"No valid horizontal specs found for n2={n2}.")
    return _select_diverse(pool, count, exclude)


def random_vertical_specs(n2: int, count: int, exclude: list[HybridStrategySpec]) -> list[HybridStrategySpec]:
    pool = _enumerate_vertical_pool(n2)
    if not pool:
        raise RuntimeError(f"No valid vertical specs found for n2={n2}.")
    return _select_diverse(pool, count, exclude)


# ---------------------------------------------------------------------------
# Curve computation
# ---------------------------------------------------------------------------

def compute_curve(spec: HybridStrategySpec, p_values: Sequence[float]) -> list[float]:
    """Return analytical win rates for a spec over a sequence of p_rule values."""
    return [analytical_win_rate(spec, p).success for p in p_values]


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure(
    n2: int,
    p_values: Sequence[float],
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    fixed = build_fixed_specs(n2)
    horiz_random = random_horizontal_specs(n2, 3, exclude=fixed)
    vert_random = random_vertical_specs(n2, 3, exclude=fixed + horiz_random)

    all_specs = fixed + horiz_random + vert_random

    target = 9
    actual = len(all_specs)
    if actual < target:
        print(
            f"  WARNING: n2={n2} — only {actual} analytically distinct specs available "
            f"(target was {target}). The parameter space is too small for {target} unique curves."
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    linestyles = ["-", "--", "-.", ":"]
    markers = ["", "o", "s", "^", "v", "D", "P", "X", "*"]

    for idx, spec in enumerate(all_specs):
        y = compute_curve(spec, p_values)
        label = candidate_name(spec)
        ls = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]
        ax.plot(
            p_values,
            y,
            linestyle=ls,
            marker=marker,
            markevery=10,
            markersize=4,
            label=label,
        )

    ax.set_title(f"Analytical Win-Rate Curves — n2={n2}", fontsize=13)
    ax.set_xlabel("p_rule", fontsize=11)
    ax.set_ylabel("Analytical win rate", fontsize=11)
    ax.set_xlim(min(p_values), max(p_values))
    ax.set_ylim(0.48, 1.02)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"  Saved: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot closed-form analytical win-rate curves for hybrid strategy families.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save PNG figures. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PNG resolution. Default: {DEFAULT_DPI}",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display figures interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir: Path = args.output_dir

    n2_values = [4, 8, 16]
    for n2 in n2_values:
        print(f"Generating figure for n2={n2} ...")
        output_path = output_dir / f"analytics_curves_n2_{n2}.png"
        generate_figure(n2, P_VALUES, output_path, dpi=args.dpi, show=args.show)

    print("Done.")


if __name__ == "__main__":
    main()
