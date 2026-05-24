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
import csv
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
from ssf.strategy_search import enumerate_legal_specs, exhaustive_search

DEFAULT_OUTPUT_DIR = SSF_DIR / "results" / "analytics_curves"
DEFAULT_DPI = 150
P_VALUES = [round(0.50 + i * 0.01, 2) for i in range(51)]  # 0.50 .. 1.00


def _default_search_k_box_values(n2: int) -> list[int]:
    if n2 == 4:
        return [1, 3]
    if n2 == 8:
        return [1, 3, 7]
    if n2 == 16:
        return [1, 3, 7, 15]
    # Fallback: odd values up to n2-1, plus n2-1.
    values = [k for k in range(1, n2, 2)]
    if (n2 - 1) not in values and n2 > 1:
        values.append(n2 - 1)
    return sorted(set(values))


def _compute_optimal_search_envelope(
    *,
    plot_n2: int,
    p_values: Sequence[float],
    search_n2_values: list[int] | None,
    search_k_box_values: list[int] | None,
    search_families: list[str],
) -> tuple[list[float], list[tuple[float, float, str]]]:
    n2_values = [plot_n2] if search_n2_values is None else [int(v) for v in search_n2_values]
    k_box_values = (
        _default_search_k_box_values(plot_n2)
        if search_k_box_values is None
        else [int(v) for v in search_k_box_values]
    )

    specs = enumerate_legal_specs(
        n2_values=n2_values,
        k_box_values=k_box_values,
        families=search_families,
    )
    if not specs:
        raise RuntimeError(
            f"No legal specs found for optimal search overlay (plot_n2={plot_n2})."
        )

    envelope: list[float] = []
    summary_rows: list[tuple[float, float, str]] = []
    for p in p_values:
        best = exhaustive_search(
            specs=specs,
            p_rule=float(p),
            evaluation_mode="analytical",
            top_k=1,
        )
        if not best:
            raise RuntimeError(f"No search results produced for p_rule={p}.")
        winner = best[0]
        envelope.append(float(winner.success))
        summary_rows.append((float(p), float(winner.success), winner.candidate_name))

    return envelope, summary_rows


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
    show_optimal_search: bool,
    search_n2_values: list[int] | None,
    search_k_box_values: list[int] | None,
    search_families: list[str],
    optimal_search_summary_output: Path | None,
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

    if show_optimal_search:
        envelope, summary_rows = _compute_optimal_search_envelope(
            plot_n2=n2,
            p_values=p_values,
            search_n2_values=search_n2_values,
            search_k_box_values=search_k_box_values,
            search_families=search_families,
        )
        ax.plot(
            p_values,
            envelope,
            linestyle="-",
            color="black",
            linewidth=2.2,
            label="optimal analytical search",
        )

        print(f"  Optimal analytical search summary (n2={n2}):")
        print("    p_rule | best_success | best_candidate_name")
        for p_rule, best_success, best_name in summary_rows:
            print(f"    {p_rule:.2f} | {best_success:.12g} | {best_name}")

        if optimal_search_summary_output is not None:
            summary_path = optimal_search_summary_output
            if summary_path.suffix == "":
                summary_path = summary_path / f"optimal_search_summary_n2_{n2}.csv"
            elif summary_path.name == ".":
                summary_path = summary_path / f"optimal_search_summary_n2_{n2}.csv"

            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["p_rule", "best_success", "best_candidate_name"])
                for p_rule, best_success, best_name in summary_rows:
                    writer.writerow([p_rule, best_success, best_name])
            print(f"  Saved optimal-search summary: {summary_path}")

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
    parser.add_argument(
        "--n2-values",
        nargs="+",
        type=int,
        default=[4, 8, 16],
        help="n2 values to plot. Default: 4 8 16",
    )
    parser.add_argument(
        "--show-optimal-search",
        action="store_true",
        default=False,
        help="Overlay optimal analytical search envelope on each figure.",
    )
    parser.add_argument(
        "--search-n2-values",
        nargs="+",
        type=int,
        default=None,
        help="Optional search n2 values. Default: same n2 as current plot.",
    )
    parser.add_argument(
        "--search-k-box-values",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional search k_box values. Defaults by plot n2: "
            "n2=4 -> 1 3; n2=8 -> 1 3 7; n2=16 -> 1 3 7 15."
        ),
    )
    parser.add_argument(
        "--search-families",
        nargs="+",
        type=str,
        default=["majority", "pyramid", "horizontal", "vertical"],
        choices=["majority", "pyramid", "horizontal", "vertical"],
        help="Families to include in optimal analytical search.",
    )
    parser.add_argument(
        "--optimal-search-summary-output",
        type=Path,
        default=None,
        help="Optional CSV path (or directory) for p_rule,best_success,best_candidate_name summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir: Path = args.output_dir

    n2_values = [int(v) for v in args.n2_values]
    for n2 in n2_values:
        print(f"Generating figure for n2={n2} ...")
        output_path = output_dir / f"analytics_curves_n2_{n2}.png"
        generate_figure(
            n2,
            P_VALUES,
            output_path,
            dpi=args.dpi,
            show=args.show,
            show_optimal_search=args.show_optimal_search,
            search_n2_values=args.search_n2_values,
            search_k_box_values=args.search_k_box_values,
            search_families=[str(v) for v in args.search_families],
            optimal_search_summary_output=args.optimal_search_summary_output,
        )

    print("Done.")


if __name__ == "__main__":
    main()
