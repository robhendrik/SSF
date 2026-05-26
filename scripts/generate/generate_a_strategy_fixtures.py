"""Generate canonical dense A-side fixtures and manifest from HybridStrategySpec.

Architectural role:
- authoritative fixture producer for fixtures/persistent/a_strategies
- bridges canonical spec validation to dense fixture serialization and analytics

Invariants:
- naming derives from candidate_name(spec)
- only validated legal specs are emitted

Failure behavior:
- invalid specs are recorded in skipped output; hard failures raise ValueError
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PERSISTENT_FIXTURE_DIR = PROJECT_ROOT / "fixtures" / "persistent" / "a_strategies"
WORK_FIXTURE_BANK_DIR = PROJECT_ROOT / "work" / "fixture_bank"
WORK_FIXTURE_DIR = WORK_FIXTURE_BANK_DIR / "a_strategies"
WORK_EVALUATIONS_DIR = WORK_FIXTURE_BANK_DIR / "evaluations"

from ssf.hybrid_strategy_analytics import analytical_win_rate
from ssf.hybrid_strategy_fixtures import write_strategy_fixture
from ssf.hybrid_strategy_spec import (
    HybridStrategySpec,
    candidate_name,
    even_tie_values,
    required_boxes,
    stable_key,
    validate_spec,
    with_required_boxes,
)

DEFAULT_OUTPUT_DIR = PERSISTENT_FIXTURE_DIR
DEFAULT_N2_VALUES = (4, 8)
DEFAULT_K_BOX_VALUES = (1, 3, 7)
DEFAULT_P_RULE_VALUES = (0.50, 0.65, 0.77, 0.80, 0.90, 1.00)
ALL_FAMILIES = ("majority", "pyramid", "horizontal", "vertical")


def _clear_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.npz", "manifest.json", "skipped.json"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _normalize_families(raw_families: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(raw_families)
    if not requested or "all" in requested:
        return ALL_FAMILIES
    invalid = sorted(set(requested) - set(ALL_FAMILIES))
    if invalid:
        raise ValueError(f"Unknown family selection: {invalid}")
    return tuple(dict.fromkeys(requested))


def _dense_skip_reason(spec: HybridStrategySpec, max_n2_for_dense_fixtures: int, max_comm_cells: int) -> str | None:
    comm_cells = (1 << spec.n2) * (1 << spec.k_box)
    if spec.n2 > max_n2_for_dense_fixtures:
        return f"n2={spec.n2} exceeds dense threshold {max_n2_for_dense_fixtures}."
    if comm_cells > max_comm_cells:
        return f"comm_table size {comm_cells} exceeds max_comm_cells {max_comm_cells}."
    return None


def _format_p_rule_key(p_rule: float) -> str:
    return f"{p_rule:.2f}"


def _legacy_default_specs() -> list[HybridStrategySpec]:
    return [
        HybridStrategySpec("majority", n2=4, k_box=3, tie_bandwidth=0),
        HybridStrategySpec("majority", n2=8, k_box=3, tie_bandwidth=0),
        HybridStrategySpec("majority", n2=8, k_box=1, tie_bandwidth=0),
        HybridStrategySpec("pyramid", n2=4, k_box=3),
        HybridStrategySpec("pyramid", n2=8, k_box=7),
        HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="maj_pyr", tie_bandwidth=0),
        HybridStrategySpec("horizontal", n2=4, k_box=3, split=2, order="pyr_maj", tie_bandwidth=0),
        HybridStrategySpec("vertical", n2=4, k_box=3, depth_s=1, order="maj_pyr", tie_bandwidth=0),
        HybridStrategySpec("vertical", n2=4, k_box=3, depth_s=1, order="pyr_maj", tie_bandwidth=0),
    ]


def _representative_specs(
    n2_values: Iterable[int],
    k_box_values: Iterable[int],
    families: tuple[str, ...],
) -> list[HybridStrategySpec]:
    specs: list[HybridStrategySpec] = []
    for n2 in sorted({int(v) for v in n2_values}):
        for k_box in sorted({int(v) for v in k_box_values}):
            if "majority" in families:
                specs.append(HybridStrategySpec("majority", n2=n2, k_box=k_box, tie_bandwidth=0))
            if "pyramid" in families:
                specs.append(HybridStrategySpec("pyramid", n2=n2, k_box=k_box))
            if "horizontal" in families and n2 > 1:
                split = n2 // 2
                specs.append(
                    HybridStrategySpec(
                        "horizontal",
                        n2=n2,
                        k_box=k_box,
                        split=split,
                        order="maj_pyr",
                        tie_bandwidth=0,
                    )
                )
                specs.append(
                    HybridStrategySpec(
                        "horizontal",
                        n2=n2,
                        k_box=k_box,
                        split=split,
                        order="pyr_maj",
                        tie_bandwidth=0,
                    )
                )
            if "vertical" in families and n2 > 2:
                specs.append(
                    HybridStrategySpec(
                        "vertical",
                        n2=n2,
                        k_box=k_box,
                        depth_s=1,
                        order="maj_pyr",
                        tie_bandwidth=0,
                    )
                )
                specs.append(
                    HybridStrategySpec(
                        "vertical",
                        n2=n2,
                        k_box=k_box,
                        depth_s=1,
                        order="pyr_maj",
                        tie_bandwidth=0,
                    )
                )
    return specs


def _iter_all_legal_specs(
    n2_values: Iterable[int],
    k_box_values: Iterable[int],
    families: tuple[str, ...],
) -> list[HybridStrategySpec]:
    specs: list[HybridStrategySpec] = []
    for n2 in sorted({int(v) for v in n2_values}):
        for k_box in sorted({int(v) for v in k_box_values}):
            if "majority" in families:
                try:
                    tie_values = even_tie_values(n2)
                except ValueError:
                    tie_values = []
                for tie_bandwidth in tie_values:
                    specs.append(HybridStrategySpec("majority", n2=n2, k_box=k_box, tie_bandwidth=tie_bandwidth))

            if "pyramid" in families:
                specs.append(HybridStrategySpec("pyramid", n2=n2, k_box=k_box))

            if "horizontal" in families:
                for split in range(1, n2):
                    for order in ("maj_pyr", "pyr_maj"):
                        majority_len = split if order == "maj_pyr" else n2 - split
                        try:
                            tie_values = even_tie_values(majority_len)
                        except ValueError:
                            continue
                        for tie_bandwidth in tie_values:
                            specs.append(
                                HybridStrategySpec(
                                    "horizontal",
                                    n2=n2,
                                    k_box=k_box,
                                    split=split,
                                    order=order,
                                    tie_bandwidth=tie_bandwidth,
                                )
                            )

            if "vertical" in families and n2 > 0:
                max_depth = int(math.log2(n2)) if n2 > 0 and (n2 & (n2 - 1)) == 0 else 0
                for depth_s in range(1, max_depth):
                    block_count = 2 ** depth_s
                    block_size = n2 // block_count
                    apex_len = n2 // block_count

                    try:
                        maj_pyr_ties = even_tie_values(block_size)
                    except ValueError:
                        maj_pyr_ties = []
                    for tie_bandwidth in maj_pyr_ties:
                        specs.append(
                            HybridStrategySpec(
                                "vertical",
                                n2=n2,
                                k_box=k_box,
                                depth_s=depth_s,
                                order="maj_pyr",
                                tie_bandwidth=tie_bandwidth,
                            )
                        )

                    try:
                        pyr_maj_ties = even_tie_values(apex_len)
                    except ValueError:
                        pyr_maj_ties = []
                    for tie_bandwidth in pyr_maj_ties:
                        specs.append(
                            HybridStrategySpec(
                                "vertical",
                                n2=n2,
                                k_box=k_box,
                                depth_s=depth_s,
                                order="pyr_maj",
                                tie_bandwidth=tie_bandwidth,
                            )
                        )
    return specs


def _candidate_specs(
    *,
    n2_values: Iterable[int],
    k_box_values: Iterable[int],
    families: tuple[str, ...],
    include_all_legal: bool,
    legacy_defaults: bool,
) -> list[HybridStrategySpec]:
    if legacy_defaults:
        base = _legacy_default_specs()
        allowed_n2 = {int(v) for v in n2_values}
        allowed_k = {int(v) for v in k_box_values}
        return [s for s in base if s.family in families and s.n2 in allowed_n2 and s.k_box in allowed_k]
    if include_all_legal:
        return _iter_all_legal_specs(n2_values, k_box_values, families)
    return _representative_specs(n2_values, k_box_values, families)


def generate_a_strategy_fixtures(
    *,
    output_dir: Path,
    overwrite: bool = False,
    n2_values: Iterable[int] = DEFAULT_N2_VALUES,
    k_box_values: Iterable[int] = DEFAULT_K_BOX_VALUES,
    families: Iterable[str] = ("all",),
    include_all_legal: bool = False,
    legacy_defaults: bool = False,
    p_rule_values: Iterable[float] = DEFAULT_P_RULE_VALUES,
    max_n2_for_dense_fixtures: int = 8,
    max_comm_cells: int = 16_000_000,
) -> list[dict]:
    """Generate canonical fixture files and return manifest records for emitted specs."""
    family_selection = _normalize_families(families)
    p_rules = tuple(float(v) for v in p_rule_values)

    if overwrite:
        _clear_output_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    skipped: list[dict] = []
    seen_keys: set[tuple] = set()

    for candidate in _candidate_specs(
        n2_values=n2_values,
        k_box_values=k_box_values,
        families=family_selection,
        include_all_legal=include_all_legal,
        legacy_defaults=legacy_defaults,
    ):
        try:
            spec = validate_spec(candidate)
        except ValueError as exc:
            skipped.append({"spec": asdict(candidate), "reason": str(exc)})
            continue

        key = stable_key(spec)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        skip_reason = _dense_skip_reason(spec, max_n2_for_dense_fixtures, max_comm_cells)
        if skip_reason is not None:
            skipped.append(
                {
                    "spec": asdict(spec),
                    "name": candidate_name(spec),
                    "reason": skip_reason,
                }
            )
            continue

        fixture_path = write_strategy_fixture(spec, output_dir)
        analytics = [analytical_win_rate(spec, p_rule) for p_rule in p_rules]
        first = analytics[0] if analytics else None
        payload = with_required_boxes(spec)
        manifest.append(
            {
                "name": candidate_name(spec),
                "family": spec.family,
                "n2": spec.n2,
                "k_box": spec.k_box,
                "required_boxes": required_boxes(spec),
                "unused_boxes": payload["unused_boxes"],
                "split": spec.split,
                "depth_s": spec.depth_s,
                "order": spec.order,
                "tie_bandwidth": spec.tie_bandwidth,
                "tie_value": spec.tie_value,
                "file": fixture_path.name,
                "analytical_success": {
                    _format_p_rule_key(p_rule): float(result.success)
                    for p_rule, result in zip(p_rules, analytics)
                },
                "analytical_method": first.method if first is not None else "closed_form",
                "analytical_exact": first.exact if first is not None else True,
                "analytical_notes": first.notes if first is not None else "",
            }
        )

    if not manifest and not skipped:
        raise ValueError("No candidate fixtures were generated.")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if skipped:
        (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    return manifest


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for canonical fixture generation workflow."""
    parser = argparse.ArgumentParser(description="Generate canonical A-strategy fixtures from HybridStrategySpec.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n2-values", nargs="+", type=int, default=list(DEFAULT_N2_VALUES))
    parser.add_argument("--k-box-values", nargs="+", type=int, default=list(DEFAULT_K_BOX_VALUES))
    parser.add_argument(
        "--families",
        nargs="+",
        default=["all"],
        choices=["majority", "pyramid", "horizontal", "vertical", "all"],
    )
    parser.add_argument("--include-all-legal", action="store_true")
    parser.add_argument("--legacy-defaults", action="store_true")
    parser.add_argument("--p-rule-values", nargs="+", type=float, default=list(DEFAULT_P_RULE_VALUES))
    parser.add_argument("--max-n2-for-dense-fixtures", type=int, default=8)
    parser.add_argument("--max-comm-cells", type=int, default=16_000_000)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for canonical fixture generation."""
    args = _parse_args()
    manifest = generate_a_strategy_fixtures(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        n2_values=args.n2_values,
        k_box_values=args.k_box_values,
        families=args.families,
        include_all_legal=args.include_all_legal,
        legacy_defaults=args.legacy_defaults,
        p_rule_values=args.p_rule_values,
        max_n2_for_dense_fixtures=args.max_n2_for_dense_fixtures,
        max_comm_cells=args.max_comm_cells,
    )
    print(f"Generated {len(manifest)} fixtures in {args.output_dir}")


if __name__ == "__main__":
    main()
