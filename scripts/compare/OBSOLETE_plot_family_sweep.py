from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SSF_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = SSF_DIR / "results" / "plots"


def _normalize_box_limit_token(value: str | None) -> str:
    if value is None:
        return "none"
    token = str(value).strip().lower()
    if token in {"", "none", "all"}:
        return "none"
    try:
        return str(int(float(token)))
    except ValueError:
        return token


def _normalize_row(row: Dict[str, str]) -> Dict[str, Any]:
    strategy_name = (row.get("strategy_name") or row.get("candidate_name") or "").strip()
    if not strategy_name:
        raise ValueError("Missing strategy_name/candidate_name in input row.")

    p_rule_raw = row.get("p_rule")
    success_raw = row.get("success")
    if p_rule_raw is None:
        raise ValueError("Missing p_rule in input row.")
    if success_raw is None:
        raise ValueError("Missing success in input row.")

    family = (row.get("family") or "").strip() or "unknown"

    def _int_or_none(v: str | None) -> int | None:
        if v is None:
            return None
        token = str(v).strip()
        if token == "":
            return None
        try:
            return int(float(token))
        except ValueError:
            return None

    def _float_or_none(v: str | None) -> float | None:
        if v is None:
            return None
        token = str(v).strip()
        if token == "":
            return None
        try:
            return float(token)
        except ValueError:
            return None

    return {
        "strategy_name": strategy_name,
        "family": family,
        "p_rule": float(p_rule_raw),
        "success": float(success_raw),
        "subset_policy": (row.get("subset_policy") or "").strip(),
        "box_limit": _normalize_box_limit_token(row.get("box_limit")),
        "tie_bandwidth": _int_or_none(row.get("tie_bandwidth")),
        "depth_s": _int_or_none(row.get("depth_s")),
        "order": (row.get("order") or "").strip() or None,
        "best_for_condition": str(row.get("best_for_condition", "")).strip().lower() == "true",
        "split": (row.get("split") or "").strip() or None,
        "majority_side": (row.get("majority_side") or "").strip() or None,
        "pyramid_side": (row.get("pyramid_side") or "").strip() or None,
        "boxes_used": _int_or_none(row.get("boxes_used")),
        "majority_len": _float_or_none(row.get("majority_len")),
    }


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")
        for row in reader:
            rows.append(_normalize_row(row))

    if not rows:
        raise ValueError("Input CSV has no data rows.")
    return rows


def apply_filters(
    rows: Iterable[Dict[str, Any]],
    subset_policy: str | None,
    box_limit: str | None,
) -> List[Dict[str, Any]]:
    filtered = list(rows)

    if subset_policy is not None:
        wanted = subset_policy.strip()
        filtered = [row for row in filtered if str(row.get("subset_policy", "")) == wanted]

    if box_limit is not None:
        wanted_box = _normalize_box_limit_token(box_limit)
        filtered = [row for row in filtered if str(row.get("box_limit", "none")) == wanted_box]

    if not filtered:
        raise ValueError("No rows remain after applying filters.")

    return filtered


def build_family_sweep(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, float], Dict[str, Any]] = {}

    for row in rows:
        key = (str(row["family"]), float(row["p_rule"]))
        if key not in best or float(row["success"]) > float(best[key]["success"]):
            best[key] = row

    output: List[Dict[str, Any]] = []
    for family, p_rule in sorted(best.keys(), key=lambda t: (t[0], t[1])):
        row = best[(family, p_rule)]
        output.append(
            {
                "family": family,
                "p_rule": float(p_rule),
                "best_success": float(row["success"]),
                "best_strategy_name": str(row["strategy_name"]),
            }
        )

    return output


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    columns = ["family", "p_rule", "best_success", "best_strategy_name"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    columns = ["family", "p_rule", "best_success", "best_strategy_name"]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["family"]),
                    f"{float(row['p_rule']):.12g}",
                    f"{float(row['best_success']):.12g}",
                    str(row["best_strategy_name"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _write_markdown(path: Path, sweep_rows: List[Dict[str, Any]]) -> None:
    content = [
        "# Family Sweep Summary",
        "",
        _markdown_table(sweep_rows),
        "",
        "Plot shows best row available in the provided CSV; it is not a global optimum claim.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


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
            success_given_weight = 0.5
        total += prob_weight * success_given_weight
    return float(total)


def _pyramid_n2_4_success_probability(p_rule: float) -> float:
    return float((1.0 + (2.0 * p_rule - 1.0) ** 2) / 2.0)


def _display_family(row: Dict[str, Any]) -> str:
    family = str(row.get("family", ""))
    if family == "hybrid_horizontal":
        return "horizontal hybrids"
    if family == "hybrid_vertical_maj_pyr":
        return "vertical maj/pyr"
    if family == "hybrid_vertical_pyr_maj":
        return "vertical pyr/maj"
    return family


def _build_series(rows: List[Dict[str, Any]], include_baselines: bool) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for row in rows:
        fam = _display_family(row)
        key = (fam, float(row["p_rule"]))
        if key not in by_key or float(row["success"]) > float(by_key[key]["success"]):
            by_key[key] = row

    series: List[Dict[str, Any]] = []
    p_rules = sorted({float(r["p_rule"]) for r in rows})
    for fam, p in sorted(by_key.keys(), key=lambda t: (t[0], t[1])):
        row = by_key[(fam, p)]
        series.append(
            {
                "family": fam,
                "p_rule": p,
                "best_success": float(row["success"]),
                "best_strategy_name": str(row["strategy_name"]),
            }
        )

    if include_baselines:
        majority = _majority_success_probability(4)
        for p in p_rules:
            series.append(
                {
                    "family": "pure majority",
                    "p_rule": p,
                    "best_success": majority,
                    "best_strategy_name": "majority_n2_4_k_3 (analytical)",
                }
            )
            series.append(
                {
                    "family": "pure pyramid",
                    "p_rule": p,
                    "best_success": _pyramid_n2_4_success_probability(p),
                    "best_strategy_name": "pyramid_n2_4_k_3 (analytical)",
                }
            )

    return sorted(series, key=lambda r: (str(r["family"]), float(r["p_rule"])))


def _best_family_by_p(series_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[float, Dict[str, Any]] = {}
    for row in series_rows:
        p = float(row["p_rule"])
        if p not in best or float(row["best_success"]) > float(best[p]["best_success"]):
            best[p] = row
    out = []
    for p in sorted(best.keys()):
        out.append(
            {
                "p_rule": p,
                "best_family": str(best[p]["family"]),
                "best_success": float(best[p]["best_success"]),
                "best_strategy_name": str(best[p]["best_strategy_name"]),
            }
        )
    return out


def _plot(path: Path, sweep_rows: List[Dict[str, Any]], x_axis: str, y_axis: str, title: str | None) -> None:
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for row in sweep_rows:
        by_family.setdefault(str(row["family"]), []).append(row)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for family in sorted(by_family):
        points = sorted(by_family[family], key=lambda item: float(item["p_rule"]))
        xs = [float(item["p_rule"]) for item in points]
        ys = [float(item["best_success"]) for item in points]
        ax.plot(xs, ys, marker="o", label=family)

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(title or "Family Sweep: Best success vs p_rule")
    ax.grid(True, alpha=0.3)
    if by_family:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_best_family_vs_p(path: Path, best_rows: List[Dict[str, Any]]) -> None:
    families = sorted({str(r["best_family"]) for r in best_rows})
    fam_to_idx = {fam: idx for idx, fam in enumerate(families)}
    xs = [float(r["p_rule"]) for r in best_rows]
    ys = [fam_to_idx[str(r["best_family"])] for r in best_rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("p_rule")
    ax.set_ylabel("best family")
    ax.set_yticks(list(range(len(families))))
    ax.set_yticklabels(families)
    ax.set_title("Best family vs p_rule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_heatmap(path: Path, matrix: List[List[float]], x_vals: List[float], y_vals: List[int], title: str, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("p_rule")
    ax.set_ylabel(y_label)
    ax.set_xticks(list(range(len(x_vals))))
    ax.set_xticklabels([f"{x:.2f}" for x in x_vals], rotation=45, ha="right")
    ax.set_yticks(list(range(len(y_vals))))
    ax.set_yticklabels([str(v) for v in y_vals])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("best success")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_best_grid(rows: List[Dict[str, Any]], y_field: str) -> Tuple[List[float], List[int], List[List[float]]]:
    p_vals = sorted({float(r["p_rule"]) for r in rows})
    y_vals = sorted({int(r[y_field]) for r in rows if r.get(y_field) is not None})
    best: Dict[Tuple[float, int], float] = {}
    for row in rows:
        if row.get(y_field) is None:
            continue
        key = (float(row["p_rule"]), int(row[y_field]))
        score = float(row["success"])
        if key not in best or score > best[key]:
            best[key] = score
    matrix: List[List[float]] = []
    for y in y_vals:
        row_vals: List[float] = []
        for p in p_vals:
            row_vals.append(float(best.get((p, y), float("nan"))))
        matrix.append(row_vals)
    return p_vals, y_vals, matrix


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot family sweep (best success per p_rule by family).")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-axis", type=str, default="p_rule", choices=["p_rule"])
    parser.add_argument("--y-axis", type=str, default="success", choices=["success"])
    parser.add_argument("--group-by", type=str, default="family", choices=["family"])
    parser.add_argument("--filter-subset-policy", type=str, default=None)
    parser.add_argument("--filter-box-limit", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {args.output_dir}. Use --overwrite to replace outputs.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in (
        "success_vs_p_rule.png",
        "best_family_vs_p_rule.png",
        "heatmap_tie_bandwidth.png",
        "heatmap_vertical_depth.png",
        "comparison_results.csv",
        "comparison_results.json",
        "comparison_summary.md",
    ):
        out_path = args.output_dir / file_name
        if args.overwrite and out_path.exists():
            out_path.unlink()

    rows = load_rows(args.input_csv)
    rows = apply_filters(rows, args.filter_subset_policy, args.filter_box_limit)
    sweep_rows = _build_series(rows, include_baselines=True)
    best_rows = _best_family_by_p(sweep_rows)

    _write_csv(args.output_dir / "comparison_results.csv", sweep_rows)
    _write_json(
        args.output_dir / "comparison_results.json",
        {
            "row_count": len(sweep_rows),
            "best_by_p_rule": best_rows,
            "rows": sweep_rows,
            "note": "Best within searched family; not a global optimum claim.",
        },
    )
    _write_markdown(args.output_dir / "comparison_summary.md", sweep_rows)
    _plot(
        args.output_dir / "success_vs_p_rule.png",
        sweep_rows,
        x_axis=args.x_axis,
        y_axis=args.y_axis,
        title=args.title or "Best success vs p_rule by family",
    )
    _plot_best_family_vs_p(args.output_dir / "best_family_vs_p_rule.png", best_rows)

    # Heatmap over tie bandwidth (all rows with tie_bandwidth)
    tie_rows = [row for row in rows if row.get("tie_bandwidth") is not None]
    if tie_rows:
        p_vals, b_vals, matrix = _build_best_grid(tie_rows, "tie_bandwidth")
        _plot_heatmap(
            args.output_dir / "heatmap_tie_bandwidth.png",
            matrix,
            p_vals,
            b_vals,
            title="Best success heatmap by tie bandwidth",
            y_label="tie bandwidth",
        )

    # Heatmap over vertical depth (only vertical rows with depth_s)
    vertical_rows = [row for row in rows if str(row.get("family", "")).startswith("hybrid_vertical") and row.get("depth_s") is not None]
    if vertical_rows:
        p_vals, s_vals, matrix = _build_best_grid(vertical_rows, "depth_s")
        _plot_heatmap(
            args.output_dir / "heatmap_vertical_depth.png",
            matrix,
            p_vals,
            s_vals,
            title="Best success heatmap by vertical depth",
            y_label="vertical depth s",
        )

    print(
        f"Loaded {len(rows)} filtered rows. Wrote comparison CSV/JSON/MD and plots to {args.output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
