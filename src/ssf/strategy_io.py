from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "persistent" / "a_strategies" / "manifest.json"


def load_manifest(manifest_path: str | Path | None = None) -> List[Dict[str, Any]]:
    path = Path(manifest_path) if manifest_path is not None else _default_manifest_path()
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, list):
        raise ValueError("Manifest must be a list of strategy records.")
    return manifest


def load_strategy(path: str | Path, manifest_path: str | Path | None = None) -> Dict[str, Any]:
    strategy_path = Path(path)
    strategy_name = None

    manifest: List[Dict[str, Any]] = []
    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
    else:
        try:
            manifest = load_manifest()
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            manifest = []

    for entry in manifest:
        if entry.get("file") == strategy_path.name:
            strategy_name = entry.get("name")
            break

    with np.load(strategy_path, allow_pickle=False) as data:
        n2 = int(data["n2"])
        k_box = int(data["k_box"])
        comm_table = data["comm_table"]

        f_tables: List[np.ndarray] = []
        for d in range(k_box):
            key = f"f_table_{d}"
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
            f_tables.append(data[key])

    return {
        "path": str(strategy_path),
        "name": strategy_name,
        "n2": n2,
        "k_box": k_box,
        "f_tables": f_tables,
        "comm": comm_table,
        "comm_table": comm_table,
    }
