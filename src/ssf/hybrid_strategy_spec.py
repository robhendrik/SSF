"""Canonical strategy schema, validation rules, and naming for SSF A-side specs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional


Family = Literal["majority", "pyramid", "horizontal", "vertical"]
Order = Literal["maj_pyr", "pyr_maj"]


@dataclass(frozen=True)
class HybridStrategySpec:
    family: Family
    n2: int
    k_box: int
    split: Optional[int] = None
    depth_s: Optional[int] = None
    order: Optional[Order] = None
    tie_bandwidth: Optional[int] = None
    tie_value: int = 0


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _check_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int.")
    return value


def _validate_tie_bandwidth(tie_bandwidth: int, majority_len: int) -> None:
    if not (0 <= tie_bandwidth < majority_len // 2):
        raise ValueError("tie_bandwidth out of range for majority domain.")
    if tie_bandwidth % 2 != 0:
        raise ValueError("tie_bandwidth must be even for even-length majority domains.")


def _required_boxes_validated(spec: HybridStrategySpec) -> int:
    if spec.family == "majority":
        return 0
    if spec.family == "pyramid":
        return spec.n2 - 1
    if spec.family == "horizontal":
        assert spec.split is not None
        if spec.order == "maj_pyr":
            pyramid_len = spec.n2 - spec.split
        else:
            pyramid_len = spec.split
        return pyramid_len - 1
    if spec.family == "vertical":
        assert spec.depth_s is not None
        if spec.order == "maj_pyr":
            block_count = 2 ** spec.depth_s
            return block_count - 1
        apex_len = spec.n2 // (2 ** spec.depth_s)
        return spec.n2 - apex_len
    raise ValueError(f"Unknown family: {spec.family}")


def validate_spec(spec: HybridStrategySpec) -> HybridStrategySpec:
    if spec.family not in ("majority", "pyramid", "horizontal", "vertical"):
        raise ValueError(f"Invalid family: {spec.family}")

    n2 = _check_int("n2", spec.n2)
    k_box = _check_int("k_box", spec.k_box)
    tie_value = _check_int("tie_value", spec.tie_value)

    if n2 <= 0:
        raise ValueError("n2 must be positive.")
    if k_box < 0:
        raise ValueError("k_box must be non-negative.")
    if tie_value not in (0, 1):
        raise ValueError("tie_value must be 0 or 1.")

    majority_len: Optional[int] = None
    tie_bandwidth: Optional[int] = None

    if spec.family == "majority":
        if n2 % 2 != 0:
            raise ValueError("Majority requires even n2.")
        if spec.tie_bandwidth is None:
            raise ValueError("Majority requires tie_bandwidth.")
        tie_bandwidth = _check_int("tie_bandwidth", spec.tie_bandwidth)
        majority_len = n2
        if spec.split is not None or spec.depth_s is not None or spec.order is not None:
            raise ValueError("Majority does not allow split, depth_s, or order.")

    elif spec.family == "pyramid":
        if not is_power_of_two(n2):
            raise ValueError("Pyramid requires n2 to be a power of two.")
        if spec.tie_bandwidth is not None or spec.split is not None or spec.depth_s is not None or spec.order is not None:
            raise ValueError("Pyramid does not allow tie_bandwidth, split, depth_s, or order.")

    elif spec.family == "horizontal":
        if spec.order not in ("maj_pyr", "pyr_maj"):
            raise ValueError("Horizontal requires order in {'maj_pyr', 'pyr_maj'}.")
        if spec.split is None:
            raise ValueError("Horizontal requires split.")
        split = _check_int("split", spec.split)
        if not (0 < split < n2):
            raise ValueError("Horizontal split must satisfy 0 < split < n2.")
        if spec.tie_bandwidth is None:
            raise ValueError("Horizontal requires tie_bandwidth.")
        tie_bandwidth = _check_int("tie_bandwidth", spec.tie_bandwidth)

        if spec.order == "maj_pyr":
            majority_len = split
            pyramid_len = n2 - split
        else:
            pyramid_len = split
            majority_len = n2 - split

        if majority_len % 2 != 0:
            raise ValueError("Horizontal majority side must have even length.")
        if not is_power_of_two(pyramid_len):
            raise ValueError("Horizontal pyramid side length must be a power of two.")
        if spec.depth_s is not None:
            raise ValueError("Horizontal does not allow depth_s.")

    elif spec.family == "vertical":
        if spec.order not in ("maj_pyr", "pyr_maj"):
            raise ValueError("Vertical requires order in {'maj_pyr', 'pyr_maj'}.")
        if spec.depth_s is None:
            raise ValueError("Vertical requires depth_s.")
        depth_s = _check_int("depth_s", spec.depth_s)
        if not is_power_of_two(n2):
            raise ValueError("Vertical requires n2 to be a power of two.")
        if spec.tie_bandwidth is None:
            raise ValueError("Vertical requires tie_bandwidth.")
        tie_bandwidth = _check_int("tie_bandwidth", spec.tie_bandwidth)
        if spec.split is not None:
            raise ValueError("Vertical does not allow split.")

        max_depth = n2.bit_length() - 1
        if not (0 < depth_s < max_depth):
            raise ValueError("Vertical requires nontrivial depth_s with 0 < depth_s < log2(n2).")

        if spec.order == "maj_pyr":
            block_count = 2 ** depth_s
            block_size = n2 // block_count
            if block_size % 2 != 0:
                raise ValueError("Vertical maj_pyr requires even block_size.")
            majority_len = block_size
        else:
            apex_len = n2 // (2 ** depth_s)
            if apex_len % 2 != 0:
                raise ValueError("Vertical pyr_maj requires even apex_len.")
            majority_len = apex_len

    if majority_len is not None:
        assert tie_bandwidth is not None
        _validate_tie_bandwidth(tie_bandwidth, majority_len)

    req = _required_boxes_validated(spec)
    if req > k_box:
        raise ValueError(f"k_box too small: requires at least {req} boxes, got {k_box}.")

    return spec


def even_tie_values(majority_len: int) -> list[int]:
    """Return all valid even tie_bandwidth values for a given even-length majority domain."""
    if not isinstance(majority_len, int) or isinstance(majority_len, bool):
        raise ValueError("majority_len must be a positive even int.")
    if majority_len <= 0 or majority_len % 2 != 0:
        raise ValueError("majority_len must be a positive even int.")
    return list(range(0, majority_len // 2, 2))


def required_boxes(spec: HybridStrategySpec) -> int:
    validated = validate_spec(spec)
    return _required_boxes_validated(validated)


def candidate_name(spec: HybridStrategySpec) -> str:
    validated = validate_spec(spec)
    if validated.family == "majority":
        assert validated.tie_bandwidth is not None
        return f"majority_n2_{validated.n2}_k_{validated.k_box}_b_{validated.tie_bandwidth}"
    if validated.family == "pyramid":
        return f"pyramid_n2_{validated.n2}_k_{validated.k_box}"
    if validated.family == "horizontal":
        assert validated.split is not None
        assert validated.order is not None
        left_len = validated.split
        right_len = validated.n2 - validated.split
        assert validated.tie_bandwidth is not None
        order_tag = "maj_pyr" if validated.order == "maj_pyr" else "pyr_maj"
        return (
            f"hybrid_horiz_{order_tag}_n2_{validated.n2}_k_{validated.k_box}"
            f"_split_{left_len}_{right_len}_b_{validated.tie_bandwidth}"
        )
    assert validated.depth_s is not None
    assert validated.order is not None
    assert validated.tie_bandwidth is not None
    return (
        f"hybrid_vert_{validated.order}_n2_{validated.n2}_k_{validated.k_box}"
        f"_s_{validated.depth_s}_b_{validated.tie_bandwidth}"
    )


def stable_key(spec: HybridStrategySpec) -> tuple:
    validated = validate_spec(spec)
    return (
        validated.family,
        validated.n2,
        validated.k_box,
        validated.split,
        validated.depth_s,
        validated.order,
        validated.tie_bandwidth,
        validated.tie_value,
    )


def with_required_boxes(spec: HybridStrategySpec) -> dict:
    validated = validate_spec(spec)
    req = _required_boxes_validated(validated)
    payload = asdict(validated)
    payload["required_boxes"] = req
    payload["unused_boxes"] = validated.k_box - req
    return payload
