from __future__ import annotations

from dataclasses import dataclass

from .hybrid_strategy_spec import HybridStrategySpec, required_boxes, validate_spec


def _bit(value: int, idx: int) -> int:
    return (value >> idx) & 1


def _out_bit(value: int, idx: int) -> int:
    return (value >> idx) & 1


def _majority_or_tie(bits: list[int], tie_bandwidth: int, tie_value: int) -> int:
    n = len(bits)
    s = sum(bits)
    d = abs(2 * s - n)
    if d <= tie_bandwidth:
        return tie_value
    return 1 if s > n / 2 else 0


def _measure_pyramid_append(values: list[int], box_idx: int, out_a_prefix: int) -> int:
    units = values[:]
    for d in range(box_idx + 1):
        left = units[2 * d]
        right = units[2 * d + 1]
        a_d = left ^ right
        if d == box_idx:
            return a_d
        units.append(left ^ _out_bit(out_a_prefix, d))
    raise ValueError("Invalid box index for pyramid measurement.")


def _run_pyramid_append(values: list[int], out_a: int, box_count: int) -> int:
    units = values[:]
    for d in range(box_count):
        left = units[2 * d]
        right = units[2 * d + 1]
        units.append(left ^ _out_bit(out_a, d))
    return units[-1]


def _measure_pyramid_layers(values: list[int], target_box: int, out_a_prefix: int, max_layers: int) -> int:
    current = values[:]
    next_box = 0
    for _ in range(max_layers):
        nxt: list[int] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1]
            a_d = left ^ right
            if next_box == target_box:
                return a_d
            nxt.append(left ^ _out_bit(out_a_prefix, next_box))
            next_box += 1
        current = nxt
    raise ValueError("Invalid box index for layered pyramid measurement.")


def _run_pyramid_layers(values: list[int], out_a: int, max_layers: int) -> list[int]:
    current = values[:]
    next_box = 0
    for _ in range(max_layers):
        nxt: list[int] = []
        for i in range(0, len(current), 2):
            left = current[i]
            nxt.append(left ^ _out_bit(out_a, next_box))
            next_box += 1
        current = nxt
    return current


@dataclass(frozen=True)
class HybridProceduralStrategy:
    spec: HybridStrategySpec
    n2: int
    k_box: int
    boxes_used: int

    def _validate_box_idx(self, box_idx: int) -> None:
        if box_idx < 0 or box_idx >= self.k_box:
            raise ValueError(f"box_idx out of range: {box_idx}")

    def _validate_field(self, field: int) -> None:
        if field < 0 or field >= (1 << self.n2):
            raise ValueError(f"field out of range: {field}")

    def _validate_prefix(self, box_idx: int, out_a_prefix: int) -> None:
        if out_a_prefix < 0 or out_a_prefix >= (1 << box_idx):
            raise ValueError(
                f"out_a_prefix out of range for box_idx={box_idx}: {out_a_prefix}"
            )

    def _validate_out_a(self, out_a: int) -> None:
        if out_a < 0 or out_a >= (1 << self.k_box):
            raise ValueError(f"out_a out of range: {out_a}")

    def _field_bits(self, field: int) -> list[int]:
        return [_bit(field, i) for i in range(self.n2)]

    def measure_a(self, box_idx: int, field: int, out_a_prefix: int) -> int:
        self._validate_box_idx(box_idx)
        self._validate_field(field)
        self._validate_prefix(box_idx, out_a_prefix)

        if box_idx >= self.boxes_used:
            return 0

        spec = self.spec
        field_bits = self._field_bits(field)

        if spec.family == "majority":
            return 0

        if spec.family == "pyramid":
            return _measure_pyramid_append(field_bits, box_idx, out_a_prefix)

        if spec.family == "horizontal":
            assert spec.split is not None
            assert spec.order is not None
            if spec.order == "maj_pyr":
                pyramid_idx = list(range(spec.split, self.n2))
            else:
                pyramid_idx = list(range(spec.split))

            pyr_bits = [field_bits[i] for i in pyramid_idx]
            return _measure_pyramid_append(pyr_bits, box_idx, out_a_prefix)

        if spec.family == "vertical":
            assert spec.depth_s is not None
            assert spec.order is not None
            assert spec.tie_bandwidth is not None

            if spec.order == "maj_pyr":
                block_count = 2 ** spec.depth_s
                block_size = self.n2 // block_count
                block_outs = []
                for b in range(block_count):
                    lo = b * block_size
                    hi = (b + 1) * block_size
                    block = field_bits[lo:hi]
                    block_outs.append(
                        _majority_or_tie(block, spec.tie_bandwidth, spec.tie_value)
                    )
                return _measure_pyramid_append(block_outs, box_idx, out_a_prefix)

            return _measure_pyramid_layers(field_bits, box_idx, out_a_prefix, spec.depth_s)

        raise ValueError(f"Unknown family: {spec.family}")

    def comm(self, field: int, out_a: int) -> int:
        self._validate_field(field)
        self._validate_out_a(out_a)

        spec = self.spec
        field_bits = self._field_bits(field)

        if spec.family == "majority":
            assert spec.tie_bandwidth is not None
            return _majority_or_tie(field_bits, spec.tie_bandwidth, spec.tie_value)

        if spec.family == "pyramid":
            return _run_pyramid_append(field_bits, out_a, self.n2 - 1)

        if spec.family == "horizontal":
            assert spec.split is not None
            assert spec.order is not None
            assert spec.tie_bandwidth is not None

            if spec.order == "maj_pyr":
                majority_idx = list(range(spec.split))
                pyramid_idx = list(range(spec.split, self.n2))
            else:
                pyramid_idx = list(range(spec.split))
                majority_idx = list(range(spec.split, self.n2))

            maj_bits = [field_bits[i] for i in majority_idx]
            pyr_bits = [field_bits[i] for i in pyramid_idx]
            s = sum(maj_bits)
            n_maj = len(maj_bits)
            is_tie = abs(2 * s - n_maj) <= spec.tie_bandwidth
            maj = 1 if s > n_maj / 2 else 0

            pyramid_len = len(pyramid_idx)
            if pyramid_len > 1:
                pyramid_apex = _run_pyramid_append(pyr_bits, out_a, pyramid_len - 1)
            else:
                pyramid_apex = pyr_bits[0] if pyramid_len == 1 else 0

            return maj if not is_tie else pyramid_apex

        if spec.family == "vertical":
            assert spec.depth_s is not None
            assert spec.order is not None
            assert spec.tie_bandwidth is not None

            if spec.order == "maj_pyr":
                block_count = 2 ** spec.depth_s
                block_size = self.n2 // block_count
                block_outs = []
                for b in range(block_count):
                    lo = b * block_size
                    hi = (b + 1) * block_size
                    block = field_bits[lo:hi]
                    block_outs.append(
                        _majority_or_tie(block, spec.tie_bandwidth, spec.tie_value)
                    )
                return _run_pyramid_append(block_outs, out_a, block_count - 1)

            apex_values = _run_pyramid_layers(field_bits, out_a, spec.depth_s)
            return _majority_or_tie(apex_values, spec.tie_bandwidth, spec.tie_value)

        raise ValueError(f"Unknown family: {spec.family}")


def build_procedural_strategy(spec: HybridStrategySpec) -> HybridProceduralStrategy:
    validated = validate_spec(spec)
    boxes_used = required_boxes(validated)
    return HybridProceduralStrategy(
        spec=validated,
        n2=validated.n2,
        k_box=validated.k_box,
        boxes_used=boxes_used,
    )
