"""Evaluation cache records, policy controls, replacement logic, and persistence APIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from . import strategy_evaluation
from .hybrid_strategy_spec import HybridStrategySpec, candidate_name, stable_key, validate_spec
from .strategy_evaluation import EvaluationRequest, EvaluationResult


@dataclass(frozen=True)
class CachePolicy:
    enabled: bool = False
    read: bool = True
    write: bool = True
    persist_analytical: bool = False
    prefer_cached: bool = True
    recompute_if_uncertain: bool = True
    min_trust_level: int | None = None
    min_n_eval_samples: int | None = None
    max_ci_width: float | None = None
    evaluator_version: str = "v1"
    cache_dir: Path | None = None


TRUST_LEVELS: dict[str, int] = {
    "dense_exhaustive": 4,
    "procedural_mc": 3,
    "dense_mc": 2,
    "analytical": 1,
}


REQUIRED_RECORD_FIELDS = {
    "key",
    "candidate_name",
    "spec",
    "success",
    "mode",
    "p_rule",
    "box_limit",
    "subset_policy",
    "exact",
    "analytical",
    "trust_level",
    "n_train_samples",
    "n_eval_samples",
    "confidence",
    "seed",
    "ci_low",
    "ci_high",
    "stderr",
    "evaluator_version",
    "created_at_utc",
    "metadata",
}


@dataclass(frozen=True)
class EvaluationCacheRecord:
    key: str
    candidate_name: str
    spec: dict[str, Any]
    success: float
    mode: str
    p_rule: float
    box_limit: int | None
    subset_policy: str
    exact: bool
    analytical: bool
    trust_level: int
    n_train_samples: int | None
    n_eval_samples: int | None
    confidence: float | None
    seed: int | None
    ci_low: float | None
    ci_high: float | None
    stderr: float | None
    evaluator_version: str
    created_at_utc: str
    metadata: dict[str, object] = field(default_factory=dict)


def _ci_width(record: EvaluationCacheRecord) -> float | None:
    if record.ci_low is None or record.ci_high is None:
        return None
    return float(record.ci_high) - float(record.ci_low)


def _record_from_payload(payload: Any, source: str) -> EvaluationCacheRecord:
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid cache record in {source}: expected JSON object.")

    missing = REQUIRED_RECORD_FIELDS.difference(payload.keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Invalid cache record in {source}: missing required fields: {missing_list}")

    try:
        return EvaluationCacheRecord(
            key=str(payload["key"]),
            candidate_name=str(payload["candidate_name"]),
            spec=dict(payload["spec"]),
            success=float(payload["success"]),
            mode=str(payload["mode"]),
            p_rule=float(payload["p_rule"]),
            box_limit=payload.get("box_limit"),
            subset_policy=str(payload["subset_policy"]),
            exact=bool(payload["exact"]),
            analytical=bool(payload["analytical"]),
            trust_level=int(payload["trust_level"]),
            n_train_samples=payload.get("n_train_samples"),
            n_eval_samples=payload.get("n_eval_samples"),
            confidence=payload.get("confidence"),
            seed=payload.get("seed"),
            ci_low=payload.get("ci_low"),
            ci_high=payload.get("ci_high"),
            stderr=payload.get("stderr"),
            evaluator_version=str(payload["evaluator_version"]),
            created_at_utc=str(payload["created_at_utc"]),
            metadata=dict(payload["metadata"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid cache record in {source}: {exc}") from exc


def _version_tuple(version: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", version)
    if not nums:
        return (0,)
    return tuple(int(part) for part in nums)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def _request_cache_key_tuple(request: EvaluationRequest, policy: CachePolicy) -> tuple:
    key_builder = getattr(strategy_evaluation, "evaluation_cache_key", None)
    if callable(key_builder):
        base_key = key_builder(request)
        if not isinstance(base_key, tuple):
            base_key = (base_key,)
    else:
        validated = validate_spec(request.candidate.spec)
        base_key = (
            stable_key(validated),
            request.mode,
            float(request.p_rule),
            request.box_limit,
            request.subset_policy,
            int(request.n_train_samples),
            int(request.n_eval_samples),
            float(request.confidence),
            int(request.seed),
        )

    return base_key + (str(policy.evaluator_version),)


def cache_key_to_string(key: tuple) -> str:
    return json.dumps(_to_jsonable(key), sort_keys=True, separators=(",", ":"))


def record_from_result(
    request: EvaluationRequest,
    result: EvaluationResult,
    policy: CachePolicy,
) -> EvaluationCacheRecord:
    validated: HybridStrategySpec = validate_spec(request.candidate.spec)
    mode = str(result.mode)
    analytical = mode == "analytical"
    is_mc = mode in ("dense_mc", "procedural_mc")
    key = cache_key_to_string(_request_cache_key_tuple(request, policy))

    return EvaluationCacheRecord(
        key=key,
        candidate_name=candidate_name(validated),
        spec=asdict(validated),
        success=float(result.success),
        mode=mode,
        p_rule=float(request.p_rule),
        box_limit=request.box_limit,
        subset_policy=str(request.subset_policy),
        exact=bool(result.exact),
        analytical=analytical,
        trust_level=TRUST_LEVELS.get(mode, 0),
        n_train_samples=int(request.n_train_samples) if is_mc else None,
        n_eval_samples=int(request.n_eval_samples) if is_mc else None,
        confidence=float(request.confidence) if is_mc else None,
        seed=int(request.seed) if is_mc else None,
        ci_low=float(result.ci_low) if result.ci_low is not None else None,
        ci_high=float(result.ci_high) if result.ci_high is not None else None,
        stderr=float(result.stderr) if result.stderr is not None else None,
        evaluator_version=str(policy.evaluator_version),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        metadata={
            "method": str(result.method),
            "notes": str(result.notes),
            "source": str(request.candidate.source),
        },
    )


def result_from_record(record: EvaluationCacheRecord) -> EvaluationResult:
    method = str(record.metadata.get("method", "cache"))
    notes = str(record.metadata.get("notes", "Loaded from evaluation cache."))
    return EvaluationResult(
        success=float(record.success),
        mode=record.mode,
        exact=bool(record.exact),
        method=method,
        notes=notes,
        ci_low=record.ci_low,
        ci_high=record.ci_high,
        stderr=record.stderr,
    )


def record_quality_tuple(record: EvaluationCacheRecord) -> tuple[Any, ...]:
    ci_width = _ci_width(record)
    ci_component = float("-inf") if ci_width is None else -ci_width
    return (
        int(record.trust_level),
        int(record.n_eval_samples) if record.n_eval_samples is not None else -1,
        ci_component,
        _version_tuple(record.evaluator_version),
    )


def should_replace_cached(
    existing: EvaluationCacheRecord,
    new: EvaluationCacheRecord,
    policy: CachePolicy,
) -> bool:
    if new.analytical and not policy.persist_analytical:
        return False

    if new.trust_level != existing.trust_level:
        return new.trust_level > existing.trust_level

    if existing.mode in ("dense_mc", "procedural_mc") and new.mode == existing.mode:
        existing_samples = existing.n_eval_samples or 0
        new_samples = new.n_eval_samples or 0
        if new_samples != existing_samples:
            return new_samples > existing_samples

        existing_ci = _ci_width(existing)
        new_ci = _ci_width(new)
        if existing_ci is not None and new_ci is not None and new_ci != existing_ci:
            return new_ci < existing_ci

    if new.evaluator_version != existing.evaluator_version:
        return _version_tuple(new.evaluator_version) > _version_tuple(existing.evaluator_version)

    if (
        policy.recompute_if_uncertain
        and existing.mode in ("dense_mc", "procedural_mc")
        and new.mode == existing.mode
    ):
        existing_samples = existing.n_eval_samples or 0
        new_samples = new.n_eval_samples or 0
        if existing_samples == new_samples:
            existing_ci = _ci_width(existing)
            new_ci = _ci_width(new)
            if existing_ci is None or new_ci is None:
                return True

    return False


class EvaluationCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.jsonl_path = self.cache_dir / "b_performance.jsonl"
        self.latest_path = self.cache_dir / "b_performance_latest.json"
        self._latest: dict[str, EvaluationCacheRecord] = {}

    def load(self) -> None:
        self._latest = {}

        if self.latest_path.exists():
            try:
                latest_payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed latest cache JSON: {self.latest_path}: {exc}") from exc
            if not isinstance(latest_payload, dict):
                raise ValueError(
                    f"Malformed latest cache JSON: {self.latest_path}: expected object at top level."
                )

        if not self.jsonl_path.exists():
            return

        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed cache JSONL line {line_number} in {self.jsonl_path}: {exc}"
                    ) from exc
                record = _record_from_payload(payload, f"{self.jsonl_path}:{line_number}")
                self._latest[record.key] = record

        self._write_latest_json()

    def lookup(self, request: EvaluationRequest, policy: CachePolicy) -> EvaluationCacheRecord | None:
        if not policy.enabled or not policy.read or not policy.prefer_cached:
            return None

        key = cache_key_to_string(_request_cache_key_tuple(request, policy))
        record = self._latest.get(key)
        if record is None:
            return None

        if policy.min_trust_level is not None and record.trust_level < policy.min_trust_level:
            return None

        if policy.min_n_eval_samples is not None:
            if record.n_eval_samples is None or record.n_eval_samples < policy.min_n_eval_samples:
                return None

        if policy.max_ci_width is not None:
            width = _ci_width(record)
            if width is None or width > policy.max_ci_width:
                return None

        return record

    def append(self, record: EvaluationCacheRecord) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=True) + "\n")
        self._latest[record.key] = record
        self._write_latest_json()

    def upsert(
        self,
        request: EvaluationRequest,
        result: EvaluationResult,
        policy: CachePolicy,
    ) -> EvaluationCacheRecord | None:
        if not policy.enabled or not policy.write:
            return None

        new_record = record_from_result(request, result, policy)
        if new_record.analytical and not policy.persist_analytical:
            return None

        existing = self._latest.get(new_record.key)
        if existing is None or should_replace_cached(existing, new_record, policy):
            self.append(new_record)
            return new_record

        return existing

    def all_records(self) -> list[EvaluationCacheRecord]:
        return list(self._latest.values())

    def invalidate_by_mode(self, mode: str) -> None:
        self._latest = {
            key: record for key, record in self._latest.items() if record.mode != mode
        }
        self._write_latest_json()

    def invalidate_by_evaluator_version(self, version: str) -> None:
        self._latest = {
            key: record
            for key, record in self._latest.items()
            if record.evaluator_version != version
        }
        self._write_latest_json()

    def clear(self) -> None:
        self._latest = {}
        if self.jsonl_path.exists():
            self.jsonl_path.unlink()
        if self.latest_path.exists():
            self.latest_path.unlink()

    def _write_latest_json(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {key: asdict(record) for key, record in self._latest.items()}
        with self.latest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=True, indent=2)
