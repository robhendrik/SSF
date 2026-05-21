# Fixture Naming and Directory Convention

## Overview

SSF distinguishes between:

1. Canonical fixtures
2. Legacy compatibility fixtures
3. Volatile/generated optimization fixtures

These categories have different naming conventions, storage locations, and lifecycle policies.

---

# Canonical Fixture Bank

## Directory

```text
fixtures/persistent/a_strategies/
```

## Purpose

Canonical deterministic A-strategy fixtures generated from:

- `HybridStrategySpec`
- `generate_a_strategy_fixtures.py`

These fixtures:

- are committed to the repository,
- are reproducible,
- define the canonical naming convention,
- are used by canonical tests and documentation,
- represent the authoritative strategy specification pipeline.

## Naming Convention

### Majority

```text
majority_n2_<n2>_k_<k_box>_b_<tie_bandwidth>.npz
```

Examples:

```text
majority_n2_4_k_3_b_0.npz
majority_n2_8_k_7_b_2.npz
```

---

### Pyramid

```text
pyramid_n2_<n2>_k_<k_box>.npz
```

Examples:

```text
pyramid_n2_4_k_3.npz
pyramid_n2_8_k_7.npz
```

---

### Horizontal Hybrid

```text
hybrid_horiz_<order>_n2_<n2>_k_<k_box>_split_<left_len>_<right_len>_b_<tie_bandwidth>.npz
```

Where:

- `order` is:
  - `maj_pyr`
  - `pyr_maj`
- `left_len` and `right_len` are the explicit left/right partition sizes that sum to `n2`

Examples:

```text
hybrid_horiz_maj_pyr_n2_8_k_7_split_6_2_b_0.npz
hybrid_horiz_pyr_maj_n2_8_k_7_split_2_6_b_0.npz
```

---

### Vertical Hybrid

```text
hybrid_vert_<order>_n2_<n2>_k_<k_box>_s_<depth_s>_b_<tie_bandwidth>.npz
```

Where:

- `order` is:
  - `maj_pyr`
  - `pyr_maj`

Examples:

```text
hybrid_vert_maj_pyr_n2_8_k_7_s_2_b_0.npz
hybrid_vert_pyr_maj_n2_8_k_7_s_2_b_0.npz
```

---

# Canonical Naming Principles

Canonical fixture names must:

- encode the full strategy geometry,
- be derivable directly from `HybridStrategySpec`,
- avoid ambiguous shorthand,
- avoid implicit assumptions,
- remain stable across generator implementations.

Canonical names are generated via:

```python
candidate_name(spec)
```

and must not be manually constructed elsewhere.

---

# Legacy Compatibility Fixtures

## Directory

```text
fixtures/a_strategies/
```

## Purpose

Temporary compatibility layer for older tests/scripts.

These fixtures:

- use older flat naming conventions,
- may omit geometry details,
- are NOT canonical,
- should gradually be phased out.

## Legacy Naming Examples

```text
hybrid_n2_4_k_3.npz
majority_n2_8_k_1.npz
```

These names are insufficient for canonical hybrid representation because they do not encode:

- horizontal vs vertical,
- ordering,
- split/depth,
- tie bandwidth.

Therefore they must not be used for new development.

---

# Volatile Fixture Bank

## Directory

```text
work/fixture_bank/a_strategies/
```

## Purpose

Non-committed generated fixtures used during:

- optimization,
- beam search,
- Monte Carlo search,
- experimentation,
- caching/memoization.

These fixtures:

- are NOT committed,
- may be deleted safely,
- may contain large/generated search outputs,
- may contain duplicate or superseded strategies.

Naming convention should still use canonical names whenever possible.

---

# Evaluation Memory Bank

## Directory

```text
work/fixture_bank/evaluations/
```

## Purpose

Stores cached B-side evaluation results.

Examples:

```text
b_performance.jsonl
b_performance_latest.json
```

These records track:

- strategy identity,
- evaluation mode,
- p_rule,
- sample count,
- achieved success,
- confidence intervals,
- evaluation timestamp,
- evaluator version.

---

# Migration Policy

## Canonical Consumers

New code must use:

```text
fixtures/persistent/a_strategies/
```

and canonical fixture names.

---

## Legacy Consumers

Older tests/scripts may temporarily continue using:

```text
fixtures/a_strategies/
```

until explicitly migrated.

---

# Deletion Policy

The following directories are transitional and may eventually be removed:

```text
fixtures/a_reference/
fixtures/a_hybrid_family/
fixtures/a_strategies/
fixtures/a_strategies_hybrid_family/
```

Only after all consumers migrate to canonical naming and canonical manifests.