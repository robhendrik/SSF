
# SSF Canonical Architecture and Strategy Specification

## Read and follow:
docs/architecture/SSF_Canonical_Architecture_and_Strategy_Specification.md

This document is authoritative for:
- fixture naming
- evaluator abstraction
- optimizer structure
- canonical ownership
- migration policy

Do not deviate from it.
Do not recreate deprecated generators or naming schemes.

## Overview

This document consolidates the canonical SSF architecture, fixture contracts,
naming conventions, hybrid strategy definitions, evaluator abstraction,
optimizer design, and migration rules.

It is the authoritative architectural reference for:
- fixture generation,
- evaluator implementation,
- optimizer implementation,
- procedural vs dense equivalence,
- canonical naming,
- cache policy,
- testing policy.

---

# 1. Canonical Ownership Rules

The canonical representation of a strategy is:

```python
HybridStrategySpec
```

All strategy semantics must derive from this object.

Canonical functions/modules:
- HybridStrategySpec
- candidate_name(spec)
- validate_spec(spec)
- required_boxes(spec)
- build_strategy_fixture(spec)
- build_procedural_strategy(spec)

No other module may redefine:
- fixture semantics,
- naming,
- required PR-box count,
- strategy geometry.

---

# 2. Canonical Fixture Bank

Directory:

```text
fixtures/persistent/a_strategies/
```

Canonical fixtures:
- are deterministic,
- reproducible,
- generated only via generate_a_strategy_fixtures.py,
- authoritative for tests and documentation.

---

# 3. Dense Fixture Contract

Dense fixtures store:

```text
n2
k_box
comm_table
f_table_0 ... f_table_{k_box-1}
```

For box index d:

```text
f_table_d.shape == (2**n2, 2**d)
```

Meaning:
- f_table_d stores the INPUT bit sent into PR box d
- comm_table stores Alice's final communication bit

Unused boxes must not influence comm_table.

Canonical pyramid ordering:
- breadth-first,
- layer-by-layer,
- left-to-right.

---

# 4. Canonical Fixture Naming

## Majority

```text
majority_n2_<n2>_k_<k_box>_b_<tie_bandwidth>.npz
```

## Pyramid

```text
pyramid_n2_<n2>_k_<k_box>.npz
```

## Horizontal Hybrid

```text
hybrid_horiz_<order>_n2_<n2>_k_<k_box>_split_<left_len>_<right_len>_b_<tie_bandwidth>.npz
```

## Vertical Hybrid

```text
hybrid_vert_<order>_n2_<n2>_k_<k_box>_s_<depth_s>_b_<tie_bandwidth>.npz
```

Canonical names are generated ONLY via:

```python
candidate_name(spec)
```

No handwritten canonical names allowed.

---

# 5. Fixture Categories

## Canonical

```text
fixtures/persistent/a_strategies/
```

## Legacy Compatibility

```text
fixtures/a_strategies/
```

Temporary only.

## Volatile Optimization Fixtures

```text
work/fixture_bank/a_strategies/
```

Not committed.

## Evaluation Cache

```text
work/fixture_bank/evaluations/
```

---

# 6. Procedural Strategy Contract

Procedural strategies expose:

```python
strategy.n2
strategy.k_box
strategy.measure_a(box_idx, field, out_a_prefix) -> int
strategy.comm(field, out_a) -> int
```

Procedural semantics must exactly match dense fixtures.

---

# 7. Evaluator Architecture

Evaluation modes:

```text
analytical
dense_exhaustive
dense_mc
procedural_mc
```

Optimizers must only see:

```python
EvaluationResult.success
```

They must not know:
- dense vs procedural,
- exhaustive vs MC,
- analytical vs simulated.

---

# 8. Evaluator Backends

## AnalyticalEvaluator

Uses:

```python
analytical_win_rate(spec, p_rule)
```

## DenseExhaustiveEvaluator

```text
spec
→ build_strategy_fixture(spec)
→ evaluate_strategy_exhaustive(...)
```

## DenseMonteCarloEvaluator

```text
spec
→ build_strategy_fixture(spec)
→ evaluate_strategy_mc(...)
```

## ProceduralMonteCarloEvaluator

```text
spec
→ build_procedural_strategy(spec)
→ evaluate_strategy_mc_procedural(...)
```

---

# 9. Optimizer Architecture

Reusable optimizer logic belongs in:

```text
src/ssf/
```

CLI wrappers belong in:

```text
scripts/optimize/
```

Search engines:
- exhaustive search,
- beam search.

Optimizer flow:

```text
generate spec
→ evaluate
→ rank
→ mutate/prune
```

---

# 10. Evaluation Cache

Cache keys must include:
- stable_key(spec)
- mode
- p_rule
- box_limit
- subset_policy
- samples
- evaluator_version

---

# 11. Forbidden Patterns

Forbidden:
- manually constructing canonical names,
- duplicating fixture semantics,
- importing deleted legacy generators,
- evaluator shortcuts based on strategy names,
- embedded expected-score tables,
- bypassing evaluator abstraction.

---

# 12. Testing Philosophy

Canonical tests must:
- derive names from specs,
- compare dense vs procedural,
- compare MC vs exhaustive,
- compare analytical vs exhaustive.

Evaluators must not:
- branch on strategy names,
- embed analytical answers,
- bypass fixture tables.

---

# 13. Migration Policy

New code must use:

```text
fixtures/persistent/a_strategies/
```

Legacy fixture folders remain temporary compatibility layers only.

---

# 14. Recommended Build Sequence

1. Freeze canonical fixture semantics
2. Freeze naming semantics
3. Build evaluator abstraction
4. Implement analytical evaluator
5. Implement dense exhaustive evaluator
6. Implement dense MC evaluator
7. Implement procedural MC evaluator
8. Build unified dispatcher
9. Add evaluation cache
10. Build exhaustive optimizer
11. Build beam optimizer
12. Rewrite CLI wrappers
13. Remove legacy optimizers

---

# 15. Source Documents

This document consolidates:
- Dense Fixture Contract for A-Side
- Fixture Naming and Directory Convention
- hybrid_family_final
- optimizer/evaluator planning discussions
