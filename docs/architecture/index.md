# Canonical Architecture

## Purpose

SSF uses a single canonical strategy representation and a backend-agnostic evaluation interface so optimizers can compare strategies without depending on backend internals.

## Core Ownership

| Concern | Canonical Owner |
|---|---|
| Strategy schema, naming, validity | [hybrid_strategy_spec](../api/src/hybrid_strategy_spec.md) |
| Dense fixture semantics | [hybrid_strategy_fixtures](../api/src/hybrid_strategy_fixtures.md) |
| Procedural semantics | [hybrid_strategy_procedural](../api/src/hybrid_strategy_procedural.md) |
| Closed-form analytical model | [hybrid_strategy_analytics](../api/src/hybrid_strategy_analytics.md) |
| Unified evaluator dispatcher | [strategy_evaluation](../api/src/strategy_evaluation.md) |
| Evaluation cache and replacement policy | [strategy_cache](../api/src/strategy_cache.md) |
| Legal spec enumeration | [strategy_search](../api/src/strategy_search.md) |
| Exhaustive and beam optimizers | [strategy_optimizer](../api/src/strategy_optimizer.md) |

## Canonical Evaluation Modes

- `analytical`
- `dense_exhaustive`
- `dense_mc`
- `procedural_mc`

## Design Invariants

- Canonical names are generated via `candidate_name(...)` only.
- Optimizers score candidates through the dispatcher (`evaluate_candidate(...)`).
- Cache keys include strategy identity, evaluation mode, and evaluator configuration.
- Dense and procedural implementations are equivalent strategy semantics with different runtime forms.

## Validation and Operations

Validation and operational scripts are documented in [API / scripts](../api/scripts/generate_a_strategy_fixtures.md).
