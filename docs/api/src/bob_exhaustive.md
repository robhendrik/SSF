# bob_exhaustive

## Purpose

Exact dense B-side evaluator over canonical dense fixtures.

## Role in canonical architecture

Implements the dense exhaustive backend used by `strategy_evaluation.evaluate_candidate` in `dense_exhaustive` mode.

## Input contract

- Strategy payload must contain canonical dense tables (`n2`, `k_box`, `f_tables`, `comm`/`comm_table`).
- Table shapes must satisfy the dense fixture contract.
- `p_rule` must be in `[0, 1]`.
- `k_box=0` is valid (no PR boxes): `f_tables` may be empty and `comm` shape must be `(2**n2, 1)`.

## Evaluation mode relation

- This module is the engine behind `dense_exhaustive`.

## Exact vs approximate

- Exact: enumerates restricted Bob responses for allowed subset policy/box limit.

## Cache and evaluator relationship

- This module does not own cache policy.
- Cache read/write is handled by `strategy_evaluation` and `strategy_cache`.

## Must not do

- No strategy-name shortcuts.
- No expected-score shortcuts.
- No fixture-name assumptions.

## See also

- [strategy_evaluation](strategy_evaluation.md)
- [strategy_cache](strategy_cache.md)
- [bob_mc](bob_mc.md)
- [bob_mc_procedural](bob_mc_procedural.md)
