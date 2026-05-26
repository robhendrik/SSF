# bob_mc

## Purpose

Approximate dense B-side evaluator using Monte Carlo over canonical dense fixtures.

## Role in canonical architecture

Implements the dense Monte Carlo backend used by `strategy_evaluation.evaluate_candidate` in `dense_mc` mode.

## Input contract

- Strategy payload must contain canonical dense tables (`n2`, `k_box`, `f_tables`, `comm`/`comm_table`).
- Sampling parameters (`n_train_samples`, `n_eval_samples`, `confidence`, `seed`) must be valid.
- `p_rule` must be in `[0, 1]`.

## Evaluation mode relation

- This module is the engine behind `dense_mc`.

## Exact vs approximate

- Approximate: two-phase Monte Carlo (training for Bob policy selection, evaluation for score estimate).

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
- [bob_exhaustive](bob_exhaustive.md)
- [bob_mc_procedural](bob_mc_procedural.md)
