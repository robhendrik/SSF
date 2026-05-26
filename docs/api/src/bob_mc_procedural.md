# bob_mc_procedural

## Purpose

Approximate procedural B-side evaluator using Monte Carlo against the procedural A-side contract.

## Role in canonical architecture

Implements the procedural Monte Carlo backend used by `strategy_evaluation.evaluate_candidate` in `procedural_mc` mode.

## Input contract

- Strategy must expose canonical procedural API:
  - `n2`
  - `k_box`
  - `measure_a(box_idx, field, out_a_prefix) -> int`
  - `comm(field, out_a) -> int`
- Outputs from `measure_a` and `comm` must be binary.
- Sampling parameters and `p_rule` must be valid.

## Evaluation mode relation

- This module is the engine behind `procedural_mc`.

## Exact vs approximate

- Approximate: two-phase Monte Carlo with the same high-level Bob search structure as dense MC.

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
- [hybrid_strategy_procedural](hybrid_strategy_procedural.md)
- [bob_exhaustive](bob_exhaustive.md)
- [bob_mc](bob_mc.md)
