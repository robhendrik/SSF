# Source Modules

Canonical source modules under `src/ssf` used by the evaluator, search, optimizer, and cache flow.

- [hybrid_strategy_spec](hybrid_strategy_spec.md): canonical strategy specification and naming.
- [hybrid_strategy_fixtures](hybrid_strategy_fixtures.md): canonical dense fixture construction and serialization.
- [hybrid_strategy_procedural](hybrid_strategy_procedural.md): procedural strategy implementation matching dense semantics.
- [hybrid_strategy_analytics](hybrid_strategy_analytics.md): closed-form analytical evaluator backend.
- [strategy_evaluation](strategy_evaluation.md): backend-agnostic evaluator dispatcher.
- [strategy_cache](strategy_cache.md): evaluation cache schema and replacement policy.
- [strategy_search](strategy_search.md): legal spec enumeration and exhaustive ranking utilities.
- [strategy_optimizer](strategy_optimizer.md): exhaustive and beam optimizer orchestration.
- [bob_exhaustive](bob_exhaustive.md): dense exhaustive B-side evaluator backend.
- [bob_mc](bob_mc.md): dense Monte Carlo B-side evaluator backend.
- [bob_mc_procedural](bob_mc_procedural.md): procedural Monte Carlo B-side evaluator backend.