## Full plan status


0. Canonical strategy core                    ✅ done
   HybridStrategySpec
   candidate_name
   validate_spec
   required_boxes

1. Canonical dense fixtures                   ✅ done
   hybrid_strategy_fixtures.py
   canonical generator
   persistent fixture directory

2. Canonical naming convention                ✅ done
   majority/pyramid/horizontal/vertical names
   no manual fixture-name construction

3. Procedural A-side strategy                 ✅ done
   hybrid_strategy_procedural.py
   dense/procedural equivalence tests

4. Bob evaluator hardening                    ✅ done
   bob_exhaustive no shortcuts
   MC vs exhaustive checks
   canonical fixture lookup

5. Evaluator abstraction                      ✅ done
   strategy_evaluation.py
   EvaluationRequest / EvaluationResult
   evaluate_candidate()

6. Analytical backend                         ✅ done

7. Dense exhaustive backend                   ✅ done

8. Dense MC backend                           ✅ done

9. Procedural MC backend                      ✅ done / validate complete

10. Cross-backend validation script           ✅ done
    compare_strategy_evaluation_backends.py

11. Analytical search scan                    ✅ done
    strategy_search.py
    run_analytical_strategy_scan.py
    optimal envelope plot overlay

12. Evaluation cache 7a                       ⬅️ next
    strategy_cache.py
    cache policy
    trust hierarchy
    JSONL/latest storage

13. Cache integration 7b                      next
    optional cache in evaluate_candidate()

14. Cache rebuild/sweep 7c                    next
    rebuild_evaluation_cache.py

15. Exhaustive A-side optimizer               next
    small exact candidate search using evaluator backend

16. Beam search optimizer                     next
    backend-agnostic beam search

17. Optimizer CLI rewrite                     later
    scripts become thin wrappers

18. Batching/performance optimization         later
    batch evaluate candidates
    reduce Numba/packing overhead

19. Legacy cleanup                            final
    delete old generators/scripts/fixtures

