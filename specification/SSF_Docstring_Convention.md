Module docstring:
- Purpose
- Architectural role
- Key invariants
- Main entry points
- What this module must not do

Public dataclass/class docstring:
- What it represents
- Ownership/meaning of fields
- Important invariants

Public function docstring:
- What it does
- Parameters only when meaning is non-obvious
- Return meaning
- Raises, especially validation failures
- No duplicated implementation detail

Document intent and invariants, not line-by-line mechanics.
Do not invent behavior.
Do not describe deprecated/legacy flows as current.

Module header:
- SPDX license identifier (optional but recommended)
- Copyright
- Short module summary

"""
Canonical strategy evaluation dispatcher for SSF.

This module owns the backend-agnostic evaluation flow connecting:
- HybridStrategySpec
- evaluator backends
- cache integration
- optimizer-facing APIs

Invariants:
- evaluators remain pure
- cache ownership stays outside evaluator implementations
- optimizer interacts only through evaluate_candidate()

See also:
- strategy_cache.py
- strategy_optimizer.py
"""

# Copyright (c) 2026 Rob Hendrik
# See LICENSE file for details.


File header convention:
- copyright line allowed
- no creation/modified dates
- no per-function author annotations
- Git history is authoritative for authorship/timestamps