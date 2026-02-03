"""
Iterative Adversarial Refinement with Checklist Enforcement.

This module provides components for:
1. Generating initial diagnostic responses (Generator)
2. Evaluating responses against a configurable checklist (Critic)
3. Applying targeted, minimal edits for failed checklist items (Editor)
4. Iterating until compliance or max iterations
5. Computing metrics: CCR_all, CCR_Q, CCR_H, iterations to compliance, minimality of edits
"""

from .refiner import IterativeRefiner
from .schema import (
    DiagnosticResponse,
    ChecklistItem,
    CriticResult,
    RefinementTrace,
    parse_diagnostic_response,
    parse_critic_result,
)
from .metrics import (
    compute_edit_distance,
    compute_ccr_metrics,
    compute_minimality_metrics,
)
from .stats import paired_bootstrap_ci, paired_permutation_test
from .io import JSONLLogger, load_refinement_traces

__all__ = [
    "IterativeRefiner",
    "DiagnosticResponse",
    "ChecklistItem",
    "CriticResult",
    "RefinementTrace",
    "parse_diagnostic_response",
    "parse_critic_result",
    "compute_edit_distance",
    "compute_ccr_metrics",
    "compute_minimality_metrics",
    "paired_bootstrap_ci",
    "paired_permutation_test",
    "JSONLLogger",
    "load_refinement_traces",
]
