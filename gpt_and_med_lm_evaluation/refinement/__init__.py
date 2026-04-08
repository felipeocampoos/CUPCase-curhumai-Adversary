"""
Iterative Adversarial Refinement with Checklist Enforcement.

This module provides components for:
1. Generating initial diagnostic responses (Generator)
2. Evaluating responses against a configurable checklist (Critic)
3. Applying targeted, minimal edits for failed checklist items (Editor)
4. Iterating until compliance or max iterations
5. Computing metrics: CCR_all, CCR_Q, CCR_H, iterations to compliance, minimality of edits
"""

try:
    from .refiner import IterativeRefiner, JudgeProvider, create_client, create_refiner
    from .variant_factory import create_refiner_variant, list_refiner_variants
    from .variants import (
        DiscriminativeQuestionRefiner,
        DifferentialAuditRefiner,
        DomainRoutedRefiner,
        HeuristicDomainRouter,
        ProgressiveDisclosureRefiner,
        RouteDecision,
        SemanticSimilarityGatedRefiner,
    )
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise

    IterativeRefiner = None
    JudgeProvider = None
    create_client = None
    create_refiner = None
    create_refiner_variant = None
    list_refiner_variants = None
    DiscriminativeQuestionRefiner = None
    DifferentialAuditRefiner = None
    DomainRoutedRefiner = None
    HeuristicDomainRouter = None
    ProgressiveDisclosureRefiner = None
    RouteDecision = None
    SemanticSimilarityGatedRefiner = None
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
    aggregate_minimality_metrics,
    compute_clinical_quality_stats,
    compute_hard_fail_rate,
    compute_compliance_rate,
    compute_curiosity_humility_stats,
)
from .stats import paired_bootstrap_ci, paired_permutation_test
from .io import JSONLLogger, load_refinement_traces
from .run_manifest import (
    compare_run_manifests,
    create_run_manifest,
    load_run_manifest,
    save_run_manifest,
)
from .similarity_gating import (
    Candidate,
    CandidateSet,
    SimilarityResult,
    DiscriminatorResult,
    JinaEmbeddingService,
    compute_similarity_for_top3,
)
from .discriminative_questioning import (
    RankedCandidate,
    RankedCandidateSet,
    DiscriminativeQuestion,
    QuestionAnswerEvidence,
    IntegratedDecision,
    FreeTextIntegratedOutput,
    parse_ranked_candidates,
    parse_discriminative_question,
    parse_answer_extraction,
    parse_integrated_decision_free_text,
)
from .differential_audit import (
    ComparativeDecision,
    ComparativeFreeTextOutput,
    CounterHypothesisSet,
    format_pooled_differential_for_prompt,
    format_seed_candidates_for_prompt,
    merge_differential_pool,
    parse_comparative_evaluation_free_text,
    parse_counter_hypotheses,
)
from .progressive_disclosure import (
    BeliefRevisionScores,
    EarlyCandidate,
    EarlyDifferential,
    EarlyRankedOptions,
    RevisionDecisionFreeText,
    RevisionDecisionMCQ,
    compute_belief_revision_scores,
    parse_early_differential_free_text,
    parse_early_ranked_option_indices_mcq,
    parse_revision_decision_free_text,
    parse_revision_decision_mcq,
    truncate_case_by_fraction,
)

__all__ = [
    "IterativeRefiner",
    "JudgeProvider",
    "create_client",
    "create_refiner",
    "create_refiner_variant",
    "list_refiner_variants",
    "DiagnosticResponse",
    "ChecklistItem",
    "CriticResult",
    "RefinementTrace",
    "parse_diagnostic_response",
    "parse_critic_result",
    "compute_edit_distance",
    "compute_ccr_metrics",
    "compute_minimality_metrics",
    "aggregate_minimality_metrics",
    "compute_clinical_quality_stats",
    "compute_hard_fail_rate",
    "compute_compliance_rate",
    "compute_curiosity_humility_stats",
    "paired_bootstrap_ci",
    "paired_permutation_test",
    "JSONLLogger",
    "load_refinement_traces",
    "create_run_manifest",
    "save_run_manifest",
    "load_run_manifest",
    "compare_run_manifests",
    "DomainRoutedRefiner",
    "DiscriminativeQuestionRefiner",
    "DifferentialAuditRefiner",
    "ProgressiveDisclosureRefiner",
    "SemanticSimilarityGatedRefiner",
    "HeuristicDomainRouter",
    "RouteDecision",
    "Candidate",
    "CandidateSet",
    "SimilarityResult",
    "DiscriminatorResult",
    "JinaEmbeddingService",
    "compute_similarity_for_top3",
    "RankedCandidate",
    "RankedCandidateSet",
    "DiscriminativeQuestion",
    "QuestionAnswerEvidence",
    "IntegratedDecision",
    "FreeTextIntegratedOutput",
    "parse_ranked_candidates",
    "parse_discriminative_question",
    "parse_answer_extraction",
    "parse_integrated_decision_free_text",
    "CounterHypothesisSet",
    "ComparativeDecision",
    "ComparativeFreeTextOutput",
    "parse_counter_hypotheses",
    "parse_comparative_evaluation_free_text",
    "merge_differential_pool",
    "format_seed_candidates_for_prompt",
    "format_pooled_differential_for_prompt",
    "EarlyCandidate",
    "EarlyDifferential",
    "EarlyRankedOptions",
    "RevisionDecisionFreeText",
    "RevisionDecisionMCQ",
    "BeliefRevisionScores",
    "truncate_case_by_fraction",
    "compute_belief_revision_scores",
    "parse_early_differential_free_text",
    "parse_early_ranked_option_indices_mcq",
    "parse_revision_decision_free_text",
    "parse_revision_decision_mcq",
]
