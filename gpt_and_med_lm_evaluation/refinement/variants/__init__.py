"""Variant implementations for refinement experiments."""

from .domain_routed import (
    DomainRoutedRefiner,
    HeuristicDomainRouter,
    RouteDecision,
)
from .semantic_similarity_gated import SemanticSimilarityGatedRefiner
from .discriminative_question import DiscriminativeQuestionRefiner
from .differential_audit import DifferentialAuditRefiner
from .progressive_disclosure import ProgressiveDisclosureRefiner

__all__ = [
    "DomainRoutedRefiner",
    "HeuristicDomainRouter",
    "RouteDecision",
    "SemanticSimilarityGatedRefiner",
    "DiscriminativeQuestionRefiner",
    "DifferentialAuditRefiner",
    "ProgressiveDisclosureRefiner",
]
