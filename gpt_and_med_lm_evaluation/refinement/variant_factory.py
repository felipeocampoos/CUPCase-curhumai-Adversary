"""Factory and registry for refinement variants."""

from typing import Dict, List, Optional, Type

from .refiner import (
    IterativeRefiner,
    RefinerConfig,
    JudgeProvider,
    create_client,
)
from .variants import (
    DiscriminativeQuestionRefiner,
    DomainRoutedRefiner,
    SemanticSimilarityGatedRefiner,
)


VARIANT_REGISTRY: Dict[str, Type[IterativeRefiner]] = {
    "baseline": IterativeRefiner,
    "domain_routed": DomainRoutedRefiner,
    "semantic_similarity_gated": SemanticSimilarityGatedRefiner,
    "discriminative_question": DiscriminativeQuestionRefiner,
}


def list_refiner_variants() -> List[str]:
    """Return sorted list of available refiner variants."""
    return sorted(VARIANT_REGISTRY.keys())


def create_refiner_variant(
    variant: str,
    api_key: Optional[str] = None,
    config: Optional[RefinerConfig] = None,
    provider: Optional[JudgeProvider] = None,
) -> IterativeRefiner:
    """
    Create a refiner by variant name.

    Args:
        variant: Variant key from `VARIANT_REGISTRY`
        api_key: Optional API key
        config: Optional refiner config
        provider: Optional provider override

    Returns:
        IterativeRefiner instance for the requested variant

    Raises:
        ValueError: If variant is unknown
    """
    if config is None:
        config = RefinerConfig()

    variant_key = variant.strip().lower()
    refiner_cls = VARIANT_REGISTRY.get(variant_key)
    if refiner_cls is None:
        valid = ", ".join(list_refiner_variants())
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {valid}")

    # Determine provider: explicit arg > config > default
    if provider is None:
        provider = config.provider

    client = create_client(provider=provider, api_key=api_key)
    return refiner_cls(client=client, config=config)
