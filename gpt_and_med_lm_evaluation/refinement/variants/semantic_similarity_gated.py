"""Semantic similarity gated differential reasoning variant."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ..refiner import IterativeRefiner
from ..schema import parse_diagnostic_response
from ..similarity_gating import (
    JinaEmbeddingService,
    compute_similarity_for_top3,
    format_candidates_for_prompt,
    format_similarity_for_prompt,
    parse_candidate_set,
    parse_free_text_discriminator_output,
)


class SemanticSimilarityGatedRefiner(IterativeRefiner):
    """Refiner variant that triggers discriminator reasoning on clustered candidates."""

    variant_name: str = "semantic_similarity_gated"

    def __init__(
        self,
        *args,
        similarity_threshold: Optional[float] = None,
        embedding_service: Optional[JinaEmbeddingService] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if similarity_threshold is None:
            similarity_threshold = self.config.similarity_threshold
        self.similarity_threshold = similarity_threshold
        self.embedding_service = embedding_service or JinaEmbeddingService()
        self._candidate_template = self._load_prompt("candidate_free_text")
        self._discriminator_template = self._load_prompt("discriminator_free_text")
        self._case_telemetry: Dict[str, object] = {}

    def refine(self, *args, **kwargs):
        self._case_telemetry = {}
        return super().refine(*args, **kwargs)

    def generate(self, case_text: str):
        self._case_telemetry["similarity_threshold"] = self.similarity_threshold

        # Pass 1: model-ranked top candidates.
        candidate_prompt = self._candidate_template.replace("{case_text}", case_text)
        try:
            candidate_set, _ = self._call_api(
                model=self.config.generator_model,
                prompt=candidate_prompt,
                parse_fn=parse_candidate_set,
            )
        except Exception as exc:
            self._case_telemetry["candidate_error"] = str(exc)
            self._case_telemetry["gate_triggered"] = False
            self._case_telemetry["final_selection_source"] = "baseline_fallback_candidate_error"
            return super().generate(case_text)

        top3 = candidate_set.candidates[:3]
        self._case_telemetry["candidate_top3"] = [candidate.label for candidate in top3]
        self._case_telemetry["candidate_confidences"] = [
            candidate.confidence for candidate in top3
        ]

        # Pass 2: similarity gate.
        try:
            similarity = compute_similarity_for_top3(
                [candidate.label for candidate in top3],
                threshold=self.similarity_threshold,
                embedding_service=self.embedding_service,
            )
        except Exception as exc:
            self._case_telemetry["similarity_error"] = str(exc)
            self._case_telemetry["gate_triggered"] = False
            self._case_telemetry["final_selection_source"] = "baseline_fallback_similarity_error"
            return super().generate(case_text)

        self._case_telemetry["pairwise_cosine"] = similarity.pairwise_cosine
        self._case_telemetry["mean_cosine"] = similarity.mean_cosine
        self._case_telemetry["gate_triggered"] = similarity.gate_triggered

        if not similarity.gate_triggered:
            self._case_telemetry["discriminator_invoked"] = False
            self._case_telemetry["final_selection_source"] = "baseline_generator"
            return super().generate(case_text)

        # Pass 3: discriminator reasoning.
        discriminator_prompt = self._discriminator_template.replace(
            "{case_text}", case_text
        ).replace(
            "{candidate_block}", format_candidates_for_prompt(top3)
        ).replace(
            "{similarity_block}", format_similarity_for_prompt(similarity)
        )

        try:
            discriminator_output, raw = self._call_api(
                model=self.config.generator_model,
                prompt=discriminator_prompt,
                parse_fn=parse_free_text_discriminator_output,
            )
            response = discriminator_output.response
            discriminator = discriminator_output.discriminator

            if discriminator.final_choice:
                response.final_diagnosis = discriminator.final_choice

            if discriminator.differentiators:
                differentiator_text = "; ".join(discriminator.differentiators)
                base_reasoning = response.conditional_reasoning or ""
                response.conditional_reasoning = (
                    f"Key differentiators: {differentiator_text}. {base_reasoning}"
                ).strip()

            self._case_telemetry["discriminator_invoked"] = True
            self._case_telemetry["discriminator_rationale"] = discriminator.rationale
            self._case_telemetry["differentiators"] = discriminator.differentiators
            self._case_telemetry["final_selection_source"] = "discriminator_pass"
            return response, raw
        except Exception as exc:
            self._case_telemetry["discriminator_error"] = str(exc)
            self._case_telemetry["discriminator_invoked"] = True
            self._case_telemetry["final_selection_source"] = "baseline_fallback_discriminator_error"
            return super().generate(case_text)

    def _get_case_variant_metadata(self) -> Dict[str, object]:
        return dict(self._case_telemetry)

    @staticmethod
    def _load_prompt(name: str) -> str:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts" / "semantic_similarity"
        path = prompts_dir / f"{name}.md"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
