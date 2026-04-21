"""Uncertainty-triggered multi-signal gated refinement variant."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from ..refiner import IterativeRefiner
from ..schema import CriticResult, DiagnosticResponse
from ..similarity_gating import (
    JinaEmbeddingService,
    compute_similarity_for_top3,
    format_candidates_for_prompt,
    format_similarity_for_prompt,
    parse_candidate_set,
    parse_free_text_discriminator_output,
)


UNCERTAINTY_HEDGE_PATTERNS = (
    r"\buncertain\b",
    r"\buncertainty\b",
    r"\bnot certain\b",
    r"\bmoderate confidence\b",
    r"\blow confidence\b",
    r"\bcannot rule out\b",
    r"\bneed more data\b",
    r"\bfurther testing\b",
    r"\bwould confirm\b",
    r"\bif .* then\b",
)


class UncertaintyConsistencyGatedRefiner(IterativeRefiner):
    """Adaptive gate using candidate agreement, uncertainty, and critic risk."""

    variant_name: str = "uncertainty_consistency_gated"

    def __init__(
        self,
        *args,
        similarity_threshold: Optional[float] = None,
        confidence_margin_threshold: Optional[float] = None,
        embedding_service: Optional[JinaEmbeddingService] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if similarity_threshold is None:
            similarity_threshold = self.config.similarity_threshold
        if confidence_margin_threshold is None:
            confidence_margin_threshold = self.config.confidence_margin_threshold
        self.similarity_threshold = similarity_threshold
        self.confidence_margin_threshold = confidence_margin_threshold
        self.embedding_service = embedding_service or JinaEmbeddingService()
        self._candidate_template = self._load_prompt("candidate_free_text")
        self._discriminator_template = self._load_prompt("discriminator_free_text")
        self._case_telemetry: Dict[str, object] = {}

    def refine(self, *args, **kwargs):
        self._case_telemetry = {}
        return super().refine(*args, **kwargs)

    def generate(self, case_text: str):
        self._case_telemetry["similarity_threshold"] = self.similarity_threshold
        self._case_telemetry["confidence_margin_threshold"] = self.confidence_margin_threshold

        candidate_set = self._run_candidate_pass(case_text)
        baseline = self._run_baseline_generation(case_text)
        if baseline is None:
            return super().generate(case_text)
        baseline_response, baseline_raw = baseline

        critic_result = self._run_verifier(case_text, baseline_response)
        uncertainty_triggered = self._has_uncertainty_signal(baseline_response)

        semantic_triggered = False
        confidence_margin_triggered = False
        similarity = None
        top3 = []

        if candidate_set is not None:
            top3 = candidate_set.candidates[:3]
            self._case_telemetry["candidate_top3"] = [candidate.label for candidate in top3]
            self._case_telemetry["candidate_confidences"] = [
                candidate.confidence for candidate in top3
            ]
            if len(top3) >= 3:
                try:
                    similarity = compute_similarity_for_top3(
                        [candidate.label for candidate in top3],
                        threshold=self.similarity_threshold,
                        embedding_service=self.embedding_service,
                    )
                    semantic_triggered = similarity.gate_triggered
                    self._case_telemetry["pairwise_cosine"] = similarity.pairwise_cosine
                    self._case_telemetry["mean_cosine"] = similarity.mean_cosine
                except Exception as exc:
                    self._case_telemetry["similarity_error"] = str(exc)

            if len(top3) >= 2:
                top1 = top3[0].confidence
                top2 = top3[1].confidence
                if top1 is not None and top2 is not None:
                    margin = float(top1) - float(top2)
                    self._case_telemetry["confidence_margin"] = margin
                    confidence_margin_triggered = margin <= self.confidence_margin_threshold

        verifier_triggered = critic_result is None or self._critic_has_risk(critic_result)
        signal_count = sum(
            int(flag)
            for flag in (
                semantic_triggered,
                confidence_margin_triggered,
                uncertainty_triggered,
                verifier_triggered,
            )
        )

        self._case_telemetry["semantic_gate_triggered"] = semantic_triggered
        self._case_telemetry["confidence_margin_triggered"] = confidence_margin_triggered
        self._case_telemetry["uncertainty_triggered"] = uncertainty_triggered
        self._case_telemetry["verifier_triggered"] = verifier_triggered
        self._case_telemetry["signal_count"] = signal_count
        self._case_telemetry["baseline_uncertainty"] = baseline_response.uncertainty or ""
        self._case_telemetry["baseline_final_diagnosis"] = baseline_response.final_diagnosis

        if critic_result is not None:
            self._case_telemetry["verifier_quality_score"] = critic_result.clinical_quality.score
            self._case_telemetry["verifier_hard_fail"] = critic_result.hard_fail.failed
            self._case_telemetry["verifier_failed_items"] = [
                item.item_id for item in critic_result.get_failed_items()
            ]
        else:
            self._case_telemetry["verifier_failed_items"] = []

        high_instability = (
            verifier_triggered
            and (semantic_triggered or uncertainty_triggered or confidence_margin_triggered)
        ) or (critic_result.hard_fail.failed if critic_result is not None else False) or signal_count >= 2

        if signal_count == 0:
            self._case_telemetry["gate_triggered"] = False
            self._case_telemetry["escalation_tier"] = "stable"
            self._case_telemetry["final_selection_source"] = "baseline_generator"
            return baseline_response, baseline_raw

        if not high_instability:
            guided_retry = self._run_guided_retry(case_text, baseline_response, critic_result)
            if guided_retry is not None:
                self._case_telemetry["gate_triggered"] = True
                self._case_telemetry["escalation_tier"] = "guided_retry"
                self._case_telemetry["final_selection_source"] = "guided_retry"
                return guided_retry

            self._case_telemetry["guided_retry_error"] = "guided_retry_unavailable"
            self._case_telemetry["gate_triggered"] = True
            self._case_telemetry["escalation_tier"] = "guided_retry"
            self._case_telemetry["final_selection_source"] = "baseline_fallback_guided_retry_error"
            return baseline_response, baseline_raw

        discriminator_result = self._run_discriminator(
            case_text=case_text,
            top3=top3,
            similarity=similarity,
            baseline_response=baseline_response,
            critic_result=critic_result,
        )
        if discriminator_result is not None:
            self._case_telemetry["gate_triggered"] = True
            self._case_telemetry["escalation_tier"] = "discriminator"
            self._case_telemetry["discriminator_invoked"] = True
            self._case_telemetry["final_selection_source"] = "discriminator_pass"
            return discriminator_result

        guided_retry = self._run_guided_retry(case_text, baseline_response, critic_result)
        if guided_retry is not None:
            self._case_telemetry["gate_triggered"] = True
            self._case_telemetry["escalation_tier"] = "discriminator"
            self._case_telemetry["final_selection_source"] = "guided_retry_fallback_after_discriminator_error"
            return guided_retry

        self._case_telemetry["gate_triggered"] = True
        self._case_telemetry["escalation_tier"] = "discriminator"
        self._case_telemetry["final_selection_source"] = "baseline_fallback_discriminator_error"
        return baseline_response, baseline_raw

    def _run_candidate_pass(self, case_text: str):
        candidate_prompt = self._candidate_template.replace("{case_text}", case_text)
        try:
            candidate_set, _ = self._call_api(
                model=self.config.generator_model,
                prompt=candidate_prompt,
                parse_fn=parse_candidate_set,
            )
            return candidate_set
        except Exception as exc:
            self._case_telemetry["candidate_error"] = str(exc)
            return None

    def _run_baseline_generation(self, case_text: str):
        try:
            return super().generate(case_text)
        except Exception as exc:
            self._case_telemetry["baseline_error"] = str(exc)
            return None

    def _run_verifier(
        self,
        case_text: str,
        response: DiagnosticResponse,
    ) -> Optional[CriticResult]:
        try:
            critic_result, _ = self.critique(case_text, response)
            return critic_result
        except Exception as exc:
            self._case_telemetry["verifier_error"] = str(exc)
            return None

    def _run_guided_retry(
        self,
        case_text: str,
        response: DiagnosticResponse,
        critic_result: Optional[CriticResult],
    ):
        edit_plan: List[str] = []
        if critic_result is not None:
            edit_plan = critic_result.edit_plan or []
        if not edit_plan:
            if critic_result is None:
                edit_plan = [
                    "[GENERAL] Reassess the answer conservatively because verifier feedback was unavailable. Preserve supported content, reduce unsupported certainty, and return a valid structured response."
                ]
            else:
                failed_items = critic_result.get_failed_items()
                if failed_items:
                    edit_plan = [
                        f"[GENERAL] Improve the response for failed checklist item {item.item_id}: {item.rationale}"
                        for item in failed_items[:2]
                    ]
                else:
                    quality = critic_result.clinical_quality.score
                    if quality < self.config.clinical_quality_threshold:
                        edit_plan = [
                            "[GENERAL] Improve the clinical usefulness and calibration of the response while preserving supported content."
                        ]
                    elif self._has_uncertainty_signal(response):
                        edit_plan = [
                            "[GENERAL] Recalibrate the response for uncertainty: strengthen discriminatory questioning, conditional reasoning, and next-step guidance while preserving supported content."
                        ]
        if not edit_plan:
            return None
        try:
            edited_response, raw = self.edit(case_text, response, edit_plan)
            self._case_telemetry["guided_retry_edit_plan"] = edit_plan
            return edited_response, raw
        except Exception as exc:
            self._case_telemetry["guided_retry_error"] = str(exc)
            return None

    def _run_discriminator(
        self,
        *,
        case_text: str,
        top3,
        similarity,
        baseline_response: DiagnosticResponse,
        critic_result: Optional[CriticResult],
    ):
        if len(top3) < 3:
            self._case_telemetry["discriminator_error"] = "insufficient_candidates"
            return None

        critic_block = self._format_critic_block(critic_result)
        uncertainty_block = self._format_uncertainty_block(baseline_response)
        similarity_block = (
            format_similarity_for_prompt(similarity)
            if similarity is not None
            else "Similarity diagnostics unavailable."
        )

        discriminator_prompt = self._discriminator_template.replace(
            "{case_text}", case_text
        ).replace(
            "{candidate_block}", format_candidates_for_prompt(top3)
        ).replace(
            "{similarity_block}", similarity_block
        ).replace(
            "{baseline_response}", baseline_response.to_json()
        ).replace(
            "{critic_block}", critic_block
        ).replace(
            "{uncertainty_block}", uncertainty_block
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

            self._case_telemetry["discriminator_rationale"] = discriminator.rationale
            self._case_telemetry["differentiators"] = discriminator.differentiators
            return response, raw
        except Exception as exc:
            self._case_telemetry["discriminator_error"] = str(exc)
            return None

    @staticmethod
    def _has_uncertainty_signal(response: DiagnosticResponse) -> bool:
        uncertainty_text = " ".join(
            part
            for part in (
                response.uncertainty or "",
                response.conditional_reasoning or "",
            )
            if part
        )
        normalized = uncertainty_text.lower()
        return any(re.search(pattern, normalized) for pattern in UNCERTAINTY_HEDGE_PATTERNS)

    def _critic_has_risk(self, critic_result: Optional[CriticResult]) -> bool:
        if critic_result is None:
            return False
        if critic_result.hard_fail.failed:
            return True
        if critic_result.clinical_quality.score < self.config.clinical_quality_threshold:
            return True
        return not critic_result.is_compliant(self.config.clinical_quality_threshold)

    @staticmethod
    def _format_critic_block(critic_result: Optional[CriticResult]) -> str:
        if critic_result is None:
            return "Critic diagnostics unavailable."
        failed = critic_result.get_failed_items()
        failed_lines = (
            "\n".join(f"- {item.item_id}: {item.rationale}" for item in failed)
            if failed
            else "- None"
        )
        edit_plan_lines = (
            "\n".join(f"- {item}" for item in critic_result.edit_plan)
            if critic_result.edit_plan
            else "- None"
        )
        return (
            f"Clinical quality score: {critic_result.clinical_quality.score}\n"
            f"Hard fail: {critic_result.hard_fail.failed}\n"
            f"Hard fail reason: {critic_result.hard_fail.reason or 'None'}\n"
            f"Failed checklist items:\n{failed_lines}\n"
            f"Suggested edits:\n{edit_plan_lines}"
        )

    @staticmethod
    def _format_uncertainty_block(response: DiagnosticResponse) -> str:
        return (
            f"Uncertainty field: {response.uncertainty or 'None'}\n"
            f"Conditional reasoning: {response.conditional_reasoning or 'None'}\n"
            f"Clarifying questions: {response.clarifying_questions or []}"
        )

    def _get_case_variant_metadata(self) -> Dict[str, object]:
        return dict(self._case_telemetry)

    @staticmethod
    def _load_prompt(name: str) -> str:
        prompts_dir = (
            Path(__file__).resolve().parent.parent / "prompts" / "uncertainty_consistency"
        )
        path = prompts_dir / f"{name}.md"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
