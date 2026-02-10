"""Self-generated discriminative question variant for free-text refinement."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..discriminative_questioning import (
    format_evidence_for_prompt,
    format_ranked_candidates_for_prompt,
    parse_answer_extraction,
    parse_discriminative_question,
    parse_integrated_decision_free_text,
    parse_ranked_candidates,
)
from ..refiner import IterativeRefiner


class DiscriminativeQuestionRefiner(IterativeRefiner):
    """Variant that generates one discriminative question and integrates its answer."""

    variant_name: str = "discriminative_question"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._candidate_template = self._load_prompt("candidate_free_text")
        self._question_template = self._load_prompt("question_free_text")
        self._answer_template = self._load_prompt("answer_extraction_free_text")
        self._integration_template = self._load_prompt("integrate_free_text")
        self._case_telemetry: Dict[str, object] = {}

    def refine(self, *args, **kwargs):
        self._case_telemetry = {}
        return super().refine(*args, **kwargs)

    def generate(self, case_text: str):
        # Pass 1: ranked candidates.
        candidate_prompt = self._candidate_template.replace("{case_text}", case_text)
        try:
            candidate_set, _ = self._call_api(
                model=self.config.generator_model,
                prompt=candidate_prompt,
                parse_fn=parse_ranked_candidates,
            )
        except Exception as exc:
            self._case_telemetry["candidate_error"] = str(exc)
            self._case_telemetry["final_selection_source"] = "baseline_fallback_candidate_error"
            return super().generate(case_text)

        top_candidates = candidate_set.candidates[:3]
        top2 = top_candidates[:2]

        self._case_telemetry["candidate_top3"] = [candidate.label for candidate in top_candidates]
        self._case_telemetry["top2_for_question"] = [candidate.label for candidate in top2]

        # Pass 2: one discriminative question.
        question_prompt = self._question_template.replace(
            "{case_text}", case_text
        ).replace(
            "{candidate_block}", format_ranked_candidates_for_prompt(top2)
        )

        try:
            question_result, _ = self._call_api(
                model=self.config.generator_model,
                prompt=question_prompt,
                parse_fn=parse_discriminative_question,
            )
        except Exception as exc:
            self._case_telemetry["question_error"] = str(exc)
            self._case_telemetry["final_selection_source"] = "baseline_fallback_question_error"
            return super().generate(case_text)

        self._case_telemetry["discriminative_question"] = question_result.question
        self._case_telemetry["question_target_variable"] = question_result.target_variable

        # Pass 3: answer extraction from case.
        answer_prompt = self._answer_template.replace(
            "{case_text}", case_text
        ).replace(
            "{question}", question_result.question
        ).replace(
            "{target_variable}", question_result.target_variable
        )

        try:
            answer_result, _ = self._call_api(
                model=self.config.generator_model,
                prompt=answer_prompt,
                parse_fn=parse_answer_extraction,
            )
        except Exception as exc:
            self._case_telemetry["answer_error"] = str(exc)
            self._case_telemetry["final_selection_source"] = "baseline_fallback_answer_error"
            return super().generate(case_text)

        self._case_telemetry["extracted_answer"] = answer_result.answer
        self._case_telemetry["answer_confidence"] = answer_result.confidence
        self._case_telemetry["evidence_spans"] = answer_result.evidence_spans

        # Pass 4: integrate extracted answer into final decision.
        integration_prompt = self._integration_template.replace(
            "{case_text}", case_text
        ).replace(
            "{candidate_block}", format_ranked_candidates_for_prompt(top_candidates)
        ).replace(
            "{question}", question_result.question
        ).replace(
            "{answer_block}", format_evidence_for_prompt(answer_result)
        )

        try:
            integrated_output, raw = self._call_api(
                model=self.config.generator_model,
                prompt=integration_prompt,
                parse_fn=parse_integrated_decision_free_text,
            )
            response = integrated_output.response
            decision = integrated_output.decision

            response.final_diagnosis = decision.final_choice
            if decision.integration_summary:
                base_reasoning = response.conditional_reasoning or ""
                response.conditional_reasoning = (
                    f"Integration: {decision.integration_summary}. {base_reasoning}"
                ).strip()

            self._case_telemetry["integration_summary"] = decision.integration_summary
            self._case_telemetry["integration_rationale"] = decision.rationale
            self._case_telemetry["final_selection_source"] = "discriminative_integration"
            return response, raw
        except Exception as exc:
            self._case_telemetry["integration_error"] = str(exc)
            self._case_telemetry["final_selection_source"] = "baseline_fallback_integration_error"
            return super().generate(case_text)

    def _get_case_variant_metadata(self) -> Dict[str, object]:
        return dict(self._case_telemetry)

    @staticmethod
    def _load_prompt(name: str) -> str:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts" / "discriminative_question"
        path = prompts_dir / f"{name}.md"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
