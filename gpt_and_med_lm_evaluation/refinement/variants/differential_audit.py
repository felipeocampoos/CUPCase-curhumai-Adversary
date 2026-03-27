"""Differential-audit variant with counter-hypothesis generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ..differential_audit import (
    format_pooled_differential_for_prompt,
    format_seed_candidates_for_prompt,
    merge_differential_pool,
    parse_comparative_evaluation_free_text,
    parse_counter_hypotheses,
)
from ..discriminative_questioning import parse_ranked_candidates
from ..refiner import IterativeRefiner


class DifferentialAuditRefiner(IterativeRefiner):
    """Variant that introduces counter-hypotheses before final selection."""

    variant_name: str = "differential_audit"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._candidate_template = self._load_prompt("candidate_free_text")
        self._counter_template = self._load_prompt("counter_hypotheses_free_text")
        self._comparative_template = self._load_prompt("comparative_evaluation_free_text")
        self._case_telemetry: Dict[str, object] = {}

    def refine(self, *args, **kwargs):
        self._case_telemetry = {}
        return super().refine(*args, **kwargs)

    def generate(self, case_text: str):
        candidate_prompt = self._candidate_template.replace("{case_text}", case_text)
        try:
            ranked, _ = self._call_api(
                model=self.config.generator_model,
                prompt=candidate_prompt,
                parse_fn=parse_ranked_candidates,
            )
        except Exception as exc:
            self._case_telemetry["candidate_error"] = str(exc)
            self._case_telemetry["final_selection_source"] = "baseline_fallback_candidate_error"
            return super().generate(case_text)

        seed_candidates = [candidate.label for candidate in ranked.candidates[:3]]
        self._case_telemetry["seed_candidates"] = seed_candidates

        counter_hypotheses_by_seed: Dict[str, List[str]] = {}
        for seed in seed_candidates:
            counter_prompt = self._counter_template.replace(
                "{case_text}", case_text
            ).replace(
                "{seed_candidate}", seed
            ).replace(
                "{seed_candidate_block}", format_seed_candidates_for_prompt(seed_candidates)
            )
            try:
                counter_result, _ = self._call_api(
                    model=self.config.generator_model,
                    prompt=counter_prompt,
                    parse_fn=parse_counter_hypotheses,
                )
                filtered = [
                    label for label in counter_result.hypotheses
                    if label.strip().casefold() != seed.strip().casefold()
                ]
                counter_hypotheses_by_seed[seed] = filtered[:2]
            except Exception as exc:
                self._case_telemetry["counter_hypothesis_error"] = {
                    "seed": seed,
                    "error": str(exc),
                }
                self._case_telemetry["final_selection_source"] = (
                    "baseline_fallback_counter_hypothesis_error"
                )
                return super().generate(case_text)

        pooled_differential = merge_differential_pool(
            seed_candidates=seed_candidates,
            counter_hypotheses_by_seed=counter_hypotheses_by_seed,
            max_total=9,
        )
        self._case_telemetry["counter_hypotheses_by_seed"] = counter_hypotheses_by_seed
        self._case_telemetry["pooled_differential"] = pooled_differential

        comparative_prompt = self._comparative_template.replace(
            "{case_text}", case_text
        ).replace(
            "{pooled_differential_block}",
            format_pooled_differential_for_prompt(
                seed_candidates=seed_candidates,
                counter_hypotheses_by_seed=counter_hypotheses_by_seed,
                pooled_differential=pooled_differential,
            ),
        )

        try:
            comparative_output, raw = self._call_api(
                model=self.config.generator_model,
                prompt=comparative_prompt,
                parse_fn=parse_comparative_evaluation_free_text,
            )
            response = comparative_output.response
            decision = comparative_output.decision

            if decision.final_choice:
                response.final_diagnosis = decision.final_choice

            if decision.rationale:
                base_reasoning = response.conditional_reasoning or ""
                response.conditional_reasoning = (
                    f"Comparative evaluation: {decision.rationale}. {base_reasoning}"
                ).strip()

            final_key = response.final_diagnosis.strip().casefold()
            self._case_telemetry["comparative_rationale"] = decision.rationale
            self._case_telemetry["evidence_for"] = decision.evidence_for
            self._case_telemetry["evidence_against"] = decision.evidence_against
            self._case_telemetry["missing_information"] = decision.missing_information
            self._case_telemetry["final_selection_source"] = (
                "comparative_counter_hypothesis"
                if final_key not in {label.casefold() for label in seed_candidates}
                else "comparative_seed_candidate"
            )
            return response, raw
        except Exception as exc:
            self._case_telemetry["comparative_error"] = str(exc)
            self._case_telemetry["final_selection_source"] = (
                "baseline_fallback_comparative_error"
            )
            return super().generate(case_text)

    def _get_case_variant_metadata(self) -> Dict[str, object]:
        return dict(self._case_telemetry)

    @staticmethod
    def _load_prompt(name: str) -> str:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts" / "differential_audit"
        path = prompts_dir / f"{name}.md"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
