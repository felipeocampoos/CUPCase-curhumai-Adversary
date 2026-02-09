"""Progressive disclosure variant with explicit belief revision telemetry."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..progressive_disclosure import (
    compute_belief_revision_scores,
    format_early_candidates_for_prompt,
    parse_early_differential_free_text,
    parse_revision_decision_free_text,
    truncate_case_by_fraction,
)
from ..refiner import IterativeRefiner


class ProgressiveDisclosureRefiner(IterativeRefiner):
    """Two-stage differential reasoning: early snippet then full-case revision."""

    variant_name: str = "progressive_disclosure"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._early_template = self._load_prompt("early_free_text")
        self._revision_template = self._load_prompt("revision_free_text")
        self._case_telemetry: Dict[str, object] = {}

    def refine(self, *args, **kwargs):
        self._case_telemetry = {}
        return super().refine(*args, **kwargs)

    def generate(self, case_text: str):
        fraction = self.config.disclosure_fraction
        early_case_text = truncate_case_by_fraction(case_text, fraction=fraction)

        self._case_telemetry["disclosure_fraction"] = fraction
        self._case_telemetry["early_case_token_proxy_count"] = len(early_case_text.split())

        early_prompt = self._early_template.replace("{early_case_text}", early_case_text)

        try:
            early_differential, _ = self._call_api(
                model=self.config.generator_model,
                prompt=early_prompt,
                parse_fn=parse_early_differential_free_text,
            )
        except Exception as exc:
            self._case_telemetry["early_stage_error"] = str(exc)
            self._case_telemetry["final_selection_source"] = "baseline_fallback_early_error"
            return super().generate(case_text)

        top3 = early_differential.candidates[:3]
        top1 = top3[0]

        self._case_telemetry["early_candidate_top3"] = [c.label for c in top3]
        self._case_telemetry["early_candidate_confidences"] = [c.confidence for c in top3]
        self._case_telemetry["early_candidate_rationale"] = [c.rationale for c in top3]

        revision_prompt = self._revision_template.replace(
            "{early_candidate_block}", format_early_candidates_for_prompt(top3)
        ).replace(
            "{early_case_text}", early_case_text
        ).replace(
            "{case_text}", case_text
        )

        try:
            revision_decision, raw = self._call_api(
                model=self.config.generator_model,
                prompt=revision_prompt,
                parse_fn=parse_revision_decision_free_text,
            )
        except Exception as exc:
            self._case_telemetry["revision_stage_error"] = str(exc)
            self._case_telemetry["final_selection_source"] = "baseline_fallback_revision_error"
            return super().generate(case_text)

        scores = compute_belief_revision_scores(
            early_top_label=top1.label,
            final_label=revision_decision.final_choice,
            early_top_confidence=top1.confidence,
            final_confidence=revision_decision.final_confidence,
            contradiction_found=revision_decision.contradiction_found,
            kept_labels=revision_decision.kept_hypotheses,
        )
        instability = scores.confidence_instability_score
        if instability < self.config.revision_instability_threshold:
            instability = 0.0
        if top1.confidence < self.config.early_confidence_threshold:
            instability = 0.0
        unexplained_revision = 1.0 if (scores.revision_delta >= 1.0 and not revision_decision.contradiction_found) else 0.0
        penalty = (0.5 * float(scores.anchoring_flag)) + (0.3 * instability) + (0.2 * unexplained_revision)
        penalty = max(0.0, min(1.0, penalty))

        self._case_telemetry["final_candidate"] = revision_decision.final_choice
        self._case_telemetry["final_confidence"] = revision_decision.final_confidence
        self._case_telemetry["revision_summary"] = revision_decision.revision_summary
        self._case_telemetry["revision_kept"] = revision_decision.kept_hypotheses
        self._case_telemetry["revision_dropped"] = revision_decision.dropped_hypotheses
        self._case_telemetry["revision_added"] = revision_decision.added_hypotheses
        self._case_telemetry["contradiction_found"] = revision_decision.contradiction_found
        self._case_telemetry["revision_rationale"] = revision_decision.rationale
        self._case_telemetry["anchoring_flag"] = scores.anchoring_flag
        self._case_telemetry["confidence_instability_score"] = instability
        self._case_telemetry["revision_delta"] = scores.revision_delta
        self._case_telemetry["belief_penalty_score"] = penalty
        self._case_telemetry["final_selection_source"] = "full_case_revision"

        return revision_decision.response, raw

    def _get_case_variant_metadata(self) -> Dict[str, object]:
        return dict(self._case_telemetry)

    @staticmethod
    def _load_prompt(name: str) -> str:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts" / "progressive_disclosure"
        path = prompts_dir / f"{name}.md"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
