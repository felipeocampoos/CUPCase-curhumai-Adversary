"""
Domain-routed prompt specialization variant.

Implements idea #5:
1. Predict likely specialty domain from case features.
2. Route generation to a domain-specific prompt template.
3. Keep critic/editor loop unchanged for fair comparison.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..refiner import IterativeRefiner
from ..schema import parse_diagnostic_response


@dataclass
class RouteDecision:
    """Domain routing decision for a single case."""

    domain: str
    scores: Dict[str, int]


class HeuristicDomainRouter:
    """Keyword-based specialty router for clinical cases."""

    GENERAL_DOMAIN = "general_medicine"

    # Priority order resolves ties deterministically.
    DOMAIN_PRIORITY = [
        "oncology",
        "infectious_disease",
        "neurology",
        "cardiology",
    ]

    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "oncology": [
            "cancer",
            "carcinoma",
            "metastatic",
            "metastasis",
            "neoplasm",
            "tumor",
            "malignancy",
            "oncology",
            "chemotherapy",
            "radiation",
            "immunotherapy",
            "biopsy",
            "lymphoma",
            "leukemia",
        ],
        "infectious_disease": [
            "infection",
            "infectious",
            "sepsis",
            "septic",
            "antibiotic",
            "viral",
            "bacterial",
            "fungal",
            "parasitic",
            "fever",
            "chills",
            "blood culture",
            "pcr",
            "hiv",
            "tb",
            "tuberculosis",
            "endocarditis",
            "meningitis",
            "pneumonia",
        ],
        "neurology": [
            "neurologic",
            "neurology",
            "seizure",
            "stroke",
            "tia",
            "aphasia",
            "hemiparesis",
            "weakness",
            "numbness",
            "ataxia",
            "migraine",
            "headache",
            "encephalopathy",
            "dementia",
            "parkinson",
            "multiple sclerosis",
            "mri brain",
            "csf",
        ],
        "cardiology": [
            "cardiac",
            "cardiology",
            "chest pain",
            "angina",
            "myocardial",
            "infarction",
            "heart failure",
            "arrhythmia",
            "ecg",
            "ekg",
            "troponin",
            "stemi",
            "nstemi",
            "tachycardia",
            "bradycardia",
            "syncope",
            "echocardiogram",
        ],
    }

    def route(self, case_text: str) -> RouteDecision:
        """Predict domain using keyword overlap scores."""
        lowered = case_text.lower()
        scores: Dict[str, int] = {}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in lowered)
            scores[domain] = score

        best_domain = self.GENERAL_DOMAIN
        best_score = 0

        for domain in self.DOMAIN_PRIORITY:
            score = scores.get(domain, 0)
            if score > best_score:
                best_domain = domain
                best_score = score

        return RouteDecision(domain=best_domain, scores=scores)


class DomainRoutedRefiner(IterativeRefiner):
    """Refiner variant that routes generation to domain-specific templates."""

    variant_name: str = "domain_routed"

    def __init__(self, *args, router: Optional[HeuristicDomainRouter] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = router or HeuristicDomainRouter()
        self._current_route: Optional[RouteDecision] = None
        self._domain_templates = self._load_domain_templates()

    def refine(self, *args, **kwargs):
        # Reset per-case router state before each run.
        self._current_route = None
        return super().refine(*args, **kwargs)

    def generate(self, case_text: str):
        route = self.router.route(case_text)
        self._current_route = route

        template = self._domain_templates.get(route.domain)
        if template is None:
            template = self._domain_templates[HeuristicDomainRouter.GENERAL_DOMAIN]

        prompt = template.replace("{case_text}", case_text)

        return self._call_api(
            model=self.config.generator_model,
            prompt=prompt,
            parse_fn=parse_diagnostic_response,
        )

    def _get_case_variant_metadata(self) -> Dict[str, object]:
        if self._current_route is None:
            return {}

        top_score = max(self._current_route.scores.values()) if self._current_route.scores else 0
        return {
            "predicted_domain": self._current_route.domain,
            "domain_scores": self._current_route.scores,
            "top_keyword_score": top_score,
        }

    @staticmethod
    def _load_domain_templates() -> Dict[str, str]:
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts" / "domain_routes"

        template_map = {
            "general_medicine": "generator_general_medicine.md",
            "oncology": "generator_oncology.md",
            "infectious_disease": "generator_infectious_disease.md",
            "neurology": "generator_neurology.md",
            "cardiology": "generator_cardiology.md",
        }

        templates: Dict[str, str] = {}
        for domain, filename in template_map.items():
            template_path = prompts_dir / filename
            with open(template_path, "r", encoding="utf-8") as f:
                templates[domain] = f.read()

        return templates
