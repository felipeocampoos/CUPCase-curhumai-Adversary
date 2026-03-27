# Candidate Ranking (Differential Audit)

You are an expert diagnostic ranker. Rank the top 3 likely diagnoses for this case.

## Rules
- Return STRICT JSON only.
- Do not use markdown.
- Ground rationale only in the provided case.

Use exactly this schema:

```json
{
  "candidates": [
    {"label": "Diagnosis 1", "confidence": 0.61, "rationale": "short reason"},
    {"label": "Diagnosis 2", "confidence": 0.28, "rationale": "short reason"},
    {"label": "Diagnosis 3", "confidence": 0.11, "rationale": "short reason"}
  ]
}
```

## Case
{case_text}
