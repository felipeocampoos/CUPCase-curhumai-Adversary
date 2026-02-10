# Candidate Generation (Free Text)

You are an expert diagnostic ranker.
Given a clinical case, return the top 3 most likely diagnoses in ranked order.

## Output rules
- Return STRICT JSON only.
- Do not include markdown.
- Use this schema exactly:

```json
{
  "candidates": [
    {"label": "Diagnosis 1", "confidence": 0.71, "evidence": "short reason"},
    {"label": "Diagnosis 2", "confidence": 0.19, "evidence": "short reason"},
    {"label": "Diagnosis 3", "confidence": 0.10, "evidence": "short reason"}
  ]
}
```

- `confidence` must be numeric between 0 and 1.
- `evidence` must be concise and grounded in case facts.

## Case

{case_text}
