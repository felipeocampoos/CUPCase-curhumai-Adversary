# Early Differential (20% Case)

You are seeing only the opening portion of a case. Build a preliminary differential from this early information.

## Rules
- Return STRICT JSON only.
- Do not use markdown.
- Use only the early case text provided.
- Provide at least 2 and up to 3 candidates.
- `confidence` must be numeric in [0,1].

Use exactly this schema:

```json
{
  "candidates": [
    {"label": "Diagnosis 1", "confidence": 0.62, "rationale": "short reason from early evidence"},
    {"label": "Diagnosis 2", "confidence": 0.28, "rationale": "short reason from early evidence"},
    {"label": "Diagnosis 3", "confidence": 0.10, "rationale": "short reason from early evidence"}
  ]
}
```

## Early case text
{early_case_text}
