# Early Differential (MCQ, 20% Case)

Using only the early case text, rank the top 3 option indices.

## Rules
- Return STRICT JSON only.
- Indices are 1-based.
- Provide at least 2 and up to 3 unique ranked indices.
- `confidences` should align with ranked indices.

Use exactly this schema:

```json
{
  "ranked_indices": [2, 1, 4],
  "confidences": [0.56, 0.31, 0.13],
  "rationale": "brief reason from early evidence"
}
```

## Options
{options_text}

## Early case text
{early_case_text}
