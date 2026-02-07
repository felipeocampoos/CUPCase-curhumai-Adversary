# Candidate Ranking (MCQ)

Rank the top 3 most likely answer option indices for this case.

## Rules
- Return STRICT JSON only.
- Indices are 1-based.
- Provide 3 unique ranked indices.

Use exactly this schema:

```json
{
  "ranked_indices": [1, 3, 2],
  "rationale": "short reason"
}
```

## Options
{options_text}

## Case
{case_text}
