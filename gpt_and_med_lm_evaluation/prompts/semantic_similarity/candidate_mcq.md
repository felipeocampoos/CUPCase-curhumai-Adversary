# Candidate Ranking (MCQ)

Given the case and answer options, rank the top 3 most likely option indices.

## Rules
- Return STRICT JSON only.
- Indices are 1-based and must refer to existing options.
- Provide exactly 3 unique ranked indices.

Use this schema exactly:

```json
{
  "ranked_indices": [1, 3, 2],
  "rationale": "Short rationale for ranking"
}
```

## Options
{options_text}

## Case
{case_text}
