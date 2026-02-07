# Discriminator Reasoning (MCQ)

The top candidates are semantically close. Pick the best final option by explicitly comparing discriminating features.

## Top candidates
{candidate_block}

## Similarity diagnostics
{similarity_block}

## Rules
- Return STRICT JSON only.
- `final_choice_index` must be one of the ranked candidates and 1-based.

Use this schema exactly:

```json
{
  "final_choice_index": 3,
  "differentiators": [
    "Specific finding that favors option 3 over option 1",
    "Specific finding that favors option 3 over option 2"
  ],
  "rationale": "Short explanation using discriminators"
}
```

## Case
{case_text}
