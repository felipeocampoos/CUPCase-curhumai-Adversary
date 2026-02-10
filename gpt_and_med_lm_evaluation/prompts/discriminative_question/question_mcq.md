# Discriminative Question Generation (MCQ)

Generate exactly one question that best discriminates between the top 2 candidate options.

## Top candidates
{candidate_block}

## Rules
- Return STRICT JSON only.
- Produce exactly one concrete, clinically actionable question.
- Target a specific clinical variable.

Use exactly this schema:

```json
{
  "question": "single discriminative question",
  "target_variable": "concrete variable",
  "rationale": "why this variable best separates top two options"
}
```

## Case
{case_text}
