# Discriminative Question Generation (Free Text)

You are deciding between the top two diagnoses.
Generate exactly one question that best discriminates between them.

## Top candidates
{candidate_block}

## Rules
- Return STRICT JSON only.
- Do not use markdown.
- Produce exactly one concrete, clinically actionable question.
- The question must target a specific variable (e.g., orthopnea, complement level, rash morphology).

Use exactly this schema:

```json
{
  "question": "Single discriminative question",
  "target_variable": "Concrete clinical variable",
  "rationale": "Why this variable best separates candidate 1 and 2"
}
```

## Case
{case_text}
