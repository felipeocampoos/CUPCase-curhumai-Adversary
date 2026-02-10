# Answer Integration (MCQ)

Use the extracted answer to resolve top-candidate ambiguity and pick the final option.

## Ranked candidates
{candidate_block}

## Discriminative question
{question}

## Extracted answer and evidence
{answer_block}

## Rules
- Return STRICT JSON only.
- final_choice_index must be one of the provided options (1-based).

Use exactly this schema:

```json
{
  "final_choice_index": 2,
  "integration_summary": "answer -> impact on top candidates",
  "rationale": "brief rationale"
}
```

## Case
{case_text}
