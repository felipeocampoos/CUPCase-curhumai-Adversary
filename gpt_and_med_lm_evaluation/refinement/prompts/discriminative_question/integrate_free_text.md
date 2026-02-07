# Answer Integration (Free Text)

Use the extracted answer to the discriminative question to finalize diagnosis.

## Ranked candidates
{candidate_block}

## Discriminative question
{question}

## Extracted answer and evidence
{answer_block}

## Rules
- Return STRICT JSON only.
- Do not use markdown.
- The final diagnosis must reflect integration of the extracted answer.
- Include the answer impact in conditional_reasoning.

Use exactly this schema:

```json
{
  "response": {
    "final_diagnosis": "Most likely final diagnosis",
    "differential": ["Alternative 1", "Alternative 2"],
    "conditional_reasoning": "Explicitly explain how extracted answer changed or confirmed ranking",
    "clarifying_questions": ["Optional follow-up if truly unresolved"],
    "red_flags": ["Urgent red flag 1"],
    "uncertainty": "Confidence statement",
    "next_steps": ["Action 1", "Action 2"]
  },
  "final_choice": "Most likely final diagnosis",
  "integration_summary": "One sentence: answer -> impact on top2 discrimination",
  "rationale": "Brief rationale"
}
```

## Case
{case_text}
