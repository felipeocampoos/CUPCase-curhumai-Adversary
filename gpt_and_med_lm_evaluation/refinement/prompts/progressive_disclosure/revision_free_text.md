# Full-Case Belief Revision (Free Text)

You previously formed an early differential. Now revise using the full case.

## Early differential
{early_candidate_block}

## Early case text (20%)
{early_case_text}

## Full case text
{case_text}

## Rules
- Return STRICT JSON only.
- Do not use markdown.
- Explicitly describe what changed and why.
- If there is contradiction vs early top hypothesis, set `contradiction_found` true.

Use exactly this schema:

```json
{
  "response": {
    "final_diagnosis": "Most likely final diagnosis",
    "differential": ["Alternative 1", "Alternative 2"],
    "conditional_reasoning": "How full-case evidence changed or confirmed early differential",
    "clarifying_questions": ["Optional"],
    "red_flags": ["Urgent issue"],
    "uncertainty": "Confidence statement",
    "next_steps": ["Action 1", "Action 2"]
  },
  "final_choice": "Most likely final diagnosis",
  "final_confidence": 0.67,
  "revision_summary": "One sentence summary of belief revision",
  "kept_hypotheses": ["Diagnosis kept from early stage"],
  "dropped_hypotheses": ["Diagnosis dropped after full case"],
  "added_hypotheses": ["New diagnosis considered after full case"],
  "contradiction_found": true,
  "rationale": "Brief rationale"
}
```
