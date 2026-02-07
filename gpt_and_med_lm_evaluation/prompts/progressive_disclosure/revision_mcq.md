# Full-Case Belief Revision (MCQ)

Revise the early ranking using the full case.

## Early ranked options
{early_ranked_block}

## Early case text (20%)
{early_case_text}

## Full case text
{case_text}

## Rules
- Return STRICT JSON only.
- `final_choice_index` must be 1-based and valid.
- Explicitly show how beliefs changed.

Use exactly this schema:

```json
{
  "final_choice_index": 3,
  "final_confidence": 0.72,
  "revision_summary": "One sentence describing why final choice changed or stayed",
  "kept_indices": [2],
  "dropped_indices": [1],
  "added_indices": [3],
  "contradiction_found": true,
  "rationale": "Brief rationale"
}
```
