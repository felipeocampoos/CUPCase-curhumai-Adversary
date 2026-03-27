# Comparative Differential Evaluation

You are an expert medical auditor. Compare all diagnoses head-to-head and select the single best-supported final diagnosis.

## Differential pool
{pooled_differential_block}

## Rules
- Return STRICT JSON only.
- Do not use markdown.
- Evaluate all diagnoses in the pooled differential.
- The final diagnosis is NOT limited to the original ranked seed candidates.
- Do not invent case facts.
- Use concise evidence lists grounded in the case.

Use this schema exactly:

```json
{
  "response": {
    "final_diagnosis": "Best-supported diagnosis",
    "differential": ["Alternative 1", "Alternative 2"],
    "conditional_reasoning": "Comparative explanation of why the winner beats the alternatives",
    "clarifying_questions": ["Optional unresolved question"],
    "red_flags": ["Urgent concern 1"],
    "uncertainty": "Confidence statement",
    "next_steps": ["Action 1", "Action 2"]
  },
  "final_choice": "Best-supported diagnosis",
  "evidence_for": {
    "Diagnosis A": ["Finding 1", "Finding 2"],
    "Diagnosis B": ["Finding 3"]
  },
  "evidence_against": {
    "Diagnosis A": ["Conflicting finding 1"],
    "Diagnosis B": ["Conflicting finding 2"]
  },
  "missing_information": {
    "Diagnosis A": ["Needed discriminator 1"],
    "Diagnosis B": ["Needed discriminator 2"]
  },
  "rationale": "Short head-to-head summary of why the final choice wins"
}
```

## Case
{case_text}
