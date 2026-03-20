# Counter-Hypothesis Generation

You are auditing a ranked diagnosis by proposing plausible alternatives the original ranking may have missed.

## Current ranked seed candidates
{seed_candidate_block}

## Seed candidate to challenge
{seed_candidate}

## Rules
- Return STRICT JSON only.
- Do not use markdown.
- Propose 2 plausible alternative diagnoses that are not the same as the seed candidate.
- Alternatives may overlap with other seed candidates if clinically justified, but prefer genuinely new diagnoses.
- Ground every alternative in the provided case and do not invent facts.

Use exactly this schema:

```json
{
  "counter_hypotheses": [
    {"label": "Alternative diagnosis 1", "rationale": "why this remains plausible"},
    {"label": "Alternative diagnosis 2", "rationale": "why this remains plausible"}
  ]
}
```

## Case
{case_text}
