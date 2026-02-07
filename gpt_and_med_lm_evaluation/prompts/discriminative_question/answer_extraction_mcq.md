# Answer Extraction from Case Text (MCQ)

Answer the discriminative question by re-scanning the case text only.

## Question
{question}

## Target variable
{target_variable}

## Rules
- Return STRICT JSON only.
- If evidence is absent, answer as "unclear".
- Include evidence snippets from the case.

Use exactly this schema:

```json
{
  "answer": "yes/no/unclear or short value",
  "confidence": 0.74,
  "evidence_spans": ["snippet 1", "snippet 2"],
  "rationale": "how evidence supports answer"
}
```

## Case
{case_text}
