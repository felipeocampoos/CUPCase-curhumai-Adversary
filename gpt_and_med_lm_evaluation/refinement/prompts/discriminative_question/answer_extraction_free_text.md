# Answer Extraction from Case Text (Free Text)

Answer the discriminative question by re-scanning the case text only.
Do not invent information.

## Discriminative question
{question}

## Target variable
{target_variable}

## Rules
- Return STRICT JSON only.
- Do not use markdown.
- If evidence is absent, answer as "unclear" and explain briefly.
- Include direct evidence snippets from the case text.

Use exactly this schema:

```json
{
  "answer": "yes/no/unclear or short value",
  "confidence": 0.76,
  "evidence_spans": ["quoted or near-quoted snippet 1", "snippet 2"],
  "rationale": "How the evidence supports the extracted answer"
}
```

## Case
{case_text}
