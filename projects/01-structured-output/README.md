# 01 — Structured output
**Capability:** strict JSON / schema-constrained output.
**Build:** make the classify step return `{"intent": "<label>", "confidence": <0-1>}`
using the API's structured-output / `response_format` support. Measure **valid-JSON rate**
and accuracy across models.
**Definition of done:** `results.json` (metric: valid-JSON rate %, plus accuracy in notes);
`NOTES.md` noting which models honor schemas cleanly vs. need prompt coaxing.
