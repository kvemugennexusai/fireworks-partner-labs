# 02 — Function calling
**Capability:** tool / function calling.
**Build:** expose two tools — `create_refund(ticket_id, amount)` and
`escalate(ticket_id, team)` — and have the model pick the right one with correct args
for tickets that warrant an action (e.g. t01 -> refund, t12 -> escalate). Score
tool-selection + argument correctness.
**Definition of done:** `results.json` (tool-selection accuracy per model); `NOTES.md`
on schema differences and any models that don't support tools.
