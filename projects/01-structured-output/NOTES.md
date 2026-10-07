# NOTES — 01 structured-output

> Raw material for the blog — candid about what broke and how it was fixed.

## Goal
Make the classify step return strict JSON — `{"intent": "<label>",
"confidence": <0-1>}` — using the API's schema-constrained output instead of a
bare label + a fuzzy parser (project 00). Measure **valid-JSON rate** (the
production-parseability metric) with accuracy as a secondary note.

## Steps taken
1. Probed three `response_format` dialects on one model to see what Fireworks
   actually enforces (didn't want to guess the dialect):
   - `{"type":"json_object","schema":SCHEMA}` (Fireworks-native) → **enforced** ✅
   - `{"type":"json_schema","json_schema":{...}}` (OpenAI-style) → **enforced** ✅
   - `{"type":"json_object"}` (no schema) → **ignored the enum**: invented
     `"refund_request"` and dropped `confidence`. ❌
   Picked the OpenAI-style `json_schema` (matches the repo's OpenAI-compat rule).
2. Ran all 12 tickets × 5 models with the schema enforced. Validate raw `content`
   against the schema (parse + enum + confidence in [0,1]) with **no** repair /
   fence-stripping — that's the honest valid-JSON rate.
3. Probed the `confidence` field for calibration and spot-checked the one model
   that misbehaves (glm-5p1).

## Results (one official run)
| model | valid-JSON | accuracy |
|---|---|---|
| deepseek-v4-pro | 100% | 100% (12/12) |
| kimi-k2p5 | 100% | 91.7% |
| kimi-k2p6 | 100% | 91.7% |
| glm-5p1 | 100% \* | 91.7% |
| gpt-oss-120b | 100% | 91.7% |

\* 100% *this* run; see the glm caveat below — reruns drop to ~92%.

## Challenges
- **`json_object` without a schema is a trap.** It produces *syntactically* valid
  JSON that's *semantically* wrong — a label outside your enum, a missing field.
  Valid-JSON rate would look perfect while every downstream `intent in LABELS`
  check fails. You need the schema, not just "give me JSON."
- **One model doesn't honor the schema cleanly: glm-5p1.** For deepseek / kimi /
  gpt-oss, the chain-of-thought goes to a separate `reasoning_content` field and
  `content` comes back as clean enforced JSON. **glm-5p1 doesn't separate its
  reasoning** — under `json_schema` it sometimes pours the entire CoT into
  `content` ("Let's go with account... actually technical... let's compromise...")
  and burns the whole 1024-token budget before ever emitting JSON. Result:
  non-conforming output. This is exactly the README's "honors schemas cleanly vs.
  needs coaxing" split: **4 of 5 clean, glm the exception.**
- **Nondeterminism strikes again (cf. project 00).** glm passed all 12 in the
  official run but failed t03 on a rerun — so its true valid-JSON rate is ~92% and
  wobbles, not a hard 100%. The headline number is one sample; the row note says so.

## Bugs & fixes
| symptom | cause | fix |
|---------|-------|-----|
| naive probe crashed: `JSONDecodeError: Extra data` on glm-5p1 | glm leaked CoT into `content` instead of JSON; `json.loads` choked | `run.py`'s `validate()` wraps parsing in try/except and counts malformed output as *invalid* rather than crashing — which is the whole reason you measure a valid-JSON *rate* |
| would-be false 100% if I'd used bare `json_object` | no schema → model invents labels | use `json_schema` with an `enum` on `intent` and `additionalProperties:false` |

## Cost
- Same reasoning-token reality as project 00: each call still costs a few hundred
  reasoning tokens; the JSON answer itself is ~10 tokens. Schema constraint does
  **not** make reasoning models stop thinking — it only shapes the final `content`.
- glm's failure mode is also a *cost* failure: it spends the full `max_tokens`
  reasoning and returns nothing usable — you pay for 1024 tokens of indecision.
- Wall clock ~4.5 min for 60 calls. Pricing TODO still tracked for project 08.

## Learnings (2-3)
- **Schema-constrained output ≠ guaranteed parseable across all models.** Four
  models give a hard 100%; one (glm-5p1) intermittently breaks the contract by
  leaking reasoning into `content`. Always validate the raw output — don't assume
  `response_format` is a guarantee.
- **Ask for a schema, not just "JSON."** Bare `json_object` yields well-formed but
  off-spec data (`"refund_request"`, missing fields) that silently fails
  downstream. The enum is what makes it useful.
- **The `confidence` field is a real, mildly-calibrated signal — the payoff over
  project 00.** Easy tickets land at 0.95–1.0; the genuinely ambiguous t11 drops
  to 0.8–0.9 (deepseek 0.9 & correct, gpt-oss 0.8 & wrong). That threshold is a
  ready-made route-to-human / escalate trigger you simply can't get from a bare
  label.
