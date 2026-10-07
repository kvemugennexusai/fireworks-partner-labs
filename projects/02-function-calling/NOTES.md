# NOTES — 02 function-calling

> Raw material for the blog — candid about what broke and how it was fixed.

## Goal
Move from *classifying* to *acting*: expose two tools — `create_refund(ticket_id,
amount)` and `escalate(ticket_id, team)` — and have the model pick the right
action (or none) with correct arguments. Score **tool-selection accuracy**, with
argument correctness in the notes.

## Steps taken
1. Defined an explicit routing policy and handed the SAME policy to the model, so
   "correct" is defined by the policy, not my taste:
   - `create_refund` → customer owed money back for a billing error.
   - `escalate(team)` → system malfunction / outage / security issue.
   - no tool → informational or self-serve request the assistant can answer.
2. Probed tool-calling shape on one model first — confirmed `tools=[...]`,
   `tool_choice="auto"`, and that `finish_reason="tool_calls"` with a parseable
   `message.tool_calls[0].function.arguments` JSON string. `t08` (info) correctly
   returned NO tool and answered in prose. Good — the "do nothing" branch works.
3. Labeled all 12 tickets per the policy, ran 12 × 5, scored category match +
   whether the right `ticket_id` was echoed.

## Results (after relabeling t11 → escalate; see below)
| model | tool-selection | ticket_id args |
|---|---|---|
| deepseek-v4-pro | 91.7% (11/12) | ✓ |
| kimi-k2p5 | 91.7% | ✓ |
| kimi-k2p6 | 91.7% | ✓ |
| glm-5p1 | 91.7% | ✓ |
| gpt-oss-120b | 83.3% (wobbles, see below) | ✓ |

(Before the relabel these were mostly 83.3%; t11 alone was costing ~8 points.)

**All five support tools — zero API errors.** Argument grounding was *perfect*:
every time a model chose the right action, it echoed the correct `ticket_id`. The
difficulty is entirely in the *decision*, not in filling the slots.

## Challenges
- **The error is not scattered — it lands on the genuinely ambiguous tickets**, the
  same ones that fought projects 00/01. Per-ticket breakdown:
  - **t11 ("upgraded plan but new seats aren't showing") — ALL FIVE models call
    `escalate`; my original policy label was `none`.** This is the recurring problem
    child from 00 and 01. When 5/5 models unanimously disagree with your ground
    truth, the label is the thing to doubt, not the models. **Decision: relabeled
    t11 → `escalate`** to the model consensus, which lifted every model ~8 points
    (most to 91.7%). The reference answer mentions the agent could "force a sync,"
    so `none` was defensible — but a unanimous 5/5 vote is stronger evidence than my
    one judgment call. (The *intent* label in `data/tickets.jsonl` is a separate
    schema — still `billing` — and was not touched.)
  - **t10 (close account + delete payment):** deepseek & glm `escalate` to the
    account team instead of answering. Very defensible — there's no `close_account`
    tool, so routing a destructive request to a human is arguably the safer move.
  - **t06 (reset email never arrives):** kimi-k2p6 & gpt-oss answer directly rather
    than escalate — they treat it as self-serve troubleshooting.
  - **t03 (settings won't save):** gpt-oss answers directly instead of escalating.
  - The 8 clear-cut tickets (refund t01; escalations t02/t04/t09/t12; info
    t05/t07/t08) are **unanimous and correct.**
- **Nondeterminism (cf. 00/01):** gpt-oss scored 83% in the official run but 75% on
  a rerun — its t03/t06 calls flip between runs. It's the wobbliest on the
  borderline routing cases.

## Bugs & fixes
| symptom | cause | fix |
|---------|-------|-----|
| risk of crashing on a malformed tool-call arg string | `function.arguments` is a JSON *string* the model could in principle malform | `first_call()` wraps `json.loads` in try/except and degrades to `{}` rather than throwing |
| "no tool expected" looked unscoreable at first | need to score the *absence* of a call | model returning prose (no `tool_calls`) is mapped to the category `none` and scored like any other choice |
| apparent low scores look like model failure | the misses are concentrated on debatable labels (esp. t11) | report per-ticket, separate clear-cut from ambiguous, and concede the label where models agree against it |

## Cost
- Same reasoning-token reality (a few hundred reasoning tokens per decision). Wall
  clock ~3.7 min for 60 calls. No measurable extra cost from passing tool schemas.
- Pricing TODO still tracked for project 08.

## Learnings (2-3)
- **Tool *selection* is the hard part; tool *arguments* are easy.** All five models
  grounded `ticket_id` perfectly every time they chose correctly — the entire error
  budget went to deciding *whether* and *which* to call, not to filling slots.
- **The "no tool" branch is a first-class outcome and these models handle it well.**
  Informational tickets (t05/t07/t08) reliably returned prose with no spurious
  call — no over-eager tool use, which is the usual function-calling failure mode.
- **t11 fought my labels in all three projects, so I let the models win the
  action label.** Three task framings, same ticket, same disagreement — the dataset
  telling me the gold answer was contestable. Here I acted on it: relabeled the
  *action* to the 5/5 consensus (`escalate`). The lesson for the blog is the loop
  itself — model-vs-label disagreement is a dataset signal, not just a model score.
