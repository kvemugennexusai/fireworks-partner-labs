# NOTES — 00 serverless-baseline

> Raw material for the blog — candid about what broke and how it was fixed.

## Goal
Establish the serverless baseline: classify each support ticket's intent
(billing / technical / account / other) on **every** text model in the live
Fireworks catalog, score accuracy, and write `results.json`. This is the reference
pattern every later project copies.

## Steps taken
1. Verified the live catalog with `GET /inference/v1/models` (via
   `shared.fw_client.list_models`). All five expected text ids were present:
   `deepseek-v4-pro`, `kimi-k2p5`, `kimi-k2p6`, `glm-5p1`, `gpt-oss-120b`
   (plus `flux-1-schnell-fp8` for the vision project later). Nothing missing.
2. Ran the classifier over all 12 tickets × 5 models. First run scored **8.3%
   (1/12) for every model** — a dead giveaway of a parsing/harness bug, not bad
   models (see below).
3. Fixed the token budget, re-ran. Accuracy jumped to the 83–100% range.
4. Captured per-ticket predictions to find *which* tickets are hard.

## Challenges
- **Every catalog model is a reasoning model.** They emit chain-of-thought in a
  separate `reasoning_content` field and put the visible answer in `content`.
  With a tiny `max_tokens`, the model never finishes thinking, so `content` comes
  back as a truncated reasoning fragment (`'We are asked: "Classify the'`) or
  empty (`gpt-oss-120b`). The "Reply with ONLY the label" instruction does not
  stop them from reasoning first.
- **`temperature=0` is NOT deterministic here.** Two runs of the identical code
  gave different scores (e.g. `gpt-oss-120b` 12/12 on one run, 11/12 on the next;
  `kimi-k2p5` 10/12 then 11/12). MoE routing / serverless batching makes the
  reasoning trace—and occasionally the final label—wobble run-to-run. Reproducible
  *code* does not mean reproducible *numbers* with these models. Treat single-run
  accuracy as ±1 ticket.
- **One genuinely ambiguous ticket carries all the disagreement.** 11 of 12
  tickets are unanimous and correct across all five models. The lone hard case is
  **t11** — *"I upgraded my plan but the new seats aren't showing for my team."*
  Gold = `billing` (plan upgrade), but the "not showing / sync" symptom reads
  `technical` and "team seats" reads `account`. Models split exactly along those
  lines (four said technical, gpt-oss said account). The gold label is itself
  debatable; this is a labeling problem, not a model problem.

## Bugs & fixes
| symptom | cause | fix |
|---------|-------|-----|
| All 5 models score 8.3% (1/12), identically | `max_tokens=8` truncated reasoning models mid-thought; `content` empty/garbage | Raised to `MAX_TOKENS=1024` (one identical budget across models for fairness) so reasoning finishes and `content` holds the bare label |
| `content` could still contain stray reasoning text | bare-substring match returned the first label in list order, not the model's conclusion | `pick_label` now scans the last non-empty line first and prefers the label appearing *latest* in the text |
| `DeprecationWarning: datetime.utcnow()` | deprecated in 3.14 | switched to `datetime.now(timezone.utc)` (also drops the manual `+ "Z"`) |

## Cost
- **Driver is reasoning tokens, not the answer.** Each one-word classification
  burns ~270–340 reasoning tokens (measured via `reasoning_content` length) plus
  ~1 content token. The output label is free; the thinking is the bill. Budgeting
  `max_tokens` for output alone would be wrong — it has to cover the hidden trace.
- **Wall clock:** ~4 min for 60 serverless calls (~4 s/call) at sequential,
  cold-ish serverless. Throughput/latency get measured properly in project 08.
- **$/1M tokens:** not yet pulled authoritatively for this specific catalog —
  these ids (`deepseek-v4-pro`, `kimi-k2p*`, `glm-5p1`) are a newer set and I did
  not want to fabricate per-model rates. **TODO:** record published serverless
  rates from the Fireworks pricing page in project 08, where token accounting is
  the focus. (Estimate flagged as estimate, per repo convention.)

## Learnings (2-3)
- **Reasoning models need a real token budget even for one-word answers** — the
  classic "tiny max_tokens for a label" optimization silently zeroes them out.
  The all-models-tie-at-the-floor pattern is the tell.
- **`temperature=0` ≠ deterministic** on serverless MoE reasoning models. Report
  accuracy with a tolerance, and where it matters, average several runs rather
  than trusting one.
- **When models disagree, suspect the label before the model.** Here the entire
  spread came from one ticket whose "correct" answer is genuinely contestable —
  a cue to fix the dataset, not chase the model.
