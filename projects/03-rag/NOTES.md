# NOTES — 03 rag

> Raw material for the blog — candid about what broke and how it was fixed.

## Goal
Ground the drafted response in policy. Chunk `data/policies.md`, embed with a
Fireworks embeddings model, retrieve the top-k policy snippet per ticket, draft a
response, and measure whether grounding actually helps — mean LLM-judge score
(1-5), **grounded vs ungrounded**, plus retrieval@1.

## Steps taken
1. Found a usable embeddings model. The chat catalog (`models.list()`) returns
   **no** embeddings model — they live in a different id namespace. Probed the
   `/embeddings` endpoint directly; three work: `nomic-ai/nomic-embed-text-v1.5`
   (768-d), `thenlper/gte-large` (1024-d), `WhereIsAI/UAE-Large-V1` (1024-d). Used
   nomic. Note the id is the **bare HF-style** id, not `accounts/fireworks/...`.
2. Chunked policies on `## ` headings → 8 sections. Embedded chunks + tickets,
   cosine top-2.
3. Hand-labeled the relevant section per ticket (and marked **t08/t09 as having
   none** — the KB genuinely doesn't cover "support hours" or "latency spike").
4. Drafted each ticket two ways (ungrounded = ticket only; grounded = ticket +
   retrieved policy) and judged both with `shared/eval.py --task judge`, using
   **gpt-oss-120b as judge while deepseek-v4-pro drafts** — different models, so no
   self-grading.

## Results (one official run, judge = gpt-oss-120b)
- **retrieval@1 = 100%** (10/10 tickets that have a relevant section).
- ungrounded mean **3.58** → grounded mean **3.83** (+0.25).
- **But the average lies — the effect is bimodal** (per-ticket delta):
  - Helped (policy has specifics): t01 **+2** (refund "5-7 business days"), t12
    **+2** (SAML re-upload remedy), t04/t05/t06/t10/t11 **+1** each. **t11 — the
    ticket that fought every prior project — improves**, because the policy literally
    says "new seats can take up to 10 minutes to propagate."
  - Hurt: **t08 −4** (5→1), t09 −1, t02 −1.

## Challenges
- **Forced retrieval poisons out-of-KB questions — the headline finding.** t08
  ("what are your support hours / where are you HQ'd?") has no policy. Top-k still
  injected irrelevant snippets, and because I told the grounded prompt to ground
  *only* in policy ("if it doesn't cover the ticket, say you'll follow up"), the
  model **deflected on a question it actually knew**:
  > "I don't have support hours or headquarters information in the policy I can
  > reference right now. I'll follow up with the team..."
  Judge gave it 1/5; the ungrounded model just answered and got 5/5. **Strict
  grounding suppressed the model's own helpful knowledge when retrieval missed.**
  This is the classic RAG footgun, reproduced in one ticket.
- **Retrieval was the easy part; knowing *when not to use it* is the hard part.**
  Embeddings nailed 10/10 relevant sections, including the subtle t11 (seats →
  proration section). The quality problem isn't bad retrieval — it's the absence of
  a "retrieval is irrelevant here, fall back" gate.

## Bugs & fixes
| symptom | cause | fix |
|---------|-------|-----|
| `models.list()` shows no embeddings model | embeddings live in a separate id namespace and aren't listed | probe `/embeddings` directly; use the bare HF-style id (`nomic-ai/nomic-embed-text-v1.5`), not `accounts/fireworks/...` |
| judge would truncate to empty on reasoning models | `eval.py` hardcoded `max_tokens=200` for the judge; reasoning+content share that budget (cf. project 00) | added `--judge-max-tokens` (default 1024) to `shared/eval.py` and pass it through — benefits project 06 too |
| grounded answer worse than ungrounded on t08/t09 | top-k retrieval is unconditional; irrelevant context + strict-grounding prompt = deflection | **known limitation, not yet fixed:** needs a similarity threshold to bypass RAG when the best chunk is below it (documented as the next step) |

## Cost
- Embeddings are cheap and fast: 8 chunks + 12 tickets embedded in well under a
  second, one batched call each. The cost is the drafting + judging (reasoning
  models), not the retrieval.
- Whole pipeline ~47s wall clock (concurrency 4 in eval.py) — much faster than the
  sequential classify projects. Pricing TODO still tracked for project 08.

## Learnings (2-3)
- **RAG is not free quality — it's conditional quality.** Net +0.25 hides big wins
  where the KB is relevant (t01/t12 +2) and a catastrophic −4 where it isn't (t08).
  Always measure grounded *vs* ungrounded per-item; a single mean can hide that your
  RAG is hurting a whole class of queries.
- **The grounding prompt is a safety/utility tradeoff.** "Use ONLY the policy" stops
  hallucination but also gags the model on things it legitimately knows. You need a
  relevance gate, not a blanket rule.
- **Embeddings retrieval was the most reliable component in the whole portfolio so
  far** — 100% retrieval@1 including the perennially-ambiguous t11. The fragile part
  of RAG here was the generation policy around retrieval, not the vectors.
