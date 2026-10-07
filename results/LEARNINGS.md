# Learnings — building on Fireworks

Cross-cutting notes pulled from each project's NOTES.md. This is the spine of the blog.
One use case (a Support Copilot) driven through 10 capabilities on a public, synthetic
dataset of 12 tickets, with the bugs left in. All numbers are reproducible from the repo;
`$/1M` figures are **estimates** (see Cost).

## What was easy
- **The OpenAI-compatible surface is real.** One `openai` client, `base_url` swap, done.
  The same `shared/eval.py` and `shared/bench.py` pointed at any model id by changing one
  string. First call in ~1 minute.
- **Structured output just works.** A JSON schema with an `enum` gave 100% valid, parseable
  output on 4 of 5 models — the fuzzy label-parser from the baseline project became dead
  code (project 01).
- **Function calling worked on every catalog model**, zero API errors, and argument
  grounding was *perfect* — every correct tool call had the right `ticket_id`. The hard
  part was deciding *which* tool, never filling the slots (project 02).
- **Embeddings + retrieval was the most reliable component in the whole portfolio** —
  100% retrieval@1, including the ticket every other project found ambiguous (project 03).
- **Fine-tuning, once you find the right tool, is fast and cheap to *train*** — an SFT LoRA
  job ran in ~4 minutes (project 04).
- **Multi-LoRA serving is trivial to operate** — load N adapters onto one base, "route" by
  picking the adapter's model id. Two tenant tone-adapters served distinct house styles
  from the same GPU and prompt (project 07).

## What fought back
- **`temperature=0` is NOT deterministic** on these MoE reasoning models. It showed up in
  *every* project: per-model accuracy wobbled ±1 ticket run-to-run, glm's valid-JSON rate
  bounced 92↔100%, gpt-oss's tool-selection moved 75↔83%. Report tolerances, average runs,
  and gate on aggregates — project 06 confirmed the *aggregate* judge mean is stable to
  ±0.039 even while single-item scores jump.
- **"In the catalog" ≠ "serverless."** Base `llama-v3p1-8b-instruct` (04/05), and every
  vision model (09), 404 on the inference endpoint — they only exist as on-demand
  deployments. Always probe the *specific* id before assuming you can call it.
- **Reasoning models hide a `max_tokens` trap.** Visible answers live in `content` while
  the chain-of-thought burns the *same* token budget in a separate `reasoning_content`
  field. `max_tokens=8` (or `eval.py`'s default 64) silently zeroes them — the baseline
  scored 8.3% for *every* model until we gave reasoning room. The tell is all models tying
  at the floor (project 00; recurred in the judge in 03/06).
- **Fine-tuning is mostly ops, not ML.** Project 04: of 8 things that broke, exactly one
  was about learning. The other seven were namespaces (fully-qualified base ids), an
  interactive deploy prompt, "no available capacity," the `#<deployment>` addressing
  syntax, addon loading, and teardown. The loss curve was the easy part.
- **The default token batch size silently undertrains.** SFT v1 packed all 96 examples into
  one 32,768-token batch, so "3 epochs" was **3 optimizer steps** — the model collapsed to
  one label and scored *worse* than base (16.7% vs 66.7%). `--batch-size-samples` fixed it
  (project 04). Count your steps, not your epochs.
- **Tooling lags the server.** Creating an RFT *evaluator* failed through reward-kit (501
  deprecated), the fireworks SDK (same deprecated gRPC), firectl (no verb), and the UI
  ("must use the `@evaluation_test` decorator"). The working path was a *different*
  framework — **eval-protocol** (`ep upload` / `ep create rft`). If you hit
  `CreateEvaluator is deprecated`, stop fighting reward-kit and switch (project 05).
- **RFT jobs silently PAUSE** mid-run and need a manual `firectl resume` (project 05).
- **Forced RAG can make answers worse.** Grounding helped +2 where policy was specific but
  cost −4 on a question the KB didn't cover (support hours) — strict "use only the policy"
  made the model deflect on something it knew. A single mean (+0.25) hid a bimodal effect
  (project 03).
- **The dataset, not the model, was often the problem.** One ticket (t11: "upgraded plan,
  seats not showing") fought *every* project; in function-calling all 5 models unanimously
  disagreed with our label, so we relabeled it. When the models agree against your gold,
  doubt the gold.

## Model-by-model character
| model | strengths | quirks |
|-------|-----------|--------|
| **gpt-oss-120b** | top accuracy (100% baseline), **fastest** (TTFT 1.4s, 828 tok/s), cheapest, clean structured output, reasoning isolated to `reasoning_content` | streams token-by-token (good for UIs); the obvious default |
| deepseek-v4-pro | strong & steady accuracy (often best on judge/SFT), clean JSON, incremental streaming | mid TTFT (~3.9s) |
| kimi-k2p6 | solid accuracy, fine structured output | wide p95 latency tail (2.7s→14s) |
| kimi-k2p5 | comparable accuracy | **slowest**, cold-start prone (29s TTFT cold, 3.2s warm) |
| glm-5p1 | fine on plain accuracy | **leaks chain-of-thought into `content`** under schema constraint and runs out of tokens before emitting JSON — the only model that broke the structured-output contract |

## Cost picture
- **`$/1M` is ESTIMATED.** `firectl billing get-usage` reported **$0.00** (trial credits),
  so real spend couldn't be measured; per-token rates are from Fireworks' published
  serverless tiers and marked EST throughout. Latency/throughput/accuracy are all measured.
  - Serverless out-token (est): **gpt-oss-120b ~$0.67/1M**, deepseek ~$0.99, glm ~$2.25,
    kimi ~$2.57. gpt-oss is both the best and the cheapest here.
- **Fine-tuning is cheap to train, but the deployment is the bill.** SFT ~4–5 min, RFT
  ~35 min (incl. a 14-min pause). The real cost is the **on-demand GPU** you need to *serve*
  a fine-tune: 1× H200 ≈ **$8.99/hr (EST)** ≈ ~$6.5k/mo always-on. We tore every deployment
  down immediately; an idle one billed ~25 min during the RFT pause for nothing.
- **Multi-LoRA is the cost story.** Amortizing one base across N tenants is **90% cheaper at
  10 tenants, 99% at 100** vs dedicated deployments (project 07). A LoRA adapter is a few MB,
  so you rent one GPU, not N.
- **Self-host vs Fireworks: not benchmarked here.** `shared/bench.py` is built for it
  (point `--base-url` at a local vLLM and run the identical command), but this portfolio had
  no local GPU, so there is no self-host number — an honest gap, not a result.

## The thesis, tested
**Does "prototype + fine-tune + deploy in one place" hold up?** Mostly yes, with caveats a
partner should know:

- **Yes for the inference path.** Serverless + structured output + tools + RAG + a vision
  model is a smooth, OpenAI-compatible day. You can build the whole Support Copilot against
  serverless and never touch infra.
- **Yes for fine-tuning *results*.** SFT lifted a small 8B +16.6 pts; RFT then beat SFT
  +16.7 to a perfect (if tiny-sample) score, fixing the exact boundary cases SFT missed. The
  platform genuinely closes the prototype→trained-model loop.
- **But budget for the ops and the tooling skew.** Fine-tuning and on-demand serving are
  where you'll spend your debugging hours: model-id namespaces, deployment shapes, capacity,
  addon loading, scale-to-zero cold starts, a deprecated evaluator API, and PAUSED jobs. The
  ML was never the blocker; the lifecycle was.
- **Watch the bill and the determinism.** GPU $/hr (not tokens) dominates once you self-serve
  a fine-tune — amortize with multi-LoRA. And design every eval around aggregates, because
  temp=0 won't save you from run-to-run noise.

Bottom line: a strong bet for teams that want one platform from prototype to fine-tuned,
multi-tenant deployment — provided they treat the deployment lifecycle as real engineering
and keep a close eye on idle GPU spend.
