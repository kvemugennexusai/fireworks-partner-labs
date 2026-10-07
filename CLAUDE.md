# CLAUDE.md — build spec for Claude Code

You are building a portfolio of small, self-contained projects that exercise the
**Fireworks AI** platform end to end, around a single use case. Output is (1) a clean
public GitHub repo and (2) raw material for a blog post on what it's like to build on
Fireworks. Optimize for **reproducibility and honesty** over polish.

## The single use case (the through-line for every project)

**Support Copilot.** Given a customer-support ticket, the system:
1. **Classifies** the intent (one of: billing, technical, account, other).
2. **Retrieves** the relevant policy snippet (RAG).
3. **Drafts** a first response grounded in that policy.
4. Optionally **calls a tool** (e.g. `create_refund`, `escalate`) via function-calling.

Data lives in `data/` (`tickets.jsonl`, `policies.md`). It's synthetic and public so
anyone can clone and run. Do **not** add proprietary data.

## Models (catalog — verify before use)

Text: `accounts/fireworks/models/deepseek-v4-pro`, `kimi-k2p5`, `kimi-k2p6`,
`glm-5p1`, `gpt-oss-120b`. Vision/image: the `flux-*` family.
**Catalogs change** — at the start of any project, list what's actually available:
`GET /inference/v1/models` and use ids that come back. Note in NOTES.md if a model is gone.

## Hard conventions (apply to every project)

- **Endpoint:** Fireworks is OpenAI-compatible. Base URL `https://api.fireworks.ai/inference/v1`.
  Use the `openai` SDK. **Never hardcode the API key** — read `FIREWORKS_API_KEY` from env.
- **Shared code lives in `shared/`** — reuse it, don't re-implement:
  - `shared/fw_client.py` — client + model list + chat helper.
  - `shared/bench.py` — latency/throughput/cost benchmark (streaming).
  - `shared/eval.py` — accuracy + LLM-judge.
  - `shared/aggregate.py` — builds `results/RESULTS.md` from every project's `results.json`.
- **Each project writes `projects/<id>/results.json`** in this shape:
  ```json
  {"project":"00-serverless-baseline","metric":"accuracy","unit":"%",
   "rows":[{"model":"deepseek-v4-pro","value":91.2,"notes":"..."}],
   "run_at":"<iso8601>"}
  ```
- **Each project keeps `projects/<id>/NOTES.md`** filled in as you go: goal, steps taken,
  challenges, bugs + fixes, and 2-3 learnings. This is the blog's raw material — be candid
  about what broke and how you fixed it.
- **Cost:** record `$/1M tokens` from Fireworks' published pricing; mark estimates as estimates.
- **Determinism:** `temperature=0`, fixed prompts, identical `max_tokens` across models.
- Pin versions in `requirements.txt`. Keep each project runnable with one command
  documented in its README.

## Projects (build in order; each has its own README with a "Definition of done")

| #  | Project | Capability exercised |
|----|---------|----------------------|
| 00 | serverless-baseline | Serverless inference across all catalog models |
| 01 | structured-output   | JSON-schema / structured output for the classify step |
| 02 | function-calling    | Tool calls (`create_refund`, `escalate`) |
| 03 | rag                 | Embeddings + retrieval to ground the draft |
| 04 | finetune-sft        | FireOptimizer SFT on labeled tickets vs. base |
| 05 | finetune-rft        | RFT with reward-kit (reward = response rubric) vs. SFT |
| 06 | evals               | Eval Protocol harness: accuracy + LLM-judge (the shared gate) |
| 07 | multi-lora          | Several LoRA adapters on one base; per-tenant cost model |
| 08 | serving-perf        | Latency/throughput/cost; serverless vs on-demand, FP8/spec-decoding |
| 09 | vision              | A `flux`/vision model reads an error screenshot (stretch) |

BYOC / Enterprise-Reserved are deployment modes that need enterprise infra — document
them in `projects/08-serving-perf/README.md` as notes, don't build runnable code.

## Definition of done for the whole repo

- Every built project has: working code, a one-command run, `results.json`, filled `NOTES.md`.
- `python shared/aggregate.py` regenerates `results/RESULTS.md` (the master table).
- `results/LEARNINGS.md` has the cross-cutting story (what was easy, what fought back, costs).
- `blog/OUTLINE.md` is fleshed into a draft post.
- README badges/links work; no secrets committed (add `.env` to `.gitignore`).

## Working style

Do one project at a time. Start each by listing models, then write the smallest code
that produces `results.json`, then fill `NOTES.md` honestly, then move on. When something
breaks (it will — fine-tuning jobs, rate limits, model availability), capture the error and
the fix in NOTES.md verbatim; that candor is the most valuable part of the blog.
