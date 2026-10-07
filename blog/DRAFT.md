# What it's actually like to build on Fireworks

*A honest builder's log: one use case, ten capabilities, all the bugs left in.*

I gave myself a constraint: build **one** product — a customer-support copilot — and drive
it through every major capability of the Fireworks platform, on a public synthetic dataset
small enough that anyone can clone and re-run it. Classify a ticket, return structured
output, call a tool, ground a draft in policy with RAG, fine-tune (SFT then RFT), gate it
all with evals, serve many tenants with LoRA, benchmark the serving, and read an error
screenshot with a vision model. Ten small projects, one `shared/` harness, `results.json`
in every folder.

This is the field report. The numbers are reproducible; I've left the failures in, because
the failures are the actual information.

---

## 1. The setup: one API, swap the model id

Fireworks is OpenAI-compatible. The entire client is:

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.environ["FIREWORKS_API_KEY"])
```

Every project imports the same three helpers and the same `eval.py` / `bench.py`. To
compare a base model, a fine-tune, and a frontier model, you change one string — the model
id. That endpoint-swap is the whole ergonomic pitch, and it held up across all ten projects.
First call to first token: about a minute.

---

## 2. The capability tour (with one gotcha each)

**Serverless baseline (classify).** First run scored **8.3% on every model** — suspiciously
identical. The catalog is all *reasoning* models: they think in a hidden `reasoning_content`
field and put the answer in `content`, and `max_tokens` covers *both*. An 8-token budget
truncated them before they answered. Give them room and accuracy jumps to 83–100%
(gpt-oss-120b: 100%). **Gotcha: for reasoning models, `max_tokens` is a reasoning budget,
not an answer budget — and "all models tie at the floor" is the tell.**

**Structured output.** A JSON schema with an `enum` gave 100% valid, parseable JSON on 4 of
5 models — and a free `confidence` field that's actually mildly calibrated (it dropped to
0.8 on the one genuinely ambiguous ticket). **Gotcha: ask for a *schema*, not just "JSON" —
bare `json_object` invented a label outside my enum. And one model (glm) leaks its
chain-of-thought into `content` and never closes the JSON.**

**Function calling.** All five models supported tools with zero API errors, and argument
grounding was *perfect* — every correct call had the right `ticket_id`. **Gotcha: tool
*selection* is the hard part, not arguments. And when all five models unanimously disagreed
with my "correct" action on one ticket, the label was wrong, not the models — so I
relabeled it.**

**RAG.** Embeddings nailed retrieval (100% retrieval@1). Grounding raised the judge score
3.58 → 3.83 — but that average lies. It was **+2 where the policy had specifics and −4 on a
question the knowledge base didn't cover**: told to "use only the policy," the model deflected
on something it actually knew (our support hours). **Gotcha: forced retrieval poisons
out-of-KB questions; you need a relevance gate, not a blanket rule.**

**Fine-tune (SFT).** A LoRA on llama-3.1-8b lifted accuracy **66.7% → 83.3%**. But my first
attempt made the model *worse* (16.7%): the default token-based batch size packed all 96
examples into one batch, so "3 epochs" was **3 optimizer steps**. **Gotcha: count steps, not
epochs — use `--batch-size-samples`.**

**Fine-tune (RFT).** Reinforcement fine-tuning then beat SFT **83.3% → 100%**, fixing the
exact two boundary cases SFT missed. Getting there meant a four-tool detour (more below).
**Gotcha: the reward was 20 lines; *creating the evaluator* was the whole fight.**

**Evals (the gate).** One reusable harness, accuracy + a rubric judge — with the judge
pointed at a *different, stronger* model to avoid self-bias. The judge mean was stable to
**±0.039** across three runs even though individual scores wobble. **Gotcha: gate on
aggregates, never a single per-item judge score.**

**Multi-LoRA.** Two tenant "tone" adapters (formal Acme, friendly Bloom) on one base, same
prompt, two house styles — routing is just picking the adapter's model id. **Gotcha: the
whole value is economic (below); below ~2 tenants there's no benefit.**

**Serving perf.** gpt-oss-120b is the clear winner: TTFT 1.4s, 828 tok/s, cheapest.
**Gotcha: TTFT ≈ reasoning time, not network — and serverless cold-start hit 29s on a
less-warm model (3.2s warm). Always warm up and always report p95.**

**Vision (stretch).** qwen3-vl-8b read every error screenshot perfectly (**100% extraction**)
and routed 4/5 — the one miss was the same account-vs-technical ambiguity from the text
projects. **Gotcha: no vision model is serverless here; it's an on-demand deployment that
cold-starts on the first request.**

---

## 3. The numbers

| project | metric | headline |
|---|---|---|
| 00 baseline | accuracy | gpt-oss-120b **100%**, others 83–92% |
| 01 structured | valid-JSON | **100%** on 4/5 (glm intermittent) |
| 02 tools | tool-selection | **91.7%**, arg-grounding 100% |
| 03 RAG | judge 1–5 | 3.58 → **3.83** (bimodal: +2 / −4) |
| 04 SFT | accuracy | 66.7 → **83.3** (+16.6) |
| 05 RFT | accuracy | 83.3 → **100** (+16.7) |
| 06 evals | judge stability | **±0.039** across 3 runs |
| 07 multi-LoRA | cost/tenant | **90% cheaper** at 10 tenants |
| 08 serving | throughput | gpt-oss **828 tok/s**, 1.4s TTFT |
| 09 vision | extraction | **100%**, routing 80% |

The two that surprised me: **RFT going from 83% to a clean 100%** by fixing precisely the
cases SFT got wrong (and de-biasing its class distribution), and **the multi-LoRA economics**
— amortizing one GPU across tenants is 90% cheaper at 10 tenants, 99% at 100.

A caveat I want to be loud about: this is a **12-ticket** eval, and these MoE models aren't
deterministic at `temperature=0`. Single numbers wobble ±1 ticket; trust the aggregates and
the deltas, not the third decimal.

---

## 4. What fought back (the honest failure log)

- **Reasoning models silently zero out** if you starve `max_tokens` — the 8.3%-across-the-board
  bug.
- **`temperature=0` isn't deterministic.** Every project's numbers moved run-to-run.
- **"In the catalog" ≠ "serverless."** The fine-tune base and every vision model 404 on the
  inference endpoint; they're on-demand only.
- **Fine-tuning is mostly ops.** Eight things broke in the SFT project; one was about ML. The
  rest: fully-qualified model ids, an interactive deploy prompt, "no available capacity," a
  `#<deployment>` addressing syntax, addon loading, teardown.
- **Tooling lags the server.** Creating an RFT evaluator failed via reward-kit (`501
  deprecated`), the official SDK (same deprecated call), firectl (no command), *and* the UI
  (wrong decorator). The path that worked was a different framework, **eval-protocol**
  (`ep upload` / `ep create rft`). Then the RFT job **paused mid-run** and needed a manual
  `firectl resume`.
- **The dataset was often the bug.** One ticket fought all ten projects.

None of this is disqualifying — it's the normal texture of a maturing platform. But it's the
part the docs don't tell you, and it's where your hours go.

---

## 5. Build local vs. serve on Fireworks

The harness is deliberately endpoint-agnostic: `shared/bench.py` runs the *identical*
benchmark against a local vLLM server or Fireworks by swapping `--base-url`. The intended
punchline is a self-host-vs-managed comparison.

**Honest disclosure: I didn't run the self-host side.** This portfolio had no local GPU, so
there's no DGX/vLLM number here — only the Fireworks measurements. I'm calling that out
rather than hand-waving a comparison I didn't do. What I *can* say from the managed side:
serverless served a 120B at 828 tok/s with a 1.4s TTFT and zero infra work from me, and a
fine-tuned 8B needed a real on-demand GPU (~$8.99/hr est) to serve. The self-host contrast
is the obvious next experiment, and the tooling is already there to run it fairly.

---

## 6. Verdict for a partner

**Bet on it if** you want one platform from prototype to fine-tuned, multi-tenant
deployment. The OpenAI-compatible inference path is genuinely a one-minute on-ramp;
structured output, tools, RAG, and vision all worked; and the fine-tuning *results* are real
(SFT +16.6, RFT to 100%). Multi-LoRA makes multi-tenant economics excellent.

**Go in with eyes open about** the deployment lifecycle and tooling skew. Fine-tuning and
on-demand serving are where you'll debug: model-id namespaces, deployment shapes, capacity,
addon loading, scale-to-zero cold starts, a deprecated evaluator API, paused jobs. Treat it
as real infra engineering, not a button. And watch idle GPU spend — the bill is $/GPU-hour,
not $/token, the moment you self-serve a fine-tune; amortize it with multi-LoRA and tear
deployments down the instant you're done.

For a support-copilot-shaped product, my default stack out of this exercise is simple:
**gpt-oss-120b on serverless** (it topped both accuracy and speed), structured output for the
classifier, tools for actions, RAG with a relevance gate, and fine-tuning only where a
measured gap justifies the serving cost.

---

*Everything here is reproducible: clone the repo, set `FIREWORKS_API_KEY`, and run each
project's one documented command. `results/RESULTS.md` is regenerated by
`python shared/aggregate.py`; the per-project `NOTES.md` files have the verbatim errors and
fixes behind every claim above.*
