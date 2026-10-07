# NOTES — 07 multi-lora

> Raw material for the blog — candid about what broke and how it was fixed.

## Goal
Serve several LoRA adapters on ONE base, route per request, and model the per-tenant
economics (one amortized base vs N dedicated deployments).

## Steps taken
1. Built two *tenant tone* adapters on `llama-v3p1-8b-instruct`:
   - **acme** — formal/enterprise ("...The Acme Support Team")
   - **bloom** — warm/casual ("Hey there! ... Team Bloom 🌸")
   Trained on a **neutral** system prompt so the tone lives in the adapter weights,
   not the prompt (two SFT jobs, LoRA r16, 4 epochs, 40 synthetic examples each;
   generated with two batched gpt-oss calls).
2. Deployed the base once with `--enable-addons`, `load-lora`'d both adapters onto it.
3. Routing demo (`run.py`): same ticket + same neutral prompt, routed to each adapter.
4. Per-tenant cost model (`cost_model.py`).

## Adapter behavior (the demo — same base, same prompt, different tenant)
TICKET: "I was charged twice for my subscription this month and need a refund."
- **acme:**  "We have identified the duplicate charge on your subscription and are
  processing a refund for the amount. The Acme Support Team"
- **bloom:** "Hi! Sorry you were charged twice for your subscription—let's issue a
  refund right away. I'll take care of it for you. Team Bloom 🌸"

Identical input, two house styles. "Routing" on a multi-LoRA base is nothing fancier
than choosing the adapter's model id per request (`.../models/tone-acme#<deployment>`
vs `.../models/tone-bloom#<deployment>`) — same GPU, same endpoint.

## Per-tenant economics (cost_model.py)
1× H200-141GB @ ~$8.99/hr (ESTIMATE — see cost note) = ~$6,563/mo/base; ~100 adapters/base.
| tenants | multi-LoRA $/tenant/mo | dedicated $/tenant/mo | savings |
|--------:|----------------------:|----------------------:|--------:|
| 1   | 6,563 | 6,563 | 0% |
| 2   | 3,281 | 6,563 | 50% |
| 10  |   656 | 6,563 | 90% |
| 50  |   131 | 6,563 | 98% |
| 100 |    66 | 6,563 | 99% |

Savings = 1 − 1/N until a base saturates (then add another base and the curve repeats).
The whole value proposition of multi-LoRA is in that table: a LoRA adapter is a few MB,
so you rent ONE GPU and split it across tenants instead of renting N GPUs.

## Bugs & fixes
| symptom | cause | fix |
|---------|-------|-----|
| n/a — the serving/loading path was already de-risked in projects 04/05 | loading 2 addons on one deployment is the same `load-lora` flow | reused it; both addons served concurrently with no interference |
| could not read a real $/GPU-hr | `firectl billing get-usage` shows $0.00 (trial credits / billing lag) | used the published on-demand H200 rate as a clearly-marked ESTIMATE; model is parametric so the % savings hold regardless |

## Cost
- **Training:** 2 tone adapters, ~4–5 min each (ran concurrently, both COMPLETED ~2 min
  apart). Cheap.
- **Serving:** one shared H200 deployment hosted both adapters; torn down right after the
  demo. The economics above are the point — exact $/hr is an **estimate** (billing read
  $0 on trial credits); the published per-GPU-hour rate gets pinned in project 08.
- Per-token inference cost is ~identical per tenant in both designs, so it cancels in the
  comparison and is omitted from the model.

## Learnings (2-3)
- **Tone/behavior fine-tunes cleanly into a small adapter.** 40 examples on a neutral
  prompt was enough for each tenant's house style to show through reliably — the adapter,
  not the prompt, carries the brand voice.
- **Multi-LoRA's pitch is purely economic, and it's dramatic.** Routing is trivial (pick
  an id); the win is amortizing one GPU across N tenants — 90% cheaper at 10 tenants, 99%
  at 100, vs dedicated deployments. Below ~2 tenants there's no benefit.
- **Know the saturation point.** Amortization holds until a base hits its adapter/VRAM or
  QPS ceiling; past that you add a base and the per-tenant curve resets. The model bakes in
  `adapters_per_base` so the cliff is explicit rather than assumed-away.
