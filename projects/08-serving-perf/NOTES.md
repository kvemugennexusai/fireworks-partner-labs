# NOTES — 08 serving-perf

> Raw material for the blog — candid about what broke and how it was fixed.

## Goal
Measure serving performance (TTFT, ITL, p95 latency, throughput, $/1M) across the
serverless catalog with `shared/bench.py`, and document serverless-vs-on-demand,
FP8/speculative-decoding, and BYOC/Enterprise modes.

## Measured perf (serverless, 24 req @ concurrency 8, max_tokens 200)
| model | TTFT p50 | TTFT p95 | ITL | throughput | $/1M out (EST) |
|---|---|---|---|---|---|
| **gpt-oss-120b** | **1.4s** | 2.2s | 2.1ms | **828 tok/s** | ~$0.67 |
| deepseek-v4-pro | 3.9s | 4.6s | 0.9ms | 358 tok/s | ~$0.99 |
| glm-5p1 | 5.7s | 8.2s | ~0 | 252 tok/s | ~$2.25 |
| kimi-k2p6 | 2.7s | 14.1s | ~0 | 243 tok/s | ~$2.57 |
| kimi-k2p5 | 29.3s* | 71.1s* | ~0 | 41 tok/s | ~$2.57 |

\* **cold-start outlier** — a warm re-bench gave kimi-k2p5 TTFT p50 3.2s / p95 12.9s.

## Findings
- **gpt-oss-120b is the serving winner** by a wide margin: lowest TTFT, ~3.5× the
  throughput of the next model, and (estimated) cheapest. For the Support Copilot it's
  both the most accurate (projects 00/06) and the fastest — an easy default.
- **TTFT is dominated by reasoning, not network.** Every catalog model thinks before
  emitting a visible token, so "time to first token" really measures "time to finish
  reasoning." That's why TTFT ranges 1.4s → 29s while ITL (once streaming starts) is
  sub-3ms everywhere. For a reasoning-model product, TTFT is a reasoning-budget decision,
  not a serving-tuning one.
- **Two streaming shapes.** gpt-oss/deepseek stream content token-by-token (measurable
  ITL ~1-2ms). kimi/glm reason, then dump the whole answer in ~one chunk (ITL ≈ 0). Same
  endpoint, different emit behavior — worth knowing if your UI shows a token stream.
- **Serverless cold-start is real and large.** kimi-k2p5's first pass hit ~29s TTFT (p95
  71s); warm it was 3.2s. Less-popular models can be cold/queued on serverless. If p95
  matters, warm the model or use on-demand.
- **p95 tail >> p50 on serverless.** kimi-k2p6 p50 2.7s but p95 14s. Shared infra means
  noisy tails; never quote only the median.

## Serverless vs on-demand (when to switch)
- **Serverless:** pay-per-token, zero idle cost, no ops — but variable latency, cold
  starts, and noisy p95. Great for spiky/low-volume and for the big frontier models you
  can't economically host.
- **On-demand (what projects 04/05/07 used):** a dedicated deployment (1× H200 here) gives
  consistent latency and no cold start once READY, but you pay the GPU $/hr whether or not
  it's busy (~$6.5k/mo always-on, project 07's estimate).
- **Crossover = utilization.** Pay-per-token wins until your sustained volume would keep a
  dedicated GPU busy enough that $/hr beats $/token. The multi-LoRA amortization in project
  07 is what pushes on-demand into the money for multi-tenant serving.

## FP8 / speculative decoding (what's exposed)
- Fireworks exposes **deployment shapes** that encode precision/optimization, visible via
  `firectl list deployment-shape-versions`. For llama-v3p1-8b-instruct the presets are:
  `-throughput` and `-fast` (1× H100, **FP8**), `-minimal` (1× H100, FP8), and the
  `rft-...` shape (1× H200, **BF16/FULL_PRECISION**) used for fine-tune-compatible serving.
- We served on the BF16 `rft-` shape (LoRA addons need it). A clean FP8-vs-BF16 throughput
  A/B would need TWO on-demand deployments held at saturation — **not run here** to avoid
  the GPU spend; documented as the next experiment. Expectation: FP8 throughput shapes
  trade a little quality for materially higher tok/s and lower $/1M at saturation.
- Speculative decoding isn't a user-facing knob in the shapes we saw; it's applied
  server-side on serverless. We didn't have a switch to toggle, so no A/B.

## BYOC / Enterprise-Reserved (notes only — need enterprise infra)
- **BYOC (Bring Your Own Cloud):** Fireworks' control plane runs the serving stack inside
  *your* cloud account (your VPC/GPUs), so data and weights never leave your boundary. Set
  up via enterprise onboarding (cloud creds / IAM role — the SFT/RFT job flags even expose
  `--aws-iam-role` / `--azure-*` hooks). No runnable code here: it needs a provisioned cloud
  account and is gated behind enterprise. Use it for data-residency/compliance.
- **Enterprise-Reserved:** committed, reserved GPU capacity (vs on-demand spot-style) for
  guaranteed throughput and better $/GPU-hr at scale. Also an enterprise contract, not a CLI
  call. Use it when sustained volume makes reserved cheaper than on-demand and you need
  capacity guarantees (no "no available capacity" — which we hit on H100 in project 04).

## Cost
- **$/1M is ESTIMATED.** `firectl billing get-usage` reads $0.00 (trial credits), so I could
  not measure real spend; the per-token rates are from Fireworks' published serverless tiers
  and marked EST in `results.json`. The **latency/throughput numbers are measured** and real.
- Token accounting in `bench.py` uses streamed `usage` when present (exact), else counts
  chunks (approx). Here usage was available, so output-token counts are exact.

## Learnings (2-3)
- **For reasoning models, TTFT ≈ reasoning time.** Optimize the reasoning budget, not the
  network; ITL is already sub-3ms. The lever for perceived latency is how much the model
  thinks, not the serving stack.
- **Always report p95, and warm before you benchmark.** A single cold call (kimi-k2p5's 29s)
  can be 9× the warm median; serverless tails are wide and model-dependent.
- **Pick the model that's fast AND accurate — here they coincide.** gpt-oss-120b tops both
  the accuracy gate (project 06) and the serving table, so the Support Copilot's default
  model choice is unambiguous.
