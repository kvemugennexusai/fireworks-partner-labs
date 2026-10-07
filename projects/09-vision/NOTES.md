# NOTES — 09 vision (stretch)

> Raw material for the blog — candid about what broke and how it was fixed.

## Goal
Vision step of the Support Copilot: read an error screenshot, extract the error text,
and route it to an intent. Score extraction + routing on a handful of sample images.

## Catalog check (the README's first instruction)
- **No vision model is serverless on this account.** `qwen3-vl-8b-instruct`,
  `qwen2-vl-7b-instruct`, `llama-v3p2-11b-vision-instruct` all 404 on the inference
  endpoint (same as the llama base — serverless is a curated subset here). The only
  serverless multimodal id is `flux-1-schnell-fp8`, which is image *generation*, not
  understanding.
- **But a VL model is deployable on-demand:** `qwen3-vl-8b-instruct` has a deployment
  shape (`qwen3-vl-8b-instruct-throughput`, 1× H200). So instead of documenting "no
  model available," I deployed it on-demand, ran the eval, and tore it down.

## Results (qwen3-vl-8b-instruct, 5 synthetic error screenshots)
| metric | score |
|---|---|
| **error-text extraction** | **100% (5/5)** |
| intent routing | 80% (4/5) |

- **Extraction was flawless** — it read every error verbatim ("HTTP 429: Too Many
  Requests...", "NullPointerException at Dashboard.render", "card ending 4242 was
  declined"), tofu-box ⚠ glyph and all. As an OCR-for-support-screenshots step, qwen3-vl
  is more than good enough.
- **The one routing miss is the same boundary that's dogged the whole portfolio:** err3
  "Login Failed: incorrect password" → model said `technical`, gold is `account`. Auth
  failures sit on the account/technical line; the model read the text perfectly and made
  a defensible call. Vision didn't add error here — it inherited the same label ambiguity
  as the text projects.

## Steps taken
1. `generate_images.py` renders 5 error screenshots with PIL (window chrome + red error
   banner) so the ground-truth text and intent are exact — no flaky real screenshots, and
   flux can't render reliable text anyway.
2. Deploy `qwen3-vl-8b-instruct` on-demand; send each image as a base64 `image_url` with a
   JSON-output prompt asking for `{error_text, intent}`; score extraction (key phrase
   present) + routing (intent vs gold).

## Bugs & fixes
| symptom | cause | fix |
|---------|-------|-----|
| no vision model on serverless (404) | serverless catalog is curated; VL models are on-demand only here | deploy `qwen3-vl-8b-instruct` on its deployment shape, address via `#<deployment>` |
| first request after READY → `503 DEPLOYMENT_SCALING_UP` | the throughput shape is **scale-to-zero**: READY means "ready to scale", 0 replicas until first request triggers a cold scale-up | warm-up retry loop (4 retries × 20s ≈ 80s) before the real run; then steady |
| ⚠ emoji rendered as a tofu box | the system font lacks that glyph | cosmetic only — the error *text* renders fine and that's what's scored |

## Cost
- One on-demand H200 (qwen3-vl throughput shape), up only for the warm-up + 5 images, then
  deleted. Billing reads $0 (trial credits); per-call cost not separately measured. The
  scale-to-zero window means it would have idled down on its own, but I deleted it
  explicitly to be safe.

## Learnings (2-3)
- **Multimodal "read the screenshot" is the easy part; routing is the same hard part.**
  qwen3-vl nailed extraction 5/5; the only error was the account-vs-technical boundary
  that text models also miss. Adding a camera didn't add a new failure mode — it inherited
  the existing label ambiguity.
- **Serverless ≠ everything in the catalog.** Vision needed an on-demand deployment here.
  Always probe the inference endpoint for the *specific* id before assuming it's callable
  (this is the third project — after 04 and 05 — where "in the catalog" ≠ "serverless").
- **Scale-to-zero deployments cold-start on first request.** READY is not "warm." Budget a
  retry/warm-up before benchmarking or a user-facing call, or you'll serve a 503.
