# Blog outline — "What it's actually like to build on Fireworks"

Audience: technical partners / founders evaluating the platform. Voice: honest builder.

1. **Why I did this** — one use case (Support Copilot) driven through every capability,
   on a public dataset, with the bugs left in.
2. **The setup** — OpenAI-compatible API, one harness, swap the model id. (1 min to first call.)
3. **Capability tour** (one short section each, with a table + one gotcha):
   - serverless across models · structured output · function calling · RAG ·
     SFT · RFT · evals · Multi-LoRA · serving perf.
4. **The numbers** — the master results table; the one or two results that surprised me.
5. **What fought back** — the honest failure log (catalog drift, fine-tune gotchas, rate limits).
6. **Build local vs. serve on Fireworks** — fold in the DGX Spark benchmark: self-hosting a
   120B took days of CUDA/memory wrangling and still served an 8B slower than Fireworks
   served a 120B. The contrast is the whole pitch.
7. **Verdict for a partner** — when this platform is the right call, and when it isn't.

Keep each section tight; let the tables and the candor do the work.
