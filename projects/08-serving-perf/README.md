# 08 — Serving performance
**Capability:** FireAttention serving — latency / throughput / cost, FP8 / speculative decoding.
**Build:** use `shared/bench.py` to measure TTFT, p95, throughput, and $/1M tokens across
models and deployment options (serverless vs. on-demand). Note FP8 / speculative-decoding
effects where exposed.
**Definition of done:** `results.json` (perf table); `NOTES.md`. Also document **BYOC /
Enterprise-Reserved** here as written notes — they need enterprise infra, so no runnable code.

## Run
```bash
python run.py     # benchmarks all serverless catalog models -> perf table + results.json
```
Or one model directly via the shared harness (swap --base-url/--model for any endpoint):
```bash
python ../../shared/bench.py --base-url https://api.fireworks.ai/inference/v1 \
  --api-key "$FIREWORKS_API_KEY" --model accounts/fireworks/models/gpt-oss-120b \
  --prompts ../../data/tickets.jsonl --num-requests 24 --concurrency 8 --max-tokens 200 \
  --price-in 0.15 --price-out 0.60
```

See NOTES.md for the serverless-vs-on-demand crossover, FP8/spec-decoding, and BYOC /
Enterprise-Reserved write-ups. Per-token $/1M figures are **estimates** (billing read $0 on
trial credits); latency/throughput are measured.
