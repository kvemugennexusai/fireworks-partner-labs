"""projects/08-serving-perf/run.py

Serving performance across the serverless catalog using shared/bench.py: TTFT, ITL,
end-to-end latency, system throughput, and $/1M output tokens. Writes a perf table.

Cost note: account billing reads $0.00 (trial credits), and these model ids are a
newer set, so the per-token PRICES below are ESTIMATES from Fireworks' published
serverless tiers — marked as such. The latency/throughput numbers are MEASURED.

    python run.py
"""
import datetime
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from shared.fw_client import TEXT_MODELS, list_models, short_name  # noqa: E402

FW_URL = "https://api.fireworks.ai/inference/v1"
PROMPTS = os.path.join(ROOT, "data", "tickets.jsonl")
BENCH = os.path.join(ROOT, "shared", "bench.py")
NUM_REQUESTS = 24
CONCURRENCY = 8
MAX_TOKENS = 200

# $/1M tokens (in, out) — ESTIMATE from Fireworks published serverless tiers; verify.
PRICES = {
    "gpt-oss-120b":    (0.15, 0.60),
    "deepseek-v4-pro": (0.90, 0.90),
    "kimi-k2p5":       (0.60, 2.50),
    "kimi-k2p6":       (0.60, 2.50),
    "glm-5p1":         (0.55, 2.19),
}


def bench(model):
    sn = short_name(model)
    pin, pout = PRICES.get(sn, (0.5, 0.5))
    out_path = os.path.join(HERE, f"_bench_{sn}.json")
    cmd = [sys.executable, BENCH, "--base-url", FW_URL,
           "--api-key", os.environ["FIREWORKS_API_KEY"], "--model", model,
           "--prompts", PROMPTS, "--num-requests", str(NUM_REQUESTS),
           "--concurrency", str(CONCURRENCY), "--max-tokens", str(MAX_TOKENS),
           "--warmup", "3", "--price-in", str(pin), "--price-out", str(pout),
           "--output", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return json.load(open(out_path))["summary"]


def main():
    available = set(list_models())
    models = [m for m in TEXT_MODELS if m in available]
    rows = []
    print(f"{'model':22} {'TTFT p50':>9} {'TTFT p95':>9} {'ITL ms':>7} {'tok/s':>7} {'$/1M out*':>9}")
    for m in models:
        s = bench(m)
        sn = short_name(m)
        ttft50 = s["ttft_s"]["p50"]; ttft95 = s["ttft_s"]["p95"]
        itl = s["itl_ms"]["p50"]; tput = s["system_throughput_tok_s"]
        cpm = s["cost"]["usd_per_1M_output_tok"] if s.get("cost") else None
        print(f"{sn:22} {ttft50:>9.3f} {ttft95:>9.3f} {itl:>7.1f} {tput:>7.0f} {cpm:>9}")
        rows.append({"model": sn, "value": tput,
                     "notes": (f"throughput {tput:.0f} tok/s; TTFT p50 {ttft50:.2f}s / "
                               f"p95 {ttft95:.2f}s; ITL p50 {itl:.1f}ms; "
                               f"~${cpm}/1M out tok (EST); serverless, conc={CONCURRENCY}")})
    result = {
        "project": "08-serving-perf",
        "metric": "serverless throughput (tok/s) + latency", "unit": "tok/s",
        "rows": rows,
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\n* $/1M is an ESTIMATE (billing read $0 on credits). wrote results.json")


if __name__ == "__main__":
    main()
