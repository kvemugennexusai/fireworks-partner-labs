"""projects/07-multi-lora/generate_tone_data.py

Build SFT data for two *tenant tone* adapters on one base:
  acme  -> formal, enterprise, buttoned-up
  bloom -> warm, casual, friendly

Both train on a NEUTRAL system prompt so the TONE is baked into the adapter weights
(not the prompt) — that's what makes it a per-tenant adapter: same request, different
house style depending on which adapter is routed to. Reuses the synthetic tickets from
project 04. Two batched generation calls (one per tone).

    python generate_tone_data.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from shared.fw_client import client  # noqa: E402

GEN_MODEL = "accounts/fireworks/models/gpt-oss-120b"
NEUTRAL_SYSTEM = "You are a customer-support agent. Draft a concise first response to the ticket."
N = 40

TONES = {
    "acme": ("formal, professional, enterprise tone — full sentences, courteous, no "
             "contractions, no emoji, signs off as 'The Acme Support Team'"),
    "bloom": ("warm, friendly, casual tone — uses the customer's vibe, contractions, an "
              "occasional emoji, upbeat, signs off as 'Team Bloom 🌸'"),
}


def tickets():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "projects", "04-finetune-sft",
                                                     "train.jsonl")) if l.strip()]
    return [r["messages"][1]["content"] for r in rows][:N]


def gen(tone_desc, tks):
    user = ("Write a first-response for each support ticket below in this tone: "
            f"{tone_desc}. 2-3 sentences each. Return ONLY a JSON object "
            '{"responses": ["...", ...]} in the same order.\n\nTICKETS:\n'
            + "\n".join(f"{i+1}. {t}" for i, t in enumerate(tks)))
    r = client().chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": "You write support replies in a given house style."},
                  {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": "r", "schema": {
            "type": "object", "properties": {"responses": {"type": "array", "items": {"type": "string"}}},
            "required": ["responses"]}}},
        max_tokens=8192, temperature=0.4)
    return json.loads(r.choices[0].message.content)["responses"]


def main():
    tks = tickets()
    for tenant, desc in TONES.items():
        resps = gen(desc, tks)
        rows = []
        for t, resp in zip(tks, resps):
            rows.append({"messages": [
                {"role": "system", "content": NEUTRAL_SYSTEM},
                {"role": "user", "content": t},
                {"role": "assistant", "content": resp.strip()}]})
        out = os.path.join(HERE, f"train_{tenant}.jsonl")
        with open(out, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
        print(f"{tenant:6s} wrote {len(rows)} examples -> {out}")
        print(f"        sample: {resps[0][:90]!r}")


if __name__ == "__main__":
    main()
