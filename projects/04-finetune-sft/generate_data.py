"""projects/04-finetune-sft/generate_data.py

Build a synthetic SFT training set. The repo's 12 tickets are the held-out EVAL
set — training on them would be train==test leakage. So we synthesize a larger,
labeled-by-construction training set (ask a strong model for diverse tickets of a
known intent) and write it in Fireworks chat SFT format:

  {"messages": [{"role":"system",...},{"role":"user",<ticket>},{"role":"assistant",<label>}]}

Run once; the output train.jsonl is committed so the SFT job is reproducible.

    python generate_data.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from shared.fw_client import client  # noqa: E402

GEN_MODEL = "accounts/fireworks/models/gpt-oss-120b"
LABELS = ["billing", "technical", "account", "other"]
PER_LABEL = 24
SYSTEM = ("Classify the support ticket into exactly one of: billing, technical, "
          "account, other. Reply with ONLY the label.")

# hold-out eval ticket texts, so we never accidentally regenerate one verbatim
EVAL_INPUTS = {json.loads(l)["input"].strip().lower()
               for l in open(os.path.join(ROOT, "data", "tickets.jsonl")) if l.strip()}

GUIDE = {
    "billing": "charges, refunds, invoices, subscriptions, proration, plan pricing, payment failures",
    "technical": "bugs, crashes, errors, API failures, latency, integrations not working, outages",
    "account": "login, password reset, profile/contact changes, seats/permissions, account closure, SSO setup",
    "other": "general questions, hours, company info, feature requests, feedback, partnerships — NOT a problem to fix",
}


def gen_for(label):
    user = (f"Generate {PER_LABEL} diverse, realistic customer-support tickets whose intent is "
            f"clearly '{label}' (themes: {GUIDE[label]}). One sentence each, varied phrasing and "
            f"products. Return ONLY a JSON array of strings.")
    r = client().chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": "You generate synthetic training data."},
                  {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": "tickets", "schema": {
            "type": "object", "properties": {"tickets": {"type": "array", "items": {"type": "string"}}},
            "required": ["tickets"]}}},
        max_tokens=4096, temperature=0.7)
    return json.loads(r.choices[0].message.content)["tickets"]


def main():
    rows, seen = [], set()
    for label in LABELS:
        items = gen_for(label)
        kept = 0
        for txt in items:
            t = txt.strip()
            key = t.lower()
            if not t or key in EVAL_INPUTS or key in seen:
                continue       # skip dupes and any accidental eval-set overlap
            seen.add(key)
            rows.append({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": t},
                {"role": "assistant", "content": label}]})
            kept += 1
        print(f"{label:10s} kept {kept}/{len(items)}")
    out = os.path.join(HERE, "train.jsonl")
    with open(out, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    print(f"wrote {len(rows)} examples -> {out}")


if __name__ == "__main__":
    main()
