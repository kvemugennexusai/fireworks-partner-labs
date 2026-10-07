"""projects/00-serverless-baseline/run.py

Reference project — the pattern every other project follows.
Classifies each support ticket's intent across all available catalog models on
Fireworks serverless, scores accuracy, and writes results.json.

    python run.py
"""
import datetime
import json
import os
import re
import sys

# make shared/ and data/ importable/readable from the repo root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from shared.fw_client import TEXT_MODELS, chat, list_models, short_name  # noqa: E402

LABELS = ["billing", "technical", "account", "other"]
SYSTEM = ("Classify the support ticket into exactly one of: "
          "billing, technical, account, other. Reply with ONLY the label.")

# Every catalog model here is a *reasoning* model: it thinks before answering and
# returns the visible answer in `content` with the chain-of-thought in a separate
# `reasoning_content` field. max_tokens must be big enough for reasoning to FINISH —
# truncate it (e.g. 8) and `content` comes back empty and every model scores at the
# floor. One identical budget across models keeps the comparison fair (see NOTES.md).
MAX_TOKENS = 1024


def load_tickets():
    path = os.path.join(ROOT, "data", "tickets.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def pick_label(text):
    """Extract the intent label from a model reply.

    `content` is normally the bare label, but if any reasoning leaks in we want the
    model's *conclusion*, so scan from the last line backward and prefer the label
    that appears latest in the text rather than the first one in LABELS order.
    """
    for line in reversed([ln for ln in (text or "").splitlines() if ln.strip()]):
        t = norm(line)
        hits = [(t.rfind(norm(L)), L) for L in LABELS if norm(L) in t]
        if hits:
            return max(hits)[1]
    return "other"


def main():
    tickets = load_tickets()
    available = set(list_models())
    models = [m for m in TEXT_MODELS if m in available]
    if not models:
        print("None of the expected text models are available. Catalog:", sorted(available))
        models = [m for m in available if "flux" not in m][:3]
        print("Falling back to:", models)

    rows = []
    for model in models:
        correct = 0
        for tk in tickets:
            try:
                out = chat(model, [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": tk["input"]}],
                           max_tokens=MAX_TOKENS)
            except Exception as e:  # noqa: BLE001
                print(f"  {short_name(model)} error: {e}")
                out = ""
            if pick_label(out) == tk["label"]:
                correct += 1
        acc = round(100 * correct / len(tickets), 1)
        print(f"{short_name(model):24s} accuracy {acc}%  ({correct}/{len(tickets)})")
        rows.append({"model": short_name(model), "value": acc,
                     "notes": f"{correct}/{len(tickets)} correct"})

    result = {
        "project": "00-serverless-baseline",
        "metric": "intent accuracy", "unit": "%",
        "rows": rows,
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("wrote results.json")


if __name__ == "__main__":
    main()
