"""projects/01-structured-output/run.py

Schema-constrained classify step. Instead of scraping a bare label out of free
text (project 00), we declare a JSON schema and let the API force the model to
emit conforming JSON: {"intent": "<label>", "confidence": <0-1>}.

Metric: valid-JSON rate % (does raw `content` parse AND match the schema), with
intent accuracy carried in the notes. The point is production-parseability, not
just smarts.

    python run.py
"""
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from shared.fw_client import TEXT_MODELS, client, list_models, short_name  # noqa: E402

LABELS = ["billing", "technical", "account", "other"]
SYSTEM = ("Classify the support ticket into one of: billing, technical, account, "
          "other. Return the intent and your confidence (0-1).")
MAX_TOKENS = 1024  # reasoning models still think first; leave room (see project 00)

# OpenAI-compatible json_schema. Fireworks also accepts {"type":"json_object",
# "schema": ...}; both enforce the enum. A bare {"type":"json_object"} does NOT —
# it invents labels like "refund_request" and drops fields (see NOTES.md).
SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": LABELS},
        "confidence": {"type": "number"},
    },
    "required": ["intent", "confidence"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "ticket_intent", "schema": SCHEMA}}


def load_tickets():
    with open(os.path.join(ROOT, "data", "tickets.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(raw):
    """Return (intent_or_None, conforms_bool). conforms = raw `content` parses as
    JSON and satisfies the schema with no cleanup (no fence-stripping, no repair)."""
    try:
        obj = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None, False
    if not isinstance(obj, dict):
        return None, False
    intent = obj.get("intent")
    conf = obj.get("confidence")
    ok = (intent in LABELS and isinstance(conf, (int, float))
          and not isinstance(conf, bool) and 0.0 <= conf <= 1.0)
    return (intent if intent in LABELS else None), ok


def main():
    tickets = load_tickets()
    available = set(list_models())
    models = [m for m in TEXT_MODELS if m in available]
    c = client()

    rows = []
    for model in models:
        valid = correct = api_errors = 0
        for tk in tickets:
            try:
                r = c.chat.completions.create(
                    model=model, messages=[{"role": "system", "content": SYSTEM},
                                            {"role": "user", "content": tk["input"]}],
                    response_format=RESPONSE_FORMAT, max_tokens=MAX_TOKENS,
                    temperature=0)
                raw = r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                print(f"  {short_name(model)} api error: {str(e)[:120]}")
                api_errors += 1
                raw = ""
            intent, ok = validate(raw)
            if ok:
                valid += 1
            if ok and intent == tk["label"]:
                correct += 1
        n = len(tickets)
        vrate = round(100 * valid / n, 1)
        acc = round(100 * correct / n, 1)
        note = f"valid {valid}/{n}, accuracy {acc}% ({correct}/{n})"
        if api_errors:
            note += f", {api_errors} api errors"
        print(f"{short_name(model):24s} valid-JSON {vrate:5}%   accuracy {acc}%")
        rows.append({"model": short_name(model), "value": vrate, "notes": note})

    result = {
        "project": "01-structured-output",
        "metric": "valid-JSON rate", "unit": "%",
        "rows": rows,
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("wrote results.json")


if __name__ == "__main__":
    main()
