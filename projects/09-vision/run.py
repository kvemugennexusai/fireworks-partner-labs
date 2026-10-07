"""projects/09-vision/run.py

Vision step of the Support Copilot: feed an error screenshot, have a multimodal model
(1) extract the error text and (2) route it to an intent. Scores extraction accuracy
(did it surface the key phrase) and routing accuracy (intent vs gold).

No vision model is serverless on this account, so qwen3-vl-8b-instruct runs on an
on-demand deployment, addressed with the `#<deployment>` suffix. Set DEPLOYMENT below.
See README. Run generate_images.py first.

    python run.py
"""
import base64
import datetime
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from shared.fw_client import client  # noqa: E402

DEPLOYMENT = "accounts/kvemu-o43cdgtt8e51/deployments/g4vk6imi"
MODEL = f"accounts/fireworks/models/qwen3-vl-8b-instruct#{DEPLOYMENT}"
LABELS = ["billing", "technical", "account", "other"]
PROMPT = ("This is a screenshot of an application error. Read the error message and "
          "classify the support intent. Respond ONLY with JSON: "
          '{"error_text": "<the exact error message>", '
          '"intent": "billing|technical|account|other"}.')


def data_url(path):
    b = open(os.path.join(HERE, path), "rb").read()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def parse(out):
    m = re.search(r"\{.*\}", out, re.S)
    try:
        o = json.loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001
        o = {}
    intent = (o.get("intent") or "").strip().lower()
    return o.get("error_text", ""), (intent if intent in LABELS else "other")


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "labels.jsonl")) if l.strip()]
    c = client()
    extract_ok = route_ok = 0
    print(f"{'id':5}{'extract':>9}{'route':>7}  detail")
    for r in rows:
        msg = [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_url(r["image"])}}]}]
        resp = c.chat.completions.create(model=MODEL, messages=msg, max_tokens=300,
                                         temperature=0).choices[0].message.content or ""
        err_text, intent = parse(resp)
        ex = r["key_phrase"].lower() in err_text.lower()
        rt = intent == r["intent"]
        extract_ok += ex
        route_ok += rt
        print(f"{r['id']:5}{('OK' if ex else 'MISS'):>9}{('OK' if rt else 'X'):>7}  "
              f"{intent} | {err_text[:46]!r}")
    n = len(rows)
    ex_acc = round(100 * extract_ok / n, 1)
    rt_acc = round(100 * route_ok / n, 1)
    print(f"\nextraction {ex_acc}%  ({extract_ok}/{n})   routing {rt_acc}%  ({route_ok}/{n})")

    result = {
        "project": "09-vision",
        "metric": "vision: error extraction + intent routing", "unit": "%",
        "rows": [
            {"model": "qwen3-vl-8b-instruct (extraction)", "value": ex_acc,
             "notes": f"key error phrase surfaced from screenshot, {extract_ok}/{n} images"},
            {"model": "qwen3-vl-8b-instruct (routing)", "value": rt_acc,
             "notes": f"intent routed correctly from screenshot, {route_ok}/{n} images"},
        ],
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("wrote results.json")


if __name__ == "__main__":
    main()
