"""projects/04-finetune-sft/run.py

Compare the SFT fine-tuned model against its base on the held-out 12 tickets
(data/tickets.jsonl is the EVAL set; training used synthetic train.jsonl). Same
eval harness (shared/eval.py, classification) pointed at two model ids — the
endpoint/model swap is the whole point of fine-tuning work.

Prereq: an on-demand deployment of the base with addons enabled, so BOTH ids are
reachable (the base is not serverless on this account). See README for the
firectl commands. Set BASE_MODEL / SFT_MODEL below to match.

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

FW_URL = "https://api.fireworks.ai/inference/v1"
DATA = os.path.join(ROOT, "data", "tickets.jsonl")
SYSTEM = ("Classify the support ticket into exactly one of: billing, technical, "
          "account, other. Reply with ONLY the label.")
LABELS = ["billing", "technical", "account", "other"]

# The base is not serverless on this account, so both ids are addressed through the
# on-demand deployment with the `#<deployment>` suffix (the LoRA addon is loaded onto
# the same deployment via `firectl load-lora`; see README).
DEPLOYMENT = "accounts/kvemu-o43cdgtt8e51/deployments/q4q7bmpb"
BASE_MODEL = f"accounts/fireworks/models/llama-v3p1-8b-instruct#{DEPLOYMENT}"
# v2: sample-count batching (--batch-size-samples 8) + 5 epochs + LoRA r16. v1
# (token batching, 3 epochs) collapsed to one class because all 96 examples packed
# into a single batch => only 3 optimizer steps total. See NOTES.md.
SFT_MODEL = f"accounts/kvemu-o43cdgtt8e51/models/support-copilot-sft-v2#{DEPLOYMENT}"


def eval_model(model, tag):
    out_path = os.path.join(HERE, f"_eval_{tag}.json")
    cmd = [sys.executable, os.path.join(ROOT, "shared", "eval.py"),
           "--base-url", FW_URL, "--api-key", os.environ["FIREWORKS_API_KEY"],
           "--model", model, "--data", DATA, "--task", "classification",
           "--system", SYSTEM, "--labels", *LABELS,
           "--max-tokens", "16", "--temperature", "0", "--output", out_path]
    subprocess.run(cmd, check=True)
    return json.load(open(out_path))["summary"]


def main():
    print(f"--- base: {BASE_MODEL} ---")
    base = eval_model(BASE_MODEL, "base")
    print(f"--- sft:  {SFT_MODEL} ---")
    sft = eval_model(SFT_MODEL, "sft")

    base_acc = round(100 * base.get("accuracy", 0), 1)
    sft_acc = round(100 * sft.get("accuracy", 0), 1)
    rows = [
        {"model": "llama-v3p1-8b-instruct (base)", "value": base_acc,
         "notes": f"intent accuracy on 12 held-out tickets, n={base.get('n')}"},
        {"model": "support-copilot-sft-v2 (LoRA SFT)", "value": sft_acc,
         "notes": (f"intent accuracy on 12 held-out tickets; LoRA r16, 5 epochs, "
                   f"batch-size-samples 8, 96 synthetic train examples; "
                   f"delta {sft_acc - base_acc:+.1f} pts vs base")},
    ]
    result = {
        "project": "04-finetune-sft",
        "metric": "intent accuracy (base vs SFT)", "unit": "%",
        "rows": rows,
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nbase {base_acc}%  ->  SFT {sft_acc}%   (delta {sft_acc - base_acc:+.1f})")
    print("wrote results.json")


if __name__ == "__main__":
    main()
