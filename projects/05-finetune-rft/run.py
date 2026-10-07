"""projects/05-finetune-rft/run.py

Compare the RFT model against the SFT model (project 04) on the 12 held-out tickets
(data/tickets.jsonl). Both are LoRA addons on llama-v3p1-8b-instruct, loaded onto
one on-demand deployment and addressed with the `#<deployment>` suffix (the base is
not serverless on this account — see project 04 NOTES). Same eval harness
(shared/eval.py, classification).

Prereq: deploy the base with addons, `firectl load-lora` BOTH addons, set DEPLOYMENT
below. See README.

    python run.py
"""
import datetime
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))

FW_URL = "https://api.fireworks.ai/inference/v1"
DATA = os.path.join(ROOT, "data", "tickets.jsonl")
SYSTEM = ("Classify the support ticket into exactly one of: billing, technical, "
          "account, other. Reply with ONLY the label.")
LABELS = ["billing", "technical", "account", "other"]

DEPLOYMENT = "accounts/kvemu-o43cdgtt8e51/deployments/meofq19v"
SFT_MODEL = f"accounts/kvemu-o43cdgtt8e51/models/support-copilot-sft-v2#{DEPLOYMENT}"
RFT_MODEL = f"accounts/kvemu-o43cdgtt8e51/models/support-copilot-rft#{DEPLOYMENT}"


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
    print(f"--- SFT v2: {SFT_MODEL} ---")
    sft = eval_model(SFT_MODEL, "sft")
    print(f"--- RFT:    {RFT_MODEL} ---")
    rft = eval_model(RFT_MODEL, "rft")

    sft_acc = round(100 * sft.get("accuracy", 0), 1)
    rft_acc = round(100 * rft.get("accuracy", 0), 1)
    rows = [
        {"model": "support-copilot-sft-v2 (SFT)", "value": sft_acc,
         "notes": f"intent accuracy on 12 held-out tickets (imitation/SFT baseline)"},
        {"model": "support-copilot-rft (RFT)", "value": rft_acc,
         "notes": (f"intent accuracy on 12 held-out tickets; reward-optimized "
                   f"(correctness reward via eval-protocol evaluator), LoRA r16, "
                   f"5 epochs; delta {rft_acc - sft_acc:+.1f} pts vs SFT")},
    ]
    result = {
        "project": "05-finetune-rft",
        "metric": "intent accuracy (SFT vs RFT)", "unit": "%",
        "rows": rows,
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSFT {sft_acc}%  ->  RFT {rft_acc}%   (delta {rft_acc - sft_acc:+.1f})")
    print("wrote results.json")


if __name__ == "__main__":
    main()
