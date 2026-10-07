"""projects/06-evals/run.py  — the shared gate

One reusable entrypoint that wraps shared/eval.py and establishes the baseline every
other project reports against:

  Track A  classification accuracy  — intent label vs gold, all catalog models.
  Track B  rubric judge (1-5)       — draft-response quality, judged by a SEPARATE
                                       stronger model (gpt-oss-120b) to reduce self-bias,
                                       run K times to report judge STABILITY (mean +/- sd).

Contamination discipline: data/tickets.jsonl is the FROZEN eval split. Projects 04/05
trained on synthetic data with these 12 held out — they are never trained on. Do not
add them to any training set.

    python run.py
"""
import datetime
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from shared.fw_client import TEXT_MODELS, list_models, short_name  # noqa: E402

FW_URL = "https://api.fireworks.ai/inference/v1"
DATA = os.path.join(ROOT, "data", "tickets.jsonl")
EVAL = os.path.join(ROOT, "shared", "eval.py")

CLF_SYSTEM = ("Classify the support ticket into exactly one of: billing, technical, "
              "account, other. Reply with ONLY the label.")
LABELS = ["billing", "technical", "account", "other"]
DRAFT_SYSTEM = ("You are a customer-support agent. Draft a concise, helpful first "
                "response to the ticket.")
MAX_TOKENS = 1024            # reasoning models need room (see project 00)
DRAFT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
JUDGE_MODEL = "accounts/fireworks/models/gpt-oss-120b"   # separate -> less self-bias
JUDGE_RUNS = 3               # stability: judge the same answers K times


def eval_cli(args, out_path):
    cmd = [sys.executable, EVAL, "--base-url", FW_URL,
           "--api-key", os.environ["FIREWORKS_API_KEY"], "--data", DATA,
           "--output", out_path, "--temperature", "0", *args]
    subprocess.run(cmd, check=True)
    return json.load(open(out_path))["summary"]


def main():
    available = set(list_models())
    models = [m for m in TEXT_MODELS if m in available]

    # --- Track A: classification accuracy across catalog models ---
    print("=== Track A: classification accuracy ===")
    acc_rows = []
    for m in models:
        # 1024, not 16: every catalog model is a reasoning model and a tiny budget
        # truncates it before the label (project 00 trap).
        s = eval_cli(["--model", m, "--task", "classification", "--system", CLF_SYSTEM,
                      "--labels", *LABELS, "--max-tokens", str(MAX_TOKENS)],
                     os.path.join(HERE, f"_clf_{short_name(m)}.json"))
        acc = round(100 * s.get("accuracy", 0), 1)
        print(f"  {short_name(m):24s} {acc}%")
        acc_rows.append({"model": short_name(m), "value": acc,
                         "notes": f"intent accuracy, {s.get('n')} tickets"})

    # --- Track B: rubric judge with stability over K runs ---
    print(f"\n=== Track B: rubric judge (drafter={short_name(DRAFT_MODEL)}, "
          f"judge={short_name(JUDGE_MODEL)}, {JUDGE_RUNS} runs) ===")
    judge_means = []
    for k in range(JUDGE_RUNS):
        s = eval_cli(["--model", DRAFT_MODEL, "--task", "judge", "--system", DRAFT_SYSTEM,
                      "--max-tokens", str(MAX_TOKENS), "--judge-base-url", FW_URL,
                      "--judge-model", JUDGE_MODEL, "--judge-max-tokens", "1024"],
                     os.path.join(HERE, f"_judge_run{k}.json"))
        judge_means.append(s.get("mean_score", 0))
        print(f"  run {k+1}: mean judge {s.get('mean_score')}")
    jmean = round(statistics.mean(judge_means), 3)
    jsd = round(statistics.pstdev(judge_means), 3)
    print(f"  -> judge {jmean} +/- {jsd} (1-5) over {JUDGE_RUNS} runs")

    rows = acc_rows + [
        {"model": f"{short_name(DRAFT_MODEL)} draft / {short_name(JUDGE_MODEL)} judge",
         "value": jmean,
         "notes": (f"rubric judge 1-5 (mean of {JUDGE_RUNS} runs), stability +/-{jsd}; "
                   f"runs={judge_means}")},
    ]
    result = {
        "project": "06-evals",
        "metric": "shared gate: accuracy (%) + rubric judge (1-5)", "unit": "mixed",
        "rows": rows,
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
