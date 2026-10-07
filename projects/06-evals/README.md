# 06 — Evals (the shared gate)
**Capability:** Eval Protocol — accuracy + LLM-judge.
**Build:** wrap `shared/eval.py` into a reusable entrypoint every project reports against:
classification accuracy and a 1-5 rubric judge (point the judge at a separate, stronger
model to reduce self-bias). Establish the baseline scores.
**Definition of done:** a documented eval command + `results.json` baseline; `NOTES.md` on
judge stability and contamination discipline (freeze the eval split, never train on it).

## The gate
One reusable entrypoint over `shared/eval.py`. Run the full baseline:

```bash
python run.py        # Track A (accuracy, all models) + Track B (judge, 3 runs)
```

Or invoke either track directly via the shared harness (this is what every project
reports against — swap `--model` to compare any base / fine-tuned / frontier endpoint):

```bash
# accuracy — note --max-tokens 1024 for reasoning models (64 default silently zeroes them)
python ../../shared/eval.py --base-url https://api.fireworks.ai/inference/v1 \
  --api-key "$FIREWORKS_API_KEY" --model accounts/fireworks/models/deepseek-v4-pro \
  --data ../../data/tickets.jsonl --task classification \
  --system "Classify the support ticket into exactly one of: billing, technical, account, other. Reply with ONLY the label." \
  --labels billing technical account other --max-tokens 1024

# rubric judge — judge with a DIFFERENT, stronger model than the drafter
python ../../shared/eval.py --base-url https://api.fireworks.ai/inference/v1 \
  --api-key "$FIREWORKS_API_KEY" --model accounts/fireworks/models/deepseek-v4-pro \
  --data ../../data/tickets.jsonl --task judge --max-tokens 1024 \
  --system "You are a customer-support agent. Draft a concise, helpful first response to the ticket." \
  --judge-base-url https://api.fireworks.ai/inference/v1 \
  --judge-model accounts/fireworks/models/gpt-oss-120b --judge-max-tokens 1024
```

**Frozen eval split:** `data/tickets.jsonl` is the only test set; never train on it.
