# 07 — Multi-LoRA
**Capability:** multiple LoRA adapters on one base deployment, per-tenant economics.
**Build:** train 2-3 small adapters (e.g. distinct tones/tenants), serve them on a single
base, route per request, and build a **per-tenant cost model** (one amortized base vs. N
dedicated deployments).
**Definition of done:** `results.json` (cost-per-tenant table + adapter behavior); `NOTES.md`
on adapter loading, routing, and the economics.

## Reproduce
```bash
# 1. two tenant tone adapters (formal "acme", friendly "bloom"), trained on a NEUTRAL prompt
python generate_tone_data.py
for t in acme bloom; do
  firectl dataset create tone-$t-train ./train_$t.jsonl
  firectl create supervised-fine-tuning-job \
    --base-model accounts/fireworks/models/llama-v3p1-8b-instruct \
    --dataset tone-$t-train --output-model tone-$t \
    --epochs 4 --lora-rank 16 --batch-size-samples 8
done

# 2. one base, BOTH adapters loaded -> route per request (set DEPLOYMENT in run.py)
firectl create deployment accounts/fireworks/models/llama-v3p1-8b-instruct \
  --deployment-shape accounts/fireworks/deploymentShapes/rft-llama-v3p1-8b-instruct --enable-addons
firectl load-lora tone-acme  --deployment <DEP>
firectl load-lora tone-bloom --deployment <DEP>
python run.py            # same ticket -> acme (formal) vs bloom (friendly)
firectl delete deployment <DEP> --ignore-checks

# 3. the economics (no GPU needed)
python cost_model.py     # per-tenant $/mo: multi-LoRA (amortized) vs N dedicated
```
