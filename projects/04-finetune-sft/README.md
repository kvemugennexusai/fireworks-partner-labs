# 04 — Fine-tune (SFT)
**Capability:** FireOptimizer supervised fine-tuning.
**Build:** format the tickets as OpenAI-style JSONL, run an SFT job on a tunable base
model via `firectl` / the fine-tuning API, deploy it, and compare eval vs. the base.
**Definition of done:** `results.json` (base vs. SFT accuracy/judge); `NOTES.md` with the
dataset format, job id, training time/cost, and every gotcha (this is prime blog material).

## Reproduce
The base model is not serverless on this account, so eval runs against an on-demand
deployment. Full sequence (see NOTES.md for the gotchas behind each step):

```bash
# 1. synthesize the training set (96 examples; 12 repo tickets stay as held-out eval)
python generate_data.py

# 2. upload + train (sample batching is essential — see NOTES bug #7)
firectl dataset create support-sft-train ./train.jsonl
firectl create supervised-fine-tuning-job \
  --base-model accounts/fireworks/models/llama-v3p1-8b-instruct \
  --dataset support-sft-train --output-model support-copilot-sft-v2 \
  --epochs 5 --lora-rank 16 --batch-size-samples 8

# 3. deploy base on-demand (must pass an explicit shape) + load the LoRA addon
firectl create deployment accounts/fireworks/models/llama-v3p1-8b-instruct \
  --deployment-shape accounts/fireworks/deploymentShapes/rft-llama-v3p1-8b-instruct \
  --enable-addons --scale-to-zero-window 20m
firectl load-lora support-copilot-sft-v2 --deployment <DEPLOYMENT_ID>

# 4. eval base vs SFT (set DEPLOYMENT in run.py to <DEPLOYMENT_ID>)
python run.py

# 5. ALWAYS tear down — the deployment is the real cost, not training
firectl delete deployment <DEPLOYMENT_ID> --ignore-checks
```
