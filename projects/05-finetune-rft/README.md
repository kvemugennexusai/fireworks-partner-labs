# 05 — Fine-tune (RFT)
**Capability:** reinforcement fine-tuning with reward-kit.
**Build:** write a reward function (reward = the response rubric / judge score from project
06), run an RFT job, and compare to the SFT model from project 04.
**Definition of done:** `results.json` (SFT vs. RFT); `NOTES.md` on the reward function,
reward hacking observed, and when RFT did/didn't beat SFT.

## Reproduce
The evaluator-create path that reward-kit/firectl use is server-deprecated; the working
path is **eval-protocol** (`ep`). See NOTES.md for the full blocker chain.

```bash
# 1. RFT dataset = SFT train set reshaped to {messages:[system,user], ground_truth:label}
#    (built inline; 12 repo tickets stay as held-out eval)
firectl dataset create support-rft-train ./rft_train.jsonl

# 2. create the evaluator (reward) via eval-protocol — NOT reward-kit/firectl/UI
ep upload --entry projects/05-finetune-rft/evaluator/test_support_intent.py::test_support_intent -y

# 3. launch RFT against the evaluator (output-model must be fully-qualified)
ep create rft --dataset support-rft-train \
  --evaluator accounts/<acct>/evaluators/test-support-intent-test-support-intent \
  --training-config-base-model accounts/fireworks/models/llama-v3p1-8b-instruct \
  --training-config-output-model accounts/<acct>/models/support-copilot-rft \
  --training-config-epochs 5 --training-config-lora-rank 16 --training-config-batch-size-samples 8 \
  --skip-validation --ignore-docker -y
# if it stalls: firectl resume reinforcement-fine-tuning-job <id>

# 4. deploy base + load BOTH addons (sft-v2 and rft), set DEPLOYMENT in run.py, eval
firectl create deployment accounts/fireworks/models/llama-v3p1-8b-instruct \
  --deployment-shape accounts/fireworks/deploymentShapes/rft-llama-v3p1-8b-instruct --enable-addons
firectl load-lora support-copilot-sft-v2 --deployment <DEP>
firectl load-lora support-copilot-rft   --deployment <DEP>
python run.py

# 5. tear down (do NOT bring the deployment up until the RFT model exists)
firectl delete deployment <DEP> --ignore-checks
```
