# NOTES — 04 finetune-sft

> Raw material for the blog — candid about what broke and how it was fixed.
> This project fought back the most. Good.

## Goal
FireOptimizer supervised fine-tuning: train a LoRA adapter on labeled tickets, deploy
it, and compare it against its base on held-out data. Metric: intent accuracy, base
vs SFT.

## Result (the headline)
| model | accuracy (12 held-out) |
|---|---|
| llama-v3p1-8b-instruct (base) | 66.7% (8/12) |
| support-copilot-sft **v1** (token batching, 3 epochs) | **16.7%** — collapsed to one class |
| support-copilot-sft **v2** (sample batching, 5 epochs, r16) | **83.3%** (10/12) |

Net: a correctly-trained LoRA lifted a small 8B base **+16.6 points** — but only after
v1 failed loudly and taught me why.

## Steps taken
1. Verified fine-tuning is enabled (`firectl` signed in; prior banking77 addons exist).
   Tunable base confirmed from a prior job: `llama-v3p1-8b-instruct`.
2. **Built a real train/test split.** The repo's 12 tickets are too few and would be
   train==test leakage, so `generate_data.py` synthesizes 96 labeled tickets (24/intent,
   gpt-oss-120b, schema-constrained) and de-dupes against the eval set. The 12 originals
   stay as held-out eval.
3. Uploaded the dataset (`firectl dataset create`), launched SFT (`create
   supervised-fine-tuning-job`), deployed the base on-demand with addons, loaded the
   LoRA (`load-lora`), and ran `shared/eval.py --task classification` against base and
   SFT model ids.

## Bugs & fixes (this is the blog gold)
| # | symptom | cause | fix |
|---|---------|-------|-----|
| 1 | `invalid base model ... not found` (under MY account) | bare `--base-model llama-v3p1-8b-instruct` resolves to `accounts/<me>/...` | use the fully-qualified `accounts/fireworks/models/llama-v3p1-8b-instruct` |
| 2 | base model 404s on the inference endpoint | the base is **not serverless** on this account | stand up an **on-demand deployment** (real GPU cost) to eval against |
| 3 | `create deployment` hangs then `^D` fails | it's **interactive** — prompts to pick a deployment shape | pass `--deployment-shape accounts/fireworks/deploymentShapes/...` explicitly |
| 4 | deployment stuck CREATING, `Message: no available capacity` | the H100-minimal shape pool had no free GPUs | retry on a different pool — the H200 (`rft-...`) shape came up READY in ~2.5 min |
| 5 | both model ids still 404 after deploy | on-demand models are addressed via a **`#<deployment>` suffix**, not the bare id | query `…/llama-v3p1-8b-instruct#accounts/<me>/deployments/<id>` |
| 6 | addon 404 even with `--enable-addons` | `--enable-addons` permits addons but doesn't load yours; and the `DeployedModel` is briefly `DEPLOYING` | `firectl load-lora <model> --deployment <id>`, then wait for state `DEPLOYED` |
| 7 | **SFT v1 made the model WORSE (66.7%→16.7%), collapsing to "other"** | default `--batch-size` is **32768 tokens**; all 96 examples (~7.5k tokens) pack into **one** batch ⇒ 3 epochs = **3 optimizer steps**. Wildly undertrained. | retrain with **`--batch-size-samples 8`** (12 steps/epoch) + `--epochs 5` + `--lora-rank 16` ⇒ ~60 steps ⇒ 83.3% |
| 8 | `delete deployment` refused | safety check: deployment served inference in the last hour | `--ignore-checks` (then verify it scales to 0 / DELETING so billing stops) |

## Cost
- **Training:** v1 ~4 min, v2 ~5 min wall clock (96 examples, ~7.5k tokens). Cheap.
- **The real cost is the deployment, not the training.** On-demand H200 (1×
  NVIDIA_H200_141GB) was up ~44 min (18:36–19:20) across debugging both SFT versions —
  that GPU-time dwarfs the training and the inference. **Lesson: tear the deployment
  down the moment the eval is done** (and prefer scale-to-zero). Exact $ is an estimate
  pending the pricing pull in project 08, but the shape of the bill is clear: GPU
  wall-time for a base that isn't serverless.
- Per-token training/inference pricing TODO tracked for project 08.

## Learnings (2-3)
- **The default token-based batch size silently undertrains small datasets.** "3 epochs"
  sounds fine until you realize it was 3 gradient steps. On a small SFT set, switch to
  `--batch-size-samples` and *count your steps* (examples/batch × epochs). The failure
  mode is a confident model that collapses to one label — looks like a serving bug, is
  actually a training-config bug.
- **Fine-tuning is mostly ops, not ML.** Of the eight things that broke, exactly one was
  about learning; the other seven were namespaces, deployment shapes, capacity, model
  addressing, addon loading, and teardown. Budget accordingly — the `firectl`/deployment
  lifecycle is the hard part, not the loss curve.
- **SFT genuinely pays off when done right:** +16.6 points on a small 8B base from 96
  synthetic examples, beating the base by sharpening exactly the intent task. The two it
  still misses are t11 and t12 (SSO→account) — the same boundary cases that are
  contestable everywhere else in the portfolio.
