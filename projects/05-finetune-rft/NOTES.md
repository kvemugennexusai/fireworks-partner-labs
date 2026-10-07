# NOTES — 05 finetune-rft

> Raw material for the blog — candid about what broke and how it was fixed.
> This one fought back even harder than 04. The fight was all tooling, not ML.

## Goal
Reinforcement fine-tuning: write a reward function, run an RFT job against it, and
compare the result to the SFT model from project 04 on the same held-out tickets.
Same task as SFT (intent classification) so SFT-vs-RFT is apples-to-apples.

## Result (the headline)
| model | accuracy (12 held-out) |
|---|---|
| support-copilot-sft-v2 (SFT / imitation) | 83.3% (10/12) |
| support-copilot-rft (RFT / reward-optimized) | **100% (12/12)** |

**RFT beat SFT by +16.7 points and fixed the EXACT two cases SFT missed** — t11
(billing) and t12 (technical), both of which SFT mis-routed to "account." RFT also
de-biased the prediction mix (SFT leaned 5/12 "account"; RFT is 4/4/3/1, matching the
true distribution). The training reward curve climbed 0.66 → 0.85 over 5 epochs.
Caveat: 100% is one sample on 12 tickets and these models are nondeterministic — the
robust claim is "RFT clearly > SFT," not "RFT is perfect."

## The blocker chain (this is the real story)
Creating the *evaluator* (the reward) is where everything fought back. Four dead ends
before the working path:
1. **`reward-kit deploy`** (latest, 0.4.1) → `501 CreateEvaluator is deprecated; please
   use CreateEvaluatorV2 or the UI flow`.
2. **`fireworks` SDK 0.19.20 gateway** (`create_evaluator_sync`) → same gRPC method,
   `UNIMPLEMENTED`. The generated stub has no `CreateEvaluatorV2`.
3. **firectl** → no evaluator-create verb at all (only `evaluator-revision` get/list).
4. **The UI** → rejects reward-kit's reward: *"Code must use the @evaluation_test
   decorator."*
5. **The working path: eval-protocol.** The V2 evaluator format is eval-protocol's
   `@evaluation_test` (a `(row)->row` scorer that sets `row.evaluation_result`), and its
   `ep` CLI talks to the V2 API: `ep upload` creates the evaluator, `ep create rft`
   launches the job. Rewrote the reward in that format (`evaluator/test_support_intent.py`)
   and it went through. `reward/main.py` (the reward-kit version) is kept as the record
   of the dead end.

## Bugs & fixes
| # | symptom | cause | fix |
|---|---------|-------|-----|
| 1 | every evaluator-create API/SDK/CLI fails (501 / UNIMPLEMENTED / no verb) | server deprecated V1 `CreateEvaluator`; public tooling (reward-kit, firectl, fireworks-ai 0.19.20) hasn't shipped V2 | use **eval-protocol** (`@evaluation_test` + `ep upload`/`ep create rft`), which targets the V2 API |
| 2 | UI rejects pasted reward | wrong framework — `@reward_function` (reward-kit) vs `@evaluation_test` (eval-protocol) | rewrite the reward as an eval-protocol test with `EvaluationRow`/`EvaluateResult` |
| 3 | `ep upload` ignored `--evaluator-id`; named it `test-support-intent-test-support-intent` | id is derived from `file-stem + function-name` | accept the generated name; reference it in `ep create rft --evaluator` |
| 4 | `ep create rft` → `error parsing output model name` | the SDK path requires a **fully-qualified** output model (firectl accepted a short name) | pass `accounts/<acct>/models/support-copilot-rft` |
| 5 | **RFT job sat in `JOB_STATE_PAUSED` ~14 min** mid-run | the job pauses (looks like a transient rollout/inference-capacity hiccup) and does NOT auto-resume on this account | `firectl resume reinforcement-fine-tuning-job <id>` kicked it back to RUNNING; my poll loop now auto-resumes on PAUSED |
| 6 | installing reward-kit/eval-protocol churned deps | reward-kit pinned `openai==1.78` (down from 2.41); eval-protocol then pulled `fireworks-ai 1.2.0a79`, litellm, etc. | verified shared code still runs on openai 1.78; **these training tools really want their own venv** — noted for reproducibility |
| 7 | side effects of `ep upload` | it tars the whole repo to GCS and uploads `.env` secrets to the Fireworks secret store | gitignore `*.tar.gz`; be aware the API key is now a platform secret (expected, but worth knowing) |

## Cost
- **Training:** RFT ran ~35 min wall clock end-to-end (5 epochs of rollouts + reward
  scoring + the 14-min pause), vs ~5 min for SFT. RFT is much heavier — it generates and
  scores hundreds of rollouts per epoch (768 rollouts/epoch here).
- **Deployment (the real bill, again):** the eval needs the base on-demand (H200) with
  both addons loaded. I tore the FIRST eval deployment down the moment the RFT job paused
  (it had idled ~25 min while paused — pure waste) and stood up a fresh one only after the
  job completed. **Lesson reinforced: never bring up the eval deployment until the model
  actually exists.** Pricing TODO still tracked for project 08.

## Learnings (2-3)
- **RFT > SFT when the reward is the metric you actually care about.** Imitation (SFT)
  copies labels; reward optimization (RFT) optimizes correctness directly, and here it
  closed the exact boundary cases SFT couldn't (t11/t12) and fixed SFT's class bias. On a
  verifiable task, RFT was worth the extra cost.
- **The hard part of RFT on Fireworks today is tooling-version skew, not RL.** The reward
  was ~20 lines; getting an evaluator *created* took four failed tools and a framework
  switch (reward-kit → eval-protocol). If you hit `CreateEvaluator is deprecated`, stop
  fighting reward-kit/firectl and go straight to eval-protocol's `ep` CLI.
- **RFT jobs can silently PAUSE and need a manual `resume`.** Watch for it, automate the
  resume, and keep your eval deployment DOWN while the job runs/pauses or you'll pay for an
  idle GPU.
