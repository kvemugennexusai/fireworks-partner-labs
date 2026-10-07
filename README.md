# Fireworks Partner Labs

A portfolio of small projects that exercise the **Fireworks AI** platform end to end,
all built around one use case — a **Support Copilot** — to see what the platform is
actually like to build on. Each project isolates one capability, produces a results
table, and logs the bugs and fixes along the way.

Two outputs:
1. **This repo** — runnable projects + aggregated results.
2. **A blog post** (`blog/`) summarizing the learnings of building on Fireworks.

> Built largely with Claude Code. The build spec is in [`CLAUDE.md`](./CLAUDE.md).

## The use case

Given a support ticket → classify intent → retrieve the governing policy (RAG) →
draft a grounded response → optionally call a tool (refund / escalate). One use case,
driven through every Fireworks capability.

## Capability matrix

| #  | Project | What it exercises | Output |
|----|---------|-------------------|--------|
| 00 | [serverless-baseline](projects/00-serverless-baseline) | Serverless inference, all catalog models | accuracy + latency table |
| 01 | [structured-output](projects/01-structured-output) | JSON-schema structured output | valid-JSON rate |
| 02 | [function-calling](projects/02-function-calling) | Tool calls | tool-selection accuracy |
| 03 | [rag](projects/03-rag) | Embeddings + retrieval | grounded-answer quality |
| 04 | [finetune-sft](projects/04-finetune-sft) | FireOptimizer SFT vs base | eval lift |
| 05 | [finetune-rft](projects/05-finetune-rft) | RFT + eval-protocol vs SFT | eval lift |
| 06 | [evals](projects/06-evals) | Eval Protocol (accuracy + judge) | the shared gate |
| 07 | [multi-lora](projects/07-multi-lora) | Multi-LoRA, per-tenant | cost-per-tenant model |
| 08 | [serving-perf](projects/08-serving-perf) | Latency / throughput / cost | perf table |
| 09 | [vision](projects/09-vision) | Vision model on a screenshot | extraction accuracy |

**Status: all 10 projects complete.** Highlights: gpt-oss-120b tops accuracy (100%) *and*
serving (828 tok/s, 1.4s TTFT); SFT lifted an 8B +16.6 pts and RFT then reached 100% on the
held-out split; multi-LoRA is ~90% cheaper per tenant at 10 tenants; vision read error
screenshots at 100% extraction. Full story in the learnings + blog draft below.

Master results: [`results/RESULTS.md`](results/RESULTS.md) · Learnings:
[`results/LEARNINGS.md`](results/LEARNINGS.md) · Blog draft: [`blog/DRAFT.md`](blog/DRAFT.md)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your FIREWORKS_API_KEY (never commit .env)
export $(grep -v '^#' .env | xargs)

# run a project
cd projects/00-serverless-baseline && python run.py

# regenerate the master results table from all projects
python shared/aggregate.py
```

## Layout

```
shared/      reusable client, benchmark, eval, and results aggregator
data/        synthetic tickets + policy KB (public, reproducible)
projects/    one folder per capability (code + results.json + NOTES.md)
results/     RESULTS.md (master table) + LEARNINGS.md (cross-cutting)
blog/        the write-up
```

## Notes

Not affiliated with or endorsed by Fireworks AI. Model catalogs change — each project
lists available models at runtime. Costs are from published pricing and marked when estimated.
