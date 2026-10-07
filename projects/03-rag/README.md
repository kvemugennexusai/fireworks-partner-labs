# 03 — RAG
**Capability:** embeddings + retrieval grounding.
**Build:** chunk `data/policies.md`, embed with a Fireworks embeddings model, retrieve
top-k per ticket, and draft a grounded response. Use `shared/eval.py --task judge` to score
grounded vs. ungrounded answers.
**Definition of done:** `results.json` (judge score: grounded vs. ungrounded); `NOTES.md`
on chunking, embedding model choice, and retrieval quality.
