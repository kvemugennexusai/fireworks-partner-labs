# 00 — Serverless baseline
**Capability:** serverless inference across every catalog model.
**Build:** classify each ticket's intent on each available model; score accuracy.
**Run:** `python run.py`  (reference implementation already provided)
**Definition of done:** `results.json` with per-model intent accuracy; `NOTES.md` filled,
including any model that was unavailable or behaved oddly (e.g. reasoning models that
won't emit a bare label).
