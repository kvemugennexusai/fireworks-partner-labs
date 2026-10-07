"""projects/05-finetune-rft/reward/main.py

Reward function for RFT on the intent-classification task (same task as the SFT
model in project 04, so SFT-vs-RFT is apples-to-apples). Verifiable reward:

  correct label   -> 1.0
  valid label, wrong -> 0.1   (small shaping so "emit a clean label" is learned
                               first; constant across the 4 classes, so it can't
                               be hacked by collapsing to one label)
  no valid label  -> 0.0

Deployed as a Fireworks evaluator via `reward-kit deploy`.
"""
import re

from reward_kit import EvaluateResult, MetricResult, reward_function

LABELS = ["billing", "technical", "account", "other"]


def _norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def _content(m):
    return m.get("content") if isinstance(m, dict) else getattr(m, "content", "")


def _extract(text):
    """Pull the predicted label: scan from the last non-empty line, prefer the
    label appearing latest (handles any reasoning that leaks into the answer)."""
    for line in reversed([ln for ln in (text or "").splitlines() if ln.strip()]):
        t = _norm(line)
        hits = [(t.rfind(_norm(L)), L) for L in LABELS if _norm(L) in t]
        if hits:
            return max(hits)[1]
    return None


@reward_function(id="support-intent-reward")
def evaluate(messages, ground_truth=None, **kwargs) -> EvaluateResult:
    resp = _content(messages[-1]) if messages else ""
    gold = ground_truth
    if isinstance(ground_truth, list) and ground_truth:        # ground_truth as messages
        gold = _content(ground_truth[-1])
    gold = _norm(gold)

    pred = _extract(resp)
    valid = pred is not None
    correct = valid and _norm(pred) == gold
    score = 1.0 if correct else (0.1 if valid else 0.0)

    return EvaluateResult(
        score=score,
        reason=f"pred={pred!r} gold={gold!r} -> {'correct' if correct else ('wrong' if valid else 'unparseable')}",
        metrics={
            "correctness": MetricResult(score=1.0 if correct else 0.0, is_score_valid=correct,
                                        reason=f"pred={pred} gold={gold}"),
            "format_valid": MetricResult(score=1.0 if valid else 0.0, is_score_valid=valid,
                                         reason="emitted a recognizable label" if valid else "no label found"),
        },
    )
