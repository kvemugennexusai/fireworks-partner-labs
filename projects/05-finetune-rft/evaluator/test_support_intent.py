"""projects/05-finetune-rft/evaluator/test_support_intent.py

Fireworks evaluator (reward) for RFT on intent classification, in the eval-protocol
`@evaluation_test` format the platform's V2 evaluator API requires. (The reward-kit
`@reward_function` form is rejected by the current server with "Code must use the
@evaluation_test decorator" — see NOTES.md.)

Reward: correct label -> 1.0, valid label but wrong -> 0.1, no valid label -> 0.0.
The 0.1 is constant across the 4 classes, so it can't be hacked by collapsing to one.

Upload as an evaluator:   ep upload --entry projects/05-finetune-rft/evaluator/test_support_intent.py::test_support_intent -y
"""
import re
from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, EvaluationRow, Message, MetricResult
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.pytest.evaluation_test import evaluation_test

LABELS = ["billing", "technical", "account", "other"]


def _norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def _content(c):
    if isinstance(c, list):
        return "".join(getattr(p, "text", "") for p in c)
    return str(c or "")


def _extract(text):
    for line in reversed([ln for ln in (text or "").splitlines() if ln.strip()]):
        t = _norm(line)
        hits = [(t.rfind(_norm(L)), L) for L in LABELS if _norm(L) in t]
        if hits:
            return max(hits)[1]
    return None


def support_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    """rft_train.jsonl rows are {messages:[system,user], ground_truth:label}."""
    out = []
    for r in rows:
        msgs = [Message(**m) for m in (r.get("messages") or [])]
        out.append(EvaluationRow(messages=msgs, ground_truth=r.get("ground_truth")))
    return out


@evaluation_test(
    input_dataset=["projects/05-finetune-rft/rft_train.jsonl"],
    dataset_adapter=support_dataset_adapter,
    completion_params=[{"max_tokens": 1024,
                        "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b"}],
    rollout_processor=SingleTurnRolloutProcessor(),
    aggregation_method="mean",
    passed_threshold=0.7,
    mode="pointwise",
)
def test_support_intent(row: EvaluationRow) -> EvaluationRow:
    assistant = [m for m in row.messages if m.role == "assistant"]
    resp = _content(assistant[-1].content) if assistant else ""
    gold = _norm(str(row.ground_truth))
    pred = _extract(resp)
    valid = pred is not None
    correct = valid and _norm(pred) == gold
    score = 1.0 if correct else (0.1 if valid else 0.0)

    row.evaluation_result = EvaluateResult(
        score=score,
        is_score_valid=valid,
        reason=f"pred={pred!r} gold={gold!r}",
        metrics={
            "correctness": MetricResult(score=1.0 if correct else 0.0,
                                        is_score_valid=correct, reason=f"pred={pred} gold={gold}"),
            "format_valid": MetricResult(score=1.0 if valid else 0.0,
                                         is_score_valid=valid, reason="label found" if valid else "no label"),
        },
    )
    return row
