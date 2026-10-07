"""projects/05-finetune-rft/deploy_evaluator.py

Register the reward function (reward/main.py) as a Fireworks *evaluator* so the RFT
job can reference it via --evaluator.

NOTE: the `reward-kit deploy` CLI (v0.4.1) hits the deprecated CreateEvaluator API
(501 "use CreateEvaluatorV2 or the UI flow"), and firectl has no evaluator-create
verb. The working path is the modern `fireworks` SDK gateway, which uses the V2
gRPC call. See NOTES.md.

    python deploy_evaluator.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward"))
from main import evaluate  # noqa: E402  the @reward_function

from fireworks.gateway import Gateway          # noqa: E402
from fireworks.evaluator import Evaluator      # noqa: E402

if __name__ == "__main__":
    gw = Gateway(api_key=os.environ["FIREWORKS_API_KEY"])
    ev = Evaluator(gw, evaluate).sync()
    print("evaluator name:", ev.name)
    print("evaluator id  :", ev.id)
