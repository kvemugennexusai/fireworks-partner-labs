"""projects/07-multi-lora/run.py

Demonstrate multiple LoRA adapters on ONE base deployment with per-request routing:
the same ticket, routed to two different tenant adapters (acme=formal, bloom=friendly),
produces each tenant's house style — proving per-tenant behavior is baked into the
adapter, not the prompt. Routing is just choosing the adapter model id per request.

Prereq: deploy base with --enable-addons, load both adapters, set DEPLOYMENT below.
Run cost_model.py for the economics half. See README.

    python run.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from shared.fw_client import chat  # noqa: E402

DEPLOYMENT = "accounts/kvemu-o43cdgtt8e51/deployments/qf9vnrr3"
ACCT = "accounts/kvemu-o43cdgtt8e51"
NEUTRAL_SYSTEM = "You are a customer-support agent. Draft a concise first response to the ticket."

# per-tenant routing table: tenant -> their LoRA adapter (all on the SAME deployment)
TENANT_ADAPTER = {
    "acme  (formal)":  f"{ACCT}/models/tone-acme#{DEPLOYMENT}",
    "bloom (friendly)": f"{ACCT}/models/tone-bloom#{DEPLOYMENT}",
}

TICKETS = [
    "I was charged twice for my subscription this month and need a refund.",
    "The mobile app keeps crashing when I open the dashboard.",
]


def route(tenant):
    """'Routing' on a multi-LoRA base is just picking the tenant's adapter id."""
    return TENANT_ADAPTER[tenant]


def main():
    for tk in TICKETS:
        print("=" * 78)
        print("TICKET:", tk)
        for tenant in TENANT_ADAPTER:
            out = chat(route(tenant), [{"role": "system", "content": NEUTRAL_SYSTEM},
                                       {"role": "user", "content": tk}], max_tokens=400)
            print(f"\n[{tenant}] {out.strip()}")
        print()
    print("Same base, same prompt, two tenants -> two house styles. "
          "Run cost_model.py for the per-tenant economics.")


if __name__ == "__main__":
    main()
