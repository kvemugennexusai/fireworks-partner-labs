"""projects/07-multi-lora/cost_model.py

Per-tenant economics of multi-LoRA serving: one amortized base hosting N tenant
adapters vs N dedicated base deployments. LoRA adapters are tiny (rank-16 on an 8B
base is a few MB) and add negligible serving cost, so the comparison is essentially
"how many GPUs am I renting."

  multi-LoRA per-tenant $/mo = ceil(N / adapters_per_base) * gpu_monthly / N
  dedicated  per-tenant $/mo = gpu_monthly                       (one base each)

Per-token inference cost is ~identical per tenant in both designs (same requests), so
it cancels in the comparison and is omitted. The structural result (savings = 1 - 1/N
until a base saturates) is independent of the exact GPU rate.

    python cost_model.py
"""
import datetime
import json
import math
import os

# --- assumptions (clearly marked) ---
# Account billing showed $0.00 (trial credits), so this is the PUBLISHED on-demand
# rate for the 1x H200-141GB shape we actually deployed (project 04/05), marked as an
# ESTIMATE to verify against the pricing page in project 08. The conclusion does not
# depend on the exact number.
GPU_HOURLY_USD = 8.99          # ESTIMATE, 1x NVIDIA_H200_141GB on-demand
HOURS_PER_MONTH = 730
ADAPTERS_PER_BASE = 100        # Fireworks multi-LoRA hosts many addons per deployment;
                               # real ceiling is QPS/VRAM — 100 is a working assumption
TENANT_COUNTS = [1, 2, 5, 10, 25, 50, 100]

GPU_MONTHLY = GPU_HOURLY_USD * HOURS_PER_MONTH


def model():
    table = []
    for n in TENANT_COUNTS:
        bases = math.ceil(n / ADAPTERS_PER_BASE)
        multi_pt = bases * GPU_MONTHLY / n
        dedic_pt = GPU_MONTHLY
        savings = round(100 * (dedic_pt - multi_pt) / dedic_pt, 1)
        table.append({"tenants": n, "multilora_per_tenant": round(multi_pt, 2),
                      "dedicated_per_tenant": round(dedic_pt, 2), "savings_pct": savings,
                      "bases_multilora": bases})
    return table


def main():
    table = model()
    print(f"GPU: 1x H200-141GB @ ${GPU_HOURLY_USD}/hr (ESTIMATE) = ${GPU_MONTHLY:,.0f}/mo, "
          f"{ADAPTERS_PER_BASE} adapters/base\n")
    print(f"{'tenants':>7} {'multi-LoRA $/tenant':>20} {'dedicated $/tenant':>19} {'savings':>8}")
    for r in table:
        print(f"{r['tenants']:>7} {r['multilora_per_tenant']:>20,.2f} "
              f"{r['dedicated_per_tenant']:>19,.2f} {r['savings_pct']:>7}%")

    rows = [{"model": f"{r['tenants']} tenants", "value": r["multilora_per_tenant"],
             "notes": (f"multi-LoRA ${r['multilora_per_tenant']:,.0f}/tenant/mo vs dedicated "
                       f"${r['dedicated_per_tenant']:,.0f} -> {r['savings_pct']}% cheaper "
                       f"({r['bases_multilora']} base(s))")}
            for r in table]
    result = {
        "project": "07-multi-lora",
        "metric": "multi-LoRA cost per tenant ($/mo, vs dedicated)", "unit": "USD",
        "rows": rows,
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
