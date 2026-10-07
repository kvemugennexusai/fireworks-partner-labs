"""projects/03-rag/run.py

RAG: ground the drafted response in policy. Chunk data/policies.md, embed the
chunks with a Fireworks embeddings model, retrieve the top-k policy snippet per
ticket, and draft a response. We compare two conditions head to head:

  ungrounded : draft from the ticket alone
  grounded   : draft from the ticket + retrieved policy snippet

Both drafts are scored by an LLM judge (shared/eval.py --task judge) against each
ticket's reference answer. The metric is mean judge score (1-5), grounded vs
ungrounded. We also report retrieval@1 (did the top hit match the hand-labeled
relevant policy section?).

    python run.py
"""
import datetime
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from shared.fw_client import client  # noqa: E402

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"          # 768-d; bare HF-style id
DRAFT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
JUDGE_MODEL = "accounts/fireworks/models/gpt-oss-120b"  # different model = no self-grading
TOP_K = 2          # snippets fed to the grounded drafter
MAX_TOKENS = 1024  # reasoning drafter needs room (see project 00)
FW_URL = "https://api.fireworks.ai/inference/v1"

UNGROUNDED_SYS = ("You are a customer-support agent. Draft a concise, helpful first "
                  "response to the ticket.")
GROUNDED_SYS = ("You are a customer-support agent. Draft a concise, helpful first "
                "response grounded ONLY in the POLICY provided. Use its specifics "
                "(timeframes, steps). If the policy does not cover the ticket, say you "
                "will follow up rather than inventing details.")

# Hand-labeled relevant policy section per ticket (0-based over the ## chunks below),
# or None where the knowledge base genuinely doesn't cover the ticket (honesty test).
# chunks: 0 Refunds, 1 Plan changes, 2 Invoices, 3 Account, 4 Authentication,
#         5 Rate limits, 6 SSO, 7 Mobile app
GOLD_SECTION = {"t01": 0, "t02": 7, "t03": 3, "t04": 5, "t05": 2, "t06": 4,
                "t07": 1, "t08": None, "t09": None, "t10": 3, "t11": 1, "t12": 6}


def load_tickets():
    with open(os.path.join(ROOT, "data", "tickets.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_policies():
    """Split policies.md on '## ' headings -> one chunk per section."""
    text = open(os.path.join(ROOT, "data", "policies.md")).read()
    chunks = []
    for part in text.split("\n## ")[1:]:           # drop the top-level '# ' title
        title, _, body = part.partition("\n")
        chunks.append({"title": title.strip(), "text": "## " + title.strip() + "\n" + body.strip()})
    return chunks


def embed(texts):
    r = client().embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in r.data])


def cosine_topk(qvec, mat, k):
    sims = mat @ qvec / (np.linalg.norm(mat, axis=1) * np.linalg.norm(qvec) + 1e-9)
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def run_judge(data_path, out_path, system):
    """Reuse shared/eval.py to draft (with `system`) + judge; return its summary."""
    cmd = [sys.executable, os.path.join(ROOT, "shared", "eval.py"),
           "--base-url", FW_URL, "--api-key", os.environ["FIREWORKS_API_KEY"],
           "--model", DRAFT_MODEL, "--data", data_path, "--task", "judge",
           "--system", system, "--max-tokens", str(MAX_TOKENS), "--temperature", "0",
           "--judge-base-url", FW_URL, "--judge-model", JUDGE_MODEL,
           "--judge-max-tokens", "1024", "--concurrency", "4", "--output", out_path]
    subprocess.run(cmd, check=True)
    return json.load(open(out_path))["summary"]


def main():
    tickets = load_tickets()
    chunks = chunk_policies()
    print(f"chunked policies into {len(chunks)} sections")

    cmat = embed([c["text"] for c in chunks])
    qmat = embed([t["input"] for t in tickets])

    # --- retrieval + build the two eval files ---
    ungrounded_rows, grounded_rows = [], []
    hits = considered = 0
    for i, tk in enumerate(tickets):
        top = cosine_topk(qmat[i], cmat, TOP_K)
        top_idx = top[0][0]
        gold = GOLD_SECTION[tk["id"]]
        if gold is not None:
            considered += 1
            hits += (top_idx == gold)
        snippet = "\n\n".join(chunks[j]["text"] for j, _ in top)
        ungrounded_rows.append({"input": tk["input"], "reference": tk.get("reference")})
        grounded_rows.append({"input": f"POLICY:\n{snippet}\n\nTICKET:\n{tk['input']}",
                              "reference": tk.get("reference")})

    retrieval_at1 = round(100 * hits / considered, 1)
    print(f"retrieval@1 = {retrieval_at1}%  ({hits}/{considered} tickets with a "
          f"relevant section; t08/t09 have none by design)")

    ung_path = os.path.join(HERE, "_ungrounded.jsonl")
    grd_path = os.path.join(HERE, "_grounded.jsonl")
    with open(ung_path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in ungrounded_rows))
    with open(grd_path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in grounded_rows))
    # --- draft + judge each condition via shared/eval.py ---
    print("\n--- ungrounded (ticket only) ---")
    ung = run_judge(ung_path, os.path.join(HERE, "_ungrounded_eval.json"), UNGROUNDED_SYS)
    print("\n--- grounded (ticket + retrieved policy) ---")
    grd = run_judge(grd_path, os.path.join(HERE, "_grounded_eval.json"), GROUNDED_SYS)

    rows = [
        {"model": f"{DRAFT_MODEL.split('/')[-1]} (ungrounded)",
         "value": ung.get("mean_score", 0),
         "notes": f"mean judge 1-5 over {ung.get('scored',0)} tickets, ticket only"},
        {"model": f"{DRAFT_MODEL.split('/')[-1]} (grounded RAG)",
         "value": grd.get("mean_score", 0),
         "notes": (f"mean judge 1-5 over {grd.get('scored',0)} tickets; "
                   f"embed={EMBED_MODEL}, top-{TOP_K}, retrieval@1={retrieval_at1}%")},
    ]
    result = {
        "project": "03-rag",
        "metric": "answer quality (LLM-judge, grounded vs ungrounded)", "unit": "1-5",
        "rows": rows,
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nungrounded {rows[0]['value']}  ->  grounded {rows[1]['value']}   "
          f"(judge=gpt-oss-120b)\nwrote results.json")


if __name__ == "__main__":
    main()
