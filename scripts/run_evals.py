"""Lesson 11: run the LoanLens eval suite against your REAL model and print a scorecard.

    uv run python scripts/run_evals.py              # every case × 2 repeats
    uv run python scripts/run_evals.py 3            # every case × 3 repeats
    uv run python scripts/run_evals.py 1 minor      # one case, once

Optional, traces in LangSmith: set LANGSMITH_TRACING=true, LANGSMITH_API_KEY and either
LANGSMITH_PROJECT (project-based workspace) or LANGSMITH_AGENT_ID + LANGSMITH_AGENT_ENVIRONMENT
(agent-based workspace) in .env. Every run is tagged "eval" with metadata eval_case=<id>.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from loanlens.evals import EVALUATORS, load_dataset, run_case, summarise  # noqa: E402
from loanlens.pipeline import build_graph  # noqa: E402

repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 2
only = sys.argv[2] if len(sys.argv) > 2 else None
cases = [c for c in load_dataset() if only in (None, c["id"])]

results = []
print(f"{'case':<20} {'run':>3}  {'outcome':<22} " + " ".join(f"{e.__name__[:12]:>12}" for e in EVALUATORS) + "   tokens  calls")
for case in cases:
    for i in range(repeats):
        r = run_case(case, build_graph)
        scores = {e.__name__: e(case, r) for e in EVALUATORS}
        results.append({"case_id": case["id"], "scores": scores, "total_tokens": r["total_tokens"],
                        "model_calls": r["model_calls"]})
        cells = " ".join(f"{ {True: '✓', False: '✗', None: '·'}[v]:>12}" for v in scores.values())
        print(f"{case['id']:<20} {i + 1:>3}  {str(r['outcome']):<22} {cells}   {r['total_tokens']:>6}  {r['model_calls']:>5}")

print("\n── pass rates (✓ / (✓+✗), '·' = not applicable) ──")
for name, rate in summarise(results).items():
    print(f"  {name:<28} {rate:.0%}")
by_case = {}
for res in results:
    by_case.setdefault(res["case_id"], []).append(res["total_tokens"])
print("\n── average tokens per case ──")
for cid, toks in by_case.items():
    print(f"  {cid:<20} {sum(toks) / len(toks):>8.0f}")
