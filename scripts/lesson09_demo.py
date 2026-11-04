"""Lesson 9 demo: the applicant and the underwriter, in separate processes, days apart (in theory).

    uv run python scripts/lesson09_demo.py apply      # Sam applies -> referred -> PAUSED for review
    uv run python scripts/lesson09_demo.py queue      # the underwriter's screen: what's waiting?
    uv run python scripts/lesson09_demo.py bad        # submit a decision with no reason -> rejected, still paused
    uv run python scripts/lesson09_demo.py approve    # ...or: decline
    uv run python scripts/lesson09_demo.py history    # the audit trail
"""

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from loanlens.cases import continue_application, decisions_made, thread_config  # noqa: E402
from loanlens.pipeline import build_graph  # noqa: E402
from loanlens.underwriting import awaiting_underwriter, submit_decision  # noqa: E402

CASE = "case-sam-jones"
graph = build_graph(checkpointer=SqliteSaver(sqlite3.connect(ROOT / "loanlens_cases.db", check_same_thread=False)))
cmd = sys.argv[1] if len(sys.argv) > 1 else "apply"

if cmd == "apply":
    result = continue_application(
        graph, CASE, "Hi, Sam Jones, 29, self-employed web designer, about £40k a year. I'd like £5,000. NI AB 12 34 56 C"
    )
    print("LoanLens to Sam:", result["messages"][-1].content)
    print("paused for underwriter?", "__interrupt__" in result)
elif cmd == "queue":
    print(json.dumps(awaiting_underwriter(graph, CASE), indent=2) if awaiting_underwriter(graph, CASE) else "nothing waiting")
elif cmd in ("bad", "approve", "decline"):
    decision = {
        "bad": {"decision": "approve", "underwriter_id": "uw-017"},
        "approve": {"decision": "approve", "reason": "Two years of accounts verified", "underwriter_id": "uw-017"},
        "decline": {"decision": "decline", "reason": "Income too volatile", "underwriter_id": "uw-017"},
    }[cmd]
    result = submit_decision(graph, CASE, decision)
    if "__interrupt__" in result:
        print("still waiting. Error shown to the underwriter:", awaiting_underwriter(graph, CASE)["error"])
    else:
        print("outcome:", result["outcome"], "| decided by:", result["underwriter_decision"])
        print("LoanLens to Sam:", result["messages"][-1].content)
elif cmd == "history":
    print("decisions:", decisions_made(graph, CASE))
    for snap in graph.get_state_history(thread_config(CASE)):
        waiting = " ⏸ waiting for underwriter" if any(t.interrupts for t in snap.tasks) else ""
        print(f"  step {snap.metadata['step']:>3} · next={snap.next or '(finished)'}{waiting}")
