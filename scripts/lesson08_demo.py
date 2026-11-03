"""Lesson 8 demo: a case that survives a PROCESS RESTART, using SQLite and your real model.

Run these as three separate commands. Each is a fresh Python process with no memory of the last:
    uv run python scripts/lesson08_demo.py 1        # Tom applies, forgets his age
    uv run python scripts/lesson08_demo.py 2        # ...later, Tom replies with his age
    uv run python scripts/lesson08_demo.py history  # the audit trail, read straight from the DB
    uv run python scripts/lesson08_demo.py reset    # delete the case DB to start again
"""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from loanlens.cases import continue_application, current_outcome, decisions_made, thread_config  # noqa: E402
from loanlens.pipeline import build_graph  # noqa: E402

DB = ROOT / "loanlens_cases.db"  # git-ignored: it holds case data
CASE = "case-tom-reid"
MESSAGES = {
    "1": "Tom Reid here, employed plumber on about £60k a year. Looking for £8,000.",
    "2": "Sorry, forgot to say: I'm 41.",
}

step = sys.argv[1] if len(sys.argv) > 1 else "1"
if step == "reset":
    DB.unlink(missing_ok=True)
    print("case DB deleted")
    sys.exit()

checkpointer = SqliteSaver(sqlite3.connect(DB, check_same_thread=False))
graph = build_graph(checkpointer=checkpointer)

if step in MESSAGES:
    print(f"case {CASE} before this message: outcome = {current_outcome(graph, CASE)}")
    final = continue_application(graph, CASE, MESSAGES[step])
    print(f"Tom: {MESSAGES[step]}")
    print(f"LoanLens: {final['messages'][-1].content}")
    print(f"outcome now: {final['outcome']}  ·  application: {final['application']}")
elif step == "history":
    print("decisions, oldest first:", decisions_made(graph, CASE))
    print("\nevery checkpoint (newest first):")
    for snap in graph.get_state_history(thread_config(CASE)):
        print(f"  step {snap.metadata['step']:>3} · next={snap.next or '(run finished)'}")
