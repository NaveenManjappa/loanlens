"""Lesson 2 demo: watch three checks run in the same super-step, then fan in.

Run (after your tests pass):  uv run python scripts/lesson02_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loanlens.screening import build_graph  # noqa: E402

app = {
    "applicant_name": "Sam Jones",
    "age": 17,
    "annual_income": 50_000,
    "loan_amount": 40_000,
    "employment_status": "self_employed",
}
graph = build_graph()

print("── Updates, one node at a time ──")
for step in graph.stream({"application": app}, stream_mode="updates"):
    for node, update in step.items():
        print(f"{node:>20} -> {update}")

final = graph.invoke({"application": app, "messages": [("user", "Please be quick, thanks!")]})
print("\n── risk_flags after the reducer merged them ──")
print(final["risk_flags"])
print("\n── messages (appended by add_messages, not overwritten) ──")
for m in final["messages"]:
    print(f"{m.type:>6}: {m.content}")
