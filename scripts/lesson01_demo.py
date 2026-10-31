"""Lesson 1 demo: draw your graph and watch state flow through it.

Run (after your tests pass):  uv run python scripts/lesson01_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loanlens.intake import build_graph  # noqa: E402

graph = build_graph()

print("── Your graph (paste into https://mermaid.live to see it) ──")
print(graph.get_graph().draw_mermaid())

app = {
    "applicant_name": "Priya Shah",
    "age": 34,
    "annual_income": 50_000,
    "loan_amount": 25_000,
    "employment_status": "employed",
}

print("── State updates, one node at a time ──")
for step in graph.stream({"application": app}, stream_mode="updates"):
    for node, update in step.items():
        print(f"{node:>22} returned {update}")

print("── Final state ──")
print(graph.invoke({"application": app}))
