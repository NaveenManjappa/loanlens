"""Lesson 4 demo: three real applications through the whole LoanLens pipeline,
using your Foundry model. Watch each one take a different path.

Run:  uv run python scripts/lesson04_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_core.messages import HumanMessage  # noqa: E402

from loanlens.pipeline import build_graph  # noqa: E402

APPLICATIONS = {
    "complete, salaried": "I'm Priya Shah, 34, a nurse earning £42,000 a year. I'd like to borrow £8,000 for a car.",
    "no age given": "Tom Reid, self-employed plumber, about £60k a year. Looking for £8,000.",
    "a minor": "Hi, I'm Alex Kim, I'm 17 and work part-time (employed) earning £9,000 a year. Can I borrow £2,000?",
}

graph = build_graph()
print("── Graph (paste into https://mermaid.live) ──")
print(graph.get_graph().draw_mermaid())

for label, text in APPLICATIONS.items():
    path, final = [], {}
    # one run, two views: "updates" tells us which node ran, "values" gives the full state
    for mode, chunk in graph.stream({"messages": [HumanMessage(text)]}, stream_mode=["updates", "values"]):
        if mode == "updates":
            path.extend(chunk.keys())
        else:
            final = chunk
    print(f"\n── {label} ──")
    print("path:   ", " → ".join(path))
    print("outcome:", final["outcome"])
    print("reply:  ", final["messages"][-1].content)
