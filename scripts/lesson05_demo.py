"""Lesson 5 demo: your real Foundry model investigates a referred application.
Watch the loop: the model asks for tools -> your code runs them -> the model reads the results -> ...

Run:  uv run python scripts/lesson05_demo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_core.messages import HumanMessage  # noqa: E402

from loanlens.investigator import build_graph  # noqa: E402

sam = {
    "applicant_name": "Sam Jones",
    "age": 29,
    "annual_income": 33_600,
    "loan_amount": 20_000,
    "employment_status": "self_employed",
}
graph = build_graph()
inputs = {"application": sam, "messages": [HumanMessage("Investigate this referred application.")]}

# recursion_limit = LangGraph's own backstop on super-steps (its default is ~10,000: far too high to be a real cap)
for step in graph.stream(inputs, {"recursion_limit": 25}, stream_mode="updates"):
    for node, update in step.items():
        for m in update.get("messages", []):
            if m.type == "ai" and m.tool_calls:
                for c in m.tool_calls:
                    print(f"🤖 {node}: call {c['name']}({json.dumps(c['args'])})")
            elif m.type == "tool":
                print(f"🔧 {node}: {m.content}")
            else:
                print(f"\n✅ {node} — final answer:\n{m.content}")
        if "tool_rounds" in update:
            print(f"   (tool_rounds = {update['tool_rounds']})")
