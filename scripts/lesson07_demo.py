"""Lesson 7 demo: PII guardrails, three ways, with your real Foundry model.

Run:  uv run python scripts/lesson07_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402

from loanlens.investigator_v2 import build_agent, make_request  # noqa: E402
from loanlens.pii import redact_at_boundary, redact_pii  # noqa: E402
from loanlens.pipeline import build_graph  # noqa: E402

RAW = (
    "Hi, I'm Priya Shah, 34, nurse on £42,000 a year, want £8,000 for a car. "
    "NI number AB 12 34 56 C. Pay into sort code 20-45-77, account 41234567. "
    "Email priya.shah@example.co.uk"
)

print("══ 1. The pipeline (redact_pii is its first node) ══")
final = build_graph().invoke({"messages": [HumanMessage(RAW)]})
print("message in state:", final["messages"][0].content)
print("extracted:       ", final["application"])
print("outcome:         ", final["outcome"])

print("\n══ 2. Why ALSO redact at the boundary? Look inside the checkpoints ══")
mini = StateGraph(MessagesState)
mini.add_node(redact_pii)
mini.add_edge(START, "redact_pii")
mini.add_edge("redact_pii", END)
saver = InMemorySaver()
graph = mini.compile(checkpointer=saver)

for label, first_message in [("in-graph only", HumanMessage(RAW)), ("boundary first", redact_at_boundary(RAW))]:
    cfg = {"configurable": {"thread_id": label}}
    graph.invoke({"messages": [first_message]}, cfg)
    leaked = [s.metadata["step"] for s in graph.get_state_history(cfg)
              if any("41234567" in str(m.content) for m in s.values.get("messages", []))]
    print(f"{label:>15}: checkpoints holding the raw account number → steps {leaked or 'none'}")

print("\n══ 3. The investigator: bureau returns bank details, the model must not see them ══")
sam = {"applicant_name": "Sam Jones", "age": 29, "annual_income": 33_600, "loan_amount": 20_000,
       "employment_status": "self_employed"}
msgs = build_agent().invoke({"messages": [make_request(sam)]}, {"recursion_limit": 100})["messages"]
for m in msgs:
    if m.type == "tool" and m.name == "lookup_credit_file":
        print("credit file as the model saw it:", m.content)
print("final memo:\n" + msgs[-1].content)
