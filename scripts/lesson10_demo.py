"""Lesson 10 demo: two specialist agents inside the pipeline, with your real Foundry model.

Run:  uv run python scripts/lesson10_demo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from loanlens.cases import continue_application  # noqa: E402
from loanlens.pipeline import build_graph  # noqa: E402
from loanlens.underwriting import awaiting_underwriter  # noqa: E402

graph = build_graph(checkpointer=InMemorySaver())  # model=None -> real model + real specialists

print("══ Sam: self-employed → referred → specialists → paused for underwriter ══")
path = []
for step in graph.stream(
    {"messages": [("user", "Sam Jones, 29, self-employed web designer, about £40k a year. I'd like £5,000.")]},
    {"configurable": {"thread_id": "sam"}},
    stream_mode="updates",
):
    path.extend(k for k in step if k != "__interrupt__")
print("path:", " → ".join(path))
for name, report in awaiting_underwriter(graph, "sam")["specialist_reports"].items():
    print(f"\n[{name}] verdict: {report['verdict']}   tools: {report['tools_used']}")
    print(report["memo"])

print("\n══ Ivan: employed, no risk flags ══")
final = continue_application(graph, "ivan", "Ivan Petrov, 45, employed engineer, £70,000 a year, want £10,000.")
print("outcome:", final["outcome"])
print("specialist reports:", json.dumps(final.get("specialist_reports", {})))
print("(Look at compliance.py's _SANCTIONS_LIST, then answer the lesson's closing question.)")
