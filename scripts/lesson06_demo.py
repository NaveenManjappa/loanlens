"""Lesson 6 demo: the create_agent investigator, twice, with your real Foundry model.
  Run 1: normal tools.
  Run 2: the credit bureau is DOWN (raises ConnectionError). Watch: retries, an error the
         model can read, and your apply_policy overruling anything optimistic.

Run:  uv run python scripts/lesson06_demo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_core.tools import tool  # noqa: E402

from loanlens.investigator_v2 import apply_policy, build_agent, make_request  # noqa: E402
from loanlens.tools import TOOLS  # noqa: E402

SAM = {
    "applicant_name": "Sam Jones",
    "age": 29,
    "annual_income": 33_600,
    "loan_amount": 20_000,
    "employment_status": "self_employed",
}


@tool
def lookup_credit_file(applicant_name: str) -> dict:
    """Look up the applicant's credit file at the credit bureau."""
    print("   ⚡ bureau call failed (simulated outage)")
    raise ConnectionError("bureau unreachable")


def show(title, agent):
    print(f"\n════ {title} ════")
    messages = agent.invoke({"messages": [make_request(SAM)]}, {"recursion_limit": 25})["messages"]
    for m in messages[1:]:
        if m.type == "ai" and m.tool_calls:
            for c in m.tool_calls:
                print(f"🤖 call {c['name']}({json.dumps(c['args'])})")
        elif m.type == "tool":
            print(f"🔧 [{m.status}] {m.name}: {m.content[:120]}")
        else:
            print(f"🤖 model says:\n{m.content}")
    print(f"⚖️  after apply_policy: {apply_policy(messages)}")


show("Run 1: all systems up", build_agent())
outage_tools = [lookup_credit_file, *[t for t in TOOLS if t.name != "lookup_credit_file"]]
show("Run 2: credit bureau down", build_agent(tools=outage_tools, retry_initial_delay=0.2))
