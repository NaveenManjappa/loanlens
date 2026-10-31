"""Lesson 3 demo: a REAL call to your Foundry model, then feed the result
into your Lesson 2 screening graph. Two graphs, glued by hand (Lesson 4 joins them).

Run:  uv run python scripts/lesson03_demo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langchain_core.messages import HumanMessage  # noqa: E402

from loanlens.extraction import build_graph as build_extraction  # noqa: E402
from loanlens.screening import build_graph as build_screening  # noqa: E402

email = HumanMessage(
    "Hello, my name is Sam Jones and I'm 29. I'm self-employed (web design) and "
    "take home roughly £2,800 a month. I'd like a loan of £20,000 for a van."
)

print("── 1. Applicant's message ──\n" + email.content)

extracted = build_extraction().invoke({"messages": [email]})["application"]
print("\n── 2. Structured output from the LLM ──")
print(json.dumps(extracted, indent=2))

if any(v is None for v in extracted.values()):
    print("\n(Some fields are None, so screening would need them. That's Lesson 4's routing.)")
else:
    screened = build_screening().invoke({"application": extracted})
    print("\n── 3. Lesson 2 screening on the extracted data ──")
    print("risk_flags:", screened["risk_flags"], "| decision_hint:", screened["decision_hint"])
