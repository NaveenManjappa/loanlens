"""Lesson 12 stretch spec: one run per case at a time. Run: uv run pytest tests/test_lesson12b_case_lock.py -v

The double-click problem: two requests for the same case run at once, both start from the same
saved state, and the one that finishes last silently overwrites the other (a lost update).
The fix: before CHANGING a case, take that case's lock. If someone else holds it, don't wait:
answer 409 "busy" (the "reject" strategy). Reads (GET) never need the lock.

  locks.py   1. InProcessCaseLocks.hold   one process: a threading.Lock per case id
             2. PostgresCaseLocks.hold    many replicas: a Postgres advisory lock (test needs LOANLENS_PG_URL)
  api.py     3. the case_busy handler     CaseBusy -> 409 {"detail": BUSY}
             4. send_message and decide:  wrap the work in `with locks.hold(case_id):`
                (in decide, the awaiting_underwriter CHECK goes inside the lock too:
                 check-then-act outside a lock is itself a race)
"""

import os
import threading
import time

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from loanlens.api import BUSY, create_app
from loanlens.extraction import ApplicationDetails
from loanlens.locks import CaseBusy, InProcessCaseLocks
from loanlens.pipeline import build_graph

PRIYA = ApplicationDetails(
    applicant_name="Priya Shah", age=34, annual_income=42_000, loan_amount=8_000, employment_status="employed"
)
PRIYA_TEXT = "Priya Shah, 34, employed nurse, £42,000 a year, want £8,000"
SAM = ApplicationDetails(
    applicant_name="Sam Jones", age=29, annual_income=40_000, loan_amount=5_000, employment_status="self_employed"
)


class SlowModel:
    """Takes a while to answer, like a real model, and counts how often it's called."""

    def __init__(self, answer, seconds=0.5):
        self.answer, self.seconds, self.calls = answer, seconds, 0

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages, *args, **kwargs):
        self.calls += 1
        time.sleep(self.seconds)
        return self.answer


# ── 1. InProcessCaseLocks ──────────────────────────────────────────────────


def test_a_held_case_is_busy():
    locks = InProcessCaseLocks()
    with locks.hold("case-1"):
        with pytest.raises(CaseBusy):
            with locks.hold("case-1"):
                pass


def test_other_cases_are_not_blocked():
    locks = InProcessCaseLocks()
    with locks.hold("case-1"):
        with locks.hold("case-2"):
            pass


def test_lock_is_released_afterwards_even_after_an_error():
    locks = InProcessCaseLocks()
    with pytest.raises(RuntimeError):
        with locks.hold("case-1"):
            raise RuntimeError("the graph crashed")
    with locks.hold("case-1"):  # would raise CaseBusy if the lock had leaked
        pass


def test_the_lock_works_across_threads():
    locks, outcomes = InProcessCaseLocks(), []

    def worker():
        try:
            with locks.hold("case-1"):
                time.sleep(0.3)
            outcomes.append("ran")
        except CaseBusy:
            outcomes.append("busy")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(outcomes) == ["busy", "ran"]


# ── 2. The API answers 409 while a case is busy ────────────────────────────


def test_message_to_a_busy_case_is_409_and_runs_nothing():
    model = SlowModel(PRIYA, seconds=0)
    app = create_app(build_graph(model, checkpointer=InMemorySaver()))
    client = TestClient(app)
    with app.state.locks.hold("case-1"):  # pretend another request is mid-run
        response = client.post("/cases/case-1/messages", json={"text": PRIYA_TEXT})
    assert response.status_code == 409
    assert response.json() == {"detail": BUSY}
    assert model.calls == 0, "a rejected request must not run the graph"


def test_reads_do_not_need_the_lock():
    app = create_app(build_graph(SlowModel(PRIYA, seconds=0), checkpointer=InMemorySaver()))
    client = TestClient(app)
    client.post("/cases/case-1/messages", json={"text": PRIYA_TEXT})
    with app.state.locks.hold("case-1"):
        assert client.get("/cases/case-1").status_code == 200


def test_decision_on_a_busy_case_is_409():
    app = create_app(build_graph(SlowModel(SAM, seconds=0), checkpointer=InMemorySaver()))
    client = TestClient(app)
    client.post("/cases/sam/messages", json={"text": "Sam Jones, 29, self-employed, £40,000, want £5,000"})
    decision = {"decision": "approve", "reason": "Accounts seen", "underwriter_id": "uw-1"}
    with app.state.locks.hold("sam"):
        assert client.post("/cases/sam/decision", json=decision).status_code == 409
    assert client.post("/cases/sam/decision", json=decision).json()["outcome"] == "approved_in_principle"


def test_double_click_runs_the_case_once():
    model = SlowModel(PRIYA, seconds=0.5)
    client = TestClient(create_app(build_graph(model, checkpointer=InMemorySaver())))
    codes = []

    def click():
        codes.append(client.post("/cases/priya/messages", json={"text": PRIYA_TEXT}).status_code)

    threads = [threading.Thread(target=click) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(codes) == [200, 409]
    assert model.calls == 1, "the second click must not reach the model"


# ── 3. Across replicas: Postgres advisory locks ────────────────────────────


@pytest.mark.skipif(not os.getenv("LOANLENS_PG_URL"), reason="set LOANLENS_PG_URL to test Postgres locks")
def test_postgres_lock_is_shared_between_replicas():
    from loanlens.checkpointing import open_backend

    url = os.environ["LOANLENS_PG_URL"]
    with open_backend(url) as replica_a, open_backend(url) as replica_b:  # two separate connection pools
        with replica_a.locks.hold("case-pg"):
            with pytest.raises(CaseBusy):
                with replica_b.locks.hold("case-pg"):
                    pass
            with replica_b.locks.hold("another-case"):
                pass
        with replica_b.locks.hold("case-pg"):  # released when A's block ended
            pass
