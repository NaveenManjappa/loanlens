"""Cases survive a restart: SQLite always, Postgres when LOANLENS_PG_URL is set, e.g.

docker run -d -e POSTGRES_PASSWORD=dev -p 5432:5432 postgres:16
LOANLENS_PG_URL=postgresql://postgres:dev@localhost:5432/postgres uv run pytest tests/integrations/test_storage.py
"""

import os
import uuid

import pytest
from helpers import current_outcome

from loanlens.domain.application import ApplicationDetails
from loanlens.graph.cases import continue_application
from loanlens.graph.pipeline import build_graph
from loanlens.integrations.storage import open_backend

PRIYA = ApplicationDetails(
    applicant_name="Priya Shah", age=34, annual_income=42_000, loan_amount=8_000, employment_status="employed"
)


def survives_restart(scripted_structured_model, **where):
    case = f"case-{uuid.uuid4().hex[:8]}"
    with open_backend(**where) as backend:  # "process 1"
        graph = build_graph(scripted_structured_model([PRIYA]), checkpointer=backend.checkpointer)
        continue_application(graph, case, "Priya Shah, 34, ...")
    with open_backend(**where) as backend:  # "process 2", after a restart: nothing in memory
        graph = build_graph(scripted_structured_model([PRIYA]), checkpointer=backend.checkpointer)
        assert current_outcome(graph, case) == "approved_in_principle"


def test_sqlite_case_survives_restart(scripted_structured_model, tmp_path):
    survives_restart(scripted_structured_model, sqlite_path=str(tmp_path / "cases.db"))


@pytest.mark.skipif(not os.getenv("LOANLENS_PG_URL"), reason="set LOANLENS_PG_URL to test Postgres")
def test_postgres_case_survives_restart(scripted_structured_model):
    survives_restart(scripted_structured_model, database_url=os.environ["LOANLENS_PG_URL"])
