"""Cases survive a restart: SQLite always, Postgres when LOANLENS_PG_URL is set, e.g.

    docker run -d -e POSTGRES_PASSWORD=dev -p 5432:5432 postgres:16
    LOANLENS_PG_URL=postgresql://postgres:dev@localhost:5432/postgres uv run pytest tests/test_checkpointing.py
"""

import os
import uuid

import pytest

from loanlens.cases import continue_application, current_outcome
from loanlens.checkpointing import open_checkpointer
from loanlens.extraction import ApplicationDetails
from loanlens.pipeline import build_graph

PRIYA = ApplicationDetails(
    applicant_name="Priya Shah", age=34, annual_income=42_000, loan_amount=8_000, employment_status="employed"
)


def survives_restart(scripted_structured_model, **where):
    case = f"case-{uuid.uuid4().hex[:8]}"
    with open_checkpointer(**where) as saver:  # "process 1"
        continue_application(build_graph(scripted_structured_model([PRIYA]), checkpointer=saver), case, "Priya Shah, 34, ...")
    with open_checkpointer(**where) as saver:  # "process 2", after a restart: nothing in memory
        assert current_outcome(build_graph(scripted_structured_model([PRIYA]), checkpointer=saver), case) == "approved_in_principle"


def test_sqlite_case_survives_restart(scripted_structured_model, tmp_path):
    survives_restart(scripted_structured_model, sqlite_path=str(tmp_path / "cases.db"))


@pytest.mark.skipif(not os.getenv("LOANLENS_PG_URL"), reason="set LOANLENS_PG_URL to test Postgres")
def test_postgres_case_survives_restart(scripted_structured_model):
    survives_restart(scripted_structured_model, database_url=os.environ["LOANLENS_PG_URL"])
