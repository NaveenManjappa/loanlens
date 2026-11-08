"""The application uvicorn runs:  uv run uvicorn loanlens.main:app --reload

Configuration comes only from environment variables, so the same image runs locally and in Azure.
Without DATABASE_URL, cases are kept in a local SQLite file; with it, in Postgres.
LOG_LEVEL sets the log level (default INFO).
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from loanlens.api import create_app
from loanlens.checkpointing import open_backend
from loanlens.llm import get_model
from loanlens.pipeline import build_graph
from loanlens.specialists import default_specialists

load_dotenv()  # local .env only; in Azure the variables are set on the container app
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: open the database and build the graph. Shutdown: close the database connections."""
    database_url = os.getenv("DATABASE_URL")
    model = get_model()
    with open_backend(database_url) as backend:
        app.state.graph = build_graph(model, backend.checkpointer, default_specialists(model))
        app.state.locks = backend.locks
        logger.info("LoanLens started, storing cases in %s", "Postgres" if database_url else "SQLite")
        yield
    logger.info("LoanLens stopped")


app = create_app(lifespan=lifespan)
