"""The application uvicorn runs:  uv run uvicorn loanlens.main:app --reload

Configuration comes only from environment variables (config/settings.py), so the same image runs
locally and in Azure. Without DATABASE_URL, cases are kept in a local SQLite file; with it, in
Postgres. LOG_LEVEL sets the log level (default INFO).
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from loanlens.api.app import create_app
from loanlens.config.settings import get_settings
from loanlens.graph.pipeline import build_graph
from loanlens.graph.specialists import default_specialists
from loanlens.integrations.llm import get_model
from loanlens.integrations.storage import open_backend

# Settings read .env themselves; this also exports it to os.environ for the LangSmith library,
# which reads its LANGSMITH_* variables from there. In Azure the variables are set on the app.
load_dotenv()
settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: open the database and build the graph. Shutdown: close the database connections."""
    database_url = settings.database_url.get_secret_value() if settings.database_url else None
    model = get_model(settings)
    with open_backend(database_url, settings.sqlite_path) as backend:
        app.state.graph = build_graph(model, backend.checkpointer, default_specialists(model))
        app.state.locks = backend.locks
        logger.info("LoanLens started, storing cases in %s", "Postgres" if database_url else "SQLite")
        yield
    logger.info("LoanLens stopped")


app = create_app(lifespan=lifespan)
