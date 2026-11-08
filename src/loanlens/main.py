"""The application uvicorn runs:  uv run uvicorn loanlens.main:app --reload

Configuration comes only from environment variables, so the same image runs locally and in Azure.
Without DATABASE_URL, cases are kept in a local SQLite file; with it, in Postgres.
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from loanlens.api import create_app
from loanlens.checkpointing import open_backend
from loanlens.pipeline import build_graph

load_dotenv()  # local .env only; in Azure the variables are set on the container app


@asynccontextmanager
async def lifespan(app):
    # Runs once at startup: open the database, build the graph (and its real model).
    with open_backend(os.getenv("DATABASE_URL")) as backend:
        app.state.graph = build_graph(checkpointer=backend.checkpointer)
        app.state.locks = backend.locks  # one run per case at a time
        yield
    # ...and once at shutdown: the `with` closes the database connections.


app = create_app(lifespan=lifespan)
