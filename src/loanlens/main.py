"""Lesson 12: the real app, as uvicorn runs it. Done for you.

    uv run uvicorn loanlens.main:app --reload            (local: SQLite file)
    DATABASE_URL=postgresql://... uvicorn loanlens.main:app   (deployed: Postgres)

Configuration comes ONLY from environment variables (12-factor): the same image runs
on your laptop and in Azure; only the environment differs. Secrets never go in the image.
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from loanlens.api import create_app
from loanlens.checkpointing import open_checkpointer
from loanlens.pipeline import build_graph

load_dotenv()  # local .env only; in Azure the variables are set on the container app


@asynccontextmanager
async def lifespan(app):
    # Runs once at startup: open the database, build the graph (and its real model).
    with open_checkpointer(os.getenv("DATABASE_URL")) as checkpointer:
        app.state.graph = build_graph(checkpointer=checkpointer)
        yield
    # ...and once at shutdown: the `with` closes the database connections.


app = create_app(lifespan=lifespan)
