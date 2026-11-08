"""Where cases are stored, and the locks that stop two requests changing one case at once.

Without DATABASE_URL: a SQLite file and in-process locks (one local process).
With DATABASE_URL:    Postgres and Postgres advisory locks, so cases survive restarts and
                      redeploys, and several replicas can safely serve the same case.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from loanlens.locks import CaseLocks, InProcessCaseLocks, PostgresCaseLocks

DEFAULT_SQLITE_PATH = "loanlens_cases.db"
POOL_MAX_SIZE = 10


class Backend(NamedTuple):
    checkpointer: BaseCheckpointSaver
    locks: CaseLocks


@contextmanager
def open_backend(
    database_url: str | None = None, sqlite_path: str = DEFAULT_SQLITE_PATH
) -> Iterator[Backend]:
    """Yield a Backend for the app's lifetime and close its connections afterwards."""
    if not database_url:
        with SqliteSaver.from_conn_string(sqlite_path) as saver:
            yield Backend(saver, InProcessCaseLocks())
        return

    # A pool, not one connection: FastAPI serves several requests at once.
    # These connection settings are what PostgresSaver requires.
    with ConnectionPool(
        conninfo=database_url,
        max_size=POOL_MAX_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    ) as pool:
        saver = PostgresSaver(pool)
        saver.setup()  # creates the checkpoint tables on first run; safe to repeat
        yield Backend(saver, PostgresCaseLocks(pool))
