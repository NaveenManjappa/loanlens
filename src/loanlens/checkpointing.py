"""Where cases live, and how concurrent changes to one case are prevented.

Without DATABASE_URL: a SQLite file and in-process locks (one local process).
With DATABASE_URL:    Postgres and Postgres advisory locks, so cases survive restarts and
                      redeploys, and several replicas can safely serve the same case.
"""

from contextlib import contextmanager
from typing import Any, NamedTuple


class Backend(NamedTuple):
    checkpointer: Any
    locks: Any


@contextmanager
def open_backend(database_url: str | None = None, sqlite_path: str = "loanlens_cases.db"):
    """Yield Backend(checkpointer, locks) for the app's lifetime and close everything afterwards."""
    from loanlens.locks import InProcessCaseLocks, PostgresCaseLocks

    if not database_url:
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(sqlite_path) as saver:
            yield Backend(saver, InProcessCaseLocks())
        return

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    # A POOL, not one connection: FastAPI serves several requests at once.
    # These connection settings are what PostgresSaver expects.
    with ConnectionPool(
        conninfo=database_url,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    ) as pool:
        saver = PostgresSaver(pool)
        saver.setup()  # creates the checkpoint tables on first run; safe to repeat
        yield Backend(saver, PostgresCaseLocks(pool))


@contextmanager
def open_checkpointer(database_url: str | None = None, sqlite_path: str = "loanlens_cases.db"):
    """Just the checkpointer (used by tests and scripts)."""
    with open_backend(database_url, sqlite_path) as backend:
        yield backend.checkpointer
