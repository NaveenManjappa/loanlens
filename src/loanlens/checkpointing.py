"""Lesson 12: where the cases live. Done for you.

Local development (no DATABASE_URL): a SQLite file, so cases survive restarts on your machine.
Deployed (DATABASE_URL set):          Postgres, so cases survive restarts, redeploys and
                                      several replicas serving the same case.
The graph doesn't care which: both are checkpointers (Lesson 8).
"""

from contextlib import contextmanager


@contextmanager
def open_checkpointer(database_url: str | None = None, sqlite_path: str = "loanlens_cases.db"):
    """Yield a ready checkpointer for the app's lifetime and close it afterwards."""
    if not database_url:
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(sqlite_path) as saver:
            yield saver
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
        yield saver
