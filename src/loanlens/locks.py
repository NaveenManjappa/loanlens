"""One run per case at a time.

Two requests for the same case must never run the graph at once: both would start from the same
saved state, and the last to finish would overwrite the other. Before changing a case, the API takes
its lock. If the lock is taken, hold() raises CaseBusy (the API answers 409) instead of waiting.

    InProcessCaseLocks   one process (tests, local development with SQLite)
    PostgresCaseLocks    several replicas sharing one Postgres (advisory locks)
"""

import threading
from contextlib import contextmanager


class CaseBusy(Exception):
    """Someone else is changing this case right now."""


class InProcessCaseLocks:
    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}  # one Lock per case id
        self._guard = threading.Lock()  # protects the dict itself (two threads, one new case id)

    @contextmanager
    def hold(self, case_id: str):
        """Hold the case's lock for the duration of the with-block, or raise CaseBusy."""
        with self._guard:
            lock = self._locks.setdefault(case_id, threading.Lock())

        if not lock.acquire(blocking=False):
            raise CaseBusy(case_id)

        try:
            yield
        finally:
            lock.release()


class PostgresCaseLocks:
    """Postgres advisory locks: every replica asks the same database, so all of them respect the lock."""

    def __init__(self, pool):
        self.pool = pool  # the same psycopg ConnectionPool the checkpointer uses (rows are dicts)

    @contextmanager
    def hold(self, case_id: str):
        """Hold a transaction-scoped advisory lock on the case, or raise CaseBusy.

        pg_try_advisory_xact_lock doesn't wait, and it's released automatically when the transaction
        ends, even if the app crashes or the connection drops, so a lock can't leak.
        """
        with self.pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0)) AS got",
                    (case_id,),
                ).fetchone()
                if row is None or not row["got"]:
                    raise CaseBusy(case_id)
                yield
