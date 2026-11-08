"""Lesson 12 stretch: one run per case at a time.

Two requests for the same case must never run the graph at the same moment: both would start
from the same saved state, and the one finishing last would silently overwrite the other.
Before CHANGING a case, the API takes that case's lock. If it's already taken, we don't wait:
we raise CaseBusy and the API answers 409 (LangGraph's server calls this the "reject" strategy).

Two implementations, one interface (`with locks.hold(case_id): ...`):
  InProcessCaseLocks   one Python process (tests, your laptop with SQLite)
  PostgresCaseLocks    several replicas sharing one Postgres (Azure + Neon)
open_backend() in checkpointing.py picks the right one, like it picks the checkpointer.

YOUR JOB:  uv run pytest tests/test_lesson12b_case_lock.py -v  -> green
"""

import threading  # noqa: F401
from contextlib import contextmanager


class CaseBusy(Exception):
    """Someone else is changing this case right now."""


class InProcessCaseLocks:
    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}  # one Lock per case id
        self._guard = threading.Lock()  # protects the dict itself (two threads, one new case id)

    @contextmanager
    def hold(self, case_id: str):
        """1. Under self._guard, get this case's Lock from self._locks (create it if missing:
              dict.setdefault does both in one step).
           2. Try to take it WITHOUT waiting: lock.acquire(blocking=False) returns False if taken.
              Taken -> raise CaseBusy(case_id).
           3. yield (the caller's `with` block runs here), and release the lock in a
              try/finally, so it's freed even if the graph crashes."""
        raise NotImplementedError
        yield  # noqa (makes this a generator)


class PostgresCaseLocks:
    """A Postgres ADVISORY LOCK: a lock that means only what your app decides it means, here
    "case X is busy". Every replica asks the same database, so all of them respect it."""

    def __init__(self, pool):
        self.pool = pool  # the same psycopg ConnectionPool the checkpointer uses (rows are dicts)

    @contextmanager
    def hold(self, case_id: str):
        """1. with self.pool.connection() as conn, conn.transaction():
        2.     row = conn.execute(
                   "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0)) AS got", (case_id,)
               ).fetchone()
               (hashtextextended turns the case id into the number Postgres locks on.
                *try* = don't wait; *xact* = released automatically when the transaction ends,
                even if the app crashes or the connection drops: no lock can leak.)
        3.     row["got"] is False -> raise CaseBusy(case_id). Otherwise yield,
               INSIDE the with-block, so the transaction (and the lock) lasts while the caller works.
        Always pass case_id as a parameter (%s), never paste it into the SQL string."""
        raise NotImplementedError
        yield  # noqa (makes this a generator)
