"""Shared test helpers. A fake chat model so LLM nodes can be unit-tested
without calling Azure: fast, free, and the same answer every time."""

import os

import pytest


class FakeStructuredModel:
    """Stands in for a chat model. `with_structured_output(schema)` returns a
    runnable that records what it was sent and returns `answer` (a schema instance)."""

    def __init__(self, answer):
        self.answer = answer
        self.schema = None
        self.sent = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        fake = self

        class _Runnable:
            def invoke(self, messages, *args, **kw):
                fake.sent = messages
                return fake.answer

        return _Runnable()


@pytest.fixture
def fake_structured_model():
    return FakeStructuredModel


def pytest_collection_modifyitems(config, items):
    """Tests marked @pytest.mark.live call the real model. They're skipped unless
    LOANLENS_LIVE=1 is set, so the normal test run stays free and offline."""
    if os.getenv("LOANLENS_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live LLM test: run with LOANLENS_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
