"""Shared test helpers. A fake chat model so LLM nodes can be unit-tested
without calling Azure: fast, free, and the same answer every time."""

import os
import uuid

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


class FakeToolModel:
    """Stands in for a tool-calling chat model. Replays a SCRIPT of AIMessages:
    each invoke() returns the next one (the last one repeats forever).
    Records the tools it was bound with and every message list it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.tools = None
        self.calls = []

    def bind_tools(self, tools, **kwargs):
        self.tools = list(tools)
        return self

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(list(messages))
        i = min(len(self.calls) - 1, len(self.script) - 1)
        # Fresh copy with a new id each time: add_messages REPLACES messages that share
        # an id (Lesson 2!), so replaying the same object would overwrite, not append.
        return self.script[i].model_copy(update={"id": str(uuid.uuid4())})


@pytest.fixture
def fake_tool_model():
    return FakeToolModel
