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
        # an id, so replaying the same object would overwrite, not append.
        return self.script[i].model_copy(update={"id": str(uuid.uuid4())})


@pytest.fixture
def fake_tool_model():
    return FakeToolModel


# ── create_agent needs a REAL chat-model class, so this fake subclasses BaseChatModel ──

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


class ScriptedChatModel(BaseChatModel):
    """A BaseChatModel that replays a script of AIMessages (last one repeats).
    `calls` records every message list it was sent; `bound_tools` the tool names."""

    script: list
    calls: list = []
    bound_tools: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [getattr(t, "name", t) for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        i = min(len(self.calls) - 1, len(self.script) - 1)
        msg = self.script[i].model_copy(update={"id": str(uuid.uuid4())})
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def scripted_model():
    def make(script):
        return ScriptedChatModel(script=list(script), calls=[], bound_tools=[])

    return make


class ScriptedStructuredModel(FakeStructuredModel):
    """Like FakeStructuredModel, but answers from a LIST, one per call (last one repeats).
    `all_sent` records the messages of every call. One model across several runs."""

    def __init__(self, answers):
        super().__init__(answers[0])
        self.answers = list(answers)
        self.all_sent = []

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        fake = self

        class _Runnable:
            def invoke(self, messages, *args, **kw):
                fake.all_sent.append(list(messages))
                fake.sent = messages
                i = min(len(fake.all_sent) - 1, len(fake.answers) - 1)
                return fake.answers[i]

        return _Runnable()


@pytest.fixture
def scripted_structured_model():
    return ScriptedStructuredModel
