"""Lesson 12: LoanLens behind HTTP.

    POST /cases/{case_id}/messages   applicant sends a message     -> CaseView
    GET  /cases/{case_id}            read a case (runs nothing)    -> CaseView, 404 if unknown
    GET  /cases/{case_id}/review     underwriter's screen          -> review payload, 404 if none pending
    POST /cases/{case_id}/decision   underwriter decides           -> CaseView, 409 if none pending
    GET  /health                     for Azure's health probes     -> {"status": "ok"}

The API is a THIN layer: every route calls functions you already wrote (cases.py,
underwriting.py). No business logic lives here. Validation of input is done by the
Pydantic models below (done for you), so a bad request gets a 422 before any graph runs.

The graph is NOT created here. create_app(graph) receives it (dependency injection again):
tests pass a graph with fake models; main.py passes the real one, built at startup.

YOUR JOB:  uv run pytest tests/test_lesson12_api.py -v  -> green
  1. case_view     2. the five route bodies     3. the model_unavailable handler
"""

from typing import Literal

import openai
from fastapi import Depends, FastAPI, HTTPException, Path, Request  # noqa: F401
from fastapi.responses import JSONResponse  # noqa: F401
from langchain_core.messages import AIMessage  # noqa: F401
from pydantic import BaseModel, Field

from loanlens.cases import continue_application, thread_config  # noqa: F401
from loanlens.locks import CaseBusy, InProcessCaseLocks  # noqa: F401
from loanlens.underwriting import awaiting_underwriter, submit_decision  # noqa: F401

# case ids become LangGraph thread ids: allow only safe characters, and never use PII
# (an email or NI number) as a case id: ids appear in URLs and access logs.
CaseId = Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")
UNAVAILABLE = "LoanLens is temporarily unavailable. Please try again shortly."
BUSY = "This case is already being processed. Please wait a moment and try again."


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class DecisionIn(BaseModel):
    decision: Literal["approve", "decline"]
    reason: str = Field(min_length=1)
    underwriter_id: str = Field(min_length=1)


class CaseView(BaseModel):
    """What the outside world sees of a case. Never the raw state (Lesson 7: least exposure)."""

    case_id: str
    outcome: str | None
    reply: str | None
    awaiting_underwriter: bool


def get_graph(request: Request):
    return request.app.state.graph


def get_locks(request: Request):
    return request.app.state.locks


def case_view(graph, case_id: str) -> dict | None:
    """Read the case with graph.get_state(thread_config(case_id)).values.
    No values (empty dict) -> None (the case doesn't exist).
    Otherwise return the CaseView fields:
      outcome              values.get("outcome")
      reply                content of the LAST AIMessage in values["messages"], or None if there isn't one
      awaiting_underwriter awaiting_underwriter(graph, case_id) is not None"""
    state = graph.get_state(thread_config(case_id))
    values = state.values
    if not values:
        return None

    reply = None
    for message in values.get("messages", []):
        if isinstance(message, AIMessage):
            reply = message.content

    return {
        "case_id": case_id,
        "outcome": values.get("outcome"),
        "reply": reply,
        "awaiting_underwriter": awaiting_underwriter(graph, case_id) is not None,
    }


def create_app(graph=None, lifespan=None, locks=None) -> FastAPI:
    app = FastAPI(title="LoanLens", lifespan=lifespan)
    app.state.locks = locks or InProcessCaseLocks()  # main.py swaps in Postgres locks at startup
    if graph is not None:
        app.state.graph = graph

    @app.exception_handler(CaseBusy)
    def case_busy(request: Request, exc: CaseBusy):
        """Lesson 12 stretch: 409 with {"detail": BUSY}.
        Then, in send_message and decide: add `locks=Depends(get_locks)` and do the work
        inside `with locks.hold(case_id):` (see tests/test_lesson12b_case_lock.py)."""
        raise NotImplementedError

    @app.exception_handler(openai.APIError)
    def model_unavailable(request: Request, exc: openai.APIError):
        """The model is down, rate-limited or unreachable. Return a 503 with
        {"detail": UNAVAILABLE}: a FIXED message. Never echo str(exc): error text can
        contain parts of the prompt, i.e. customer data. (Hint: JSONResponse.)"""
        return JSONResponse(
            status_code=503,
            content={"detail": UNAVAILABLE},
        )

    @app.get("/health")
    def health() -> dict:
        """For Azure's health probes. Return {"status": "ok"}."""
        return {"status": "ok"}

    @app.post("/cases/{case_id}/messages", response_model=CaseView)
    def send_message(body: MessageIn, case_id: str = CaseId, graph=Depends(get_graph)):
        """continue_application(...), then return case_view(...)."""
        continue_application(graph, case_id, body.text)
        return case_view(graph, case_id)

    @app.get("/cases/{case_id}", response_model=CaseView)
    def get_case(case_id: str = CaseId, graph=Depends(get_graph)):
        """case_view, or HTTPException(404) if the case doesn't exist."""
        view = case_view(graph, case_id)
        if view is None:
            raise HTTPException(status_code=404, detail="Case not found")
        return view

    @app.get("/cases/{case_id}/review")
    def get_review(case_id: str = CaseId, graph=Depends(get_graph)) -> dict:
        """The payload from awaiting_underwriter, or 404 if no review is pending."""
        review = awaiting_underwriter(graph, case_id)
        if review is None:
            raise HTTPException(status_code=404, detail="No review pending")
        return review

    @app.post("/cases/{case_id}/decision", response_model=CaseView)
    def decide(body: DecisionIn, case_id: str = CaseId, graph=Depends(get_graph)):
        """409 if no review is pending. Otherwise submit_decision(graph, case_id, body.model_dump())
        and return case_view."""
        if awaiting_underwriter(graph, case_id) is None:
            raise HTTPException(status_code=409, detail="No review pending")
        submit_decision(graph, case_id, body.model_dump())
        return case_view(graph, case_id)

    return app
