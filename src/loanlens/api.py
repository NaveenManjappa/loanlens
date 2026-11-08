"""HTTP API for LoanLens (FastAPI).

    POST /cases/{case_id}/messages   an applicant sends a message   -> CaseView
    GET  /cases/{case_id}            read a case (runs nothing)     -> CaseView, 404 if unknown
    GET  /cases/{case_id}/review     what the underwriter sees      -> review payload, 404 if none pending
    POST /cases/{case_id}/decision   the underwriter decides        -> CaseView, 409 if none pending
    GET  /health                     health probe                   -> {"status": "ok"}

A thin layer: routes call cases.py and underwriting.py and hold no business logic. Pydantic models
validate input, so a bad request gets a 422 before any graph runs. The compiled graph is injected
through create_app(graph): tests pass a graph built on fake models, main.py passes the real one.
Changes to a case run under a per-case lock (locks.py), so a concurrent request gets a 409.
"""

from typing import Literal

import openai
from fastapi import Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from loanlens.cases import continue_application, thread_config
from loanlens.locks import CaseBusy, InProcessCaseLocks
from loanlens.underwriting import awaiting_underwriter, submit_decision

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
    """What callers see of a case: never the raw state (least exposure)."""

    case_id: str
    outcome: str | None
    reply: str | None
    awaiting_underwriter: bool


def get_graph(request: Request):
    return request.app.state.graph


def get_locks(request: Request):
    return request.app.state.locks


def case_view(graph, case_id: str) -> dict | None:
    """Return the CaseView fields for a case, or None if the case doesn't exist."""
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
        """Another request is changing this case: 409 with a fixed message."""
        return JSONResponse(
            status_code=409,
            content={"detail": BUSY},
        )

    @app.exception_handler(openai.APIError)
    def model_unavailable(request: Request, exc: openai.APIError):
        """The model is down, rate-limited or unreachable: 503 with a fixed message.

        Never echo str(exc): provider error text can contain parts of the prompt, i.e. customer data.
        """
        return JSONResponse(
            status_code=503,
            content={"detail": UNAVAILABLE},
        )

    @app.get("/health")
    def health() -> dict:
        """Health probe."""
        return {"status": "ok"}

    @app.post("/cases/{case_id}/messages", response_model=CaseView)
    def send_message(body: MessageIn, case_id: str = CaseId, graph=Depends(get_graph),locks=Depends(get_locks)):
        """Send an applicant message into the case and return the updated view."""
        with locks.hold(case_id):
            continue_application(graph, case_id, body.text)
            return case_view(graph, case_id)

    @app.get("/cases/{case_id}", response_model=CaseView)
    def get_case(case_id: str = CaseId, graph=Depends(get_graph)):
        """Read a case without running anything."""
        view = case_view(graph, case_id)
        if view is None:
            raise HTTPException(status_code=404, detail="Case not found")
        return view

    @app.get("/cases/{case_id}/review")
    def get_review(case_id: str = CaseId, graph=Depends(get_graph),locks=Depends(get_locks)) -> dict:
        """The review payload for a case paused at underwriter_review."""
        with locks.hold(case_id):
            review = awaiting_underwriter(graph, case_id)
        if review is None:
            raise HTTPException(status_code=404, detail="No review pending")
        return review

    @app.post("/cases/{case_id}/decision", response_model=CaseView)
    def decide(body: DecisionIn, case_id: str = CaseId, graph=Depends(get_graph),locks=Depends(get_locks)):
        """Resume a paused case with the underwriter's decision.

        The pending-review check runs inside the lock: checking first and acting later would be a race.
        """
        with locks.hold(case_id):
            if awaiting_underwriter(graph, case_id) is None:
                raise HTTPException(status_code=409, detail="No review pending")
            submit_decision(graph, case_id, body.model_dump())
        return case_view(graph, case_id)

    return app
