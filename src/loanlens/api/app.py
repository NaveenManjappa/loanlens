"""HTTP API for LoanLens (FastAPI).

    POST /cases/{case_id}/messages   an applicant sends a message   -> CaseView
    GET  /cases/{case_id}            read a case (runs nothing)     -> CaseView, 404 if unknown
    GET  /cases/{case_id}/review     what the underwriter sees      -> review payload, 404 if none pending
    POST /cases/{case_id}/decision   the underwriter decides        -> CaseView, 409 if none pending
    GET  /health                     health probe                   -> {"status": "ok"}

A thin layer: routes call graph.cases and hold no business logic. Request and response models
are in api/schemas.py. The compiled graph is passed to create_app(graph):
tests pass a graph built on fake models; main.py sets the real one at startup. Changes to a case run
under a per-case lock (integrations/locks.py), so a concurrent request gets a 409.
"""

import logging
from typing import Any

import openai
from fastapi import Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from loanlens.api.schemas import CaseView, DecisionIn, MessageIn
from loanlens.graph.cases import (
    awaiting_underwriter,
    continue_application,
    submit_decision,
    thread_config,
)
from loanlens.integrations.locks import CaseBusy, CaseLocks, InProcessCaseLocks

logger = logging.getLogger(__name__)

# Case ids become LangGraph thread ids, so only safe characters are allowed. Never use PII (an
# email or NI number) as a case id: ids appear in URLs and access logs.
CaseId = Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")
UNAVAILABLE = "LoanLens is temporarily unavailable. Please try again shortly."
BUSY = "This case is already being processed. Please wait a moment and try again."


# ── Dependencies: FastAPI calls these to give each route the graph and the locks ──


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


def get_locks(request: Request) -> CaseLocks:
    return request.app.state.locks


def case_view(graph: CompiledStateGraph, case_id: str) -> dict[str, Any] | None:
    """The CaseView fields for a case, or None if the case doesn't exist."""
    values = graph.get_state(thread_config(case_id)).values
    if not values:
        return None

    replies = [m.content for m in values.get("messages", []) if isinstance(m, AIMessage)]
    return {
        "case_id": case_id,
        "outcome": values.get("outcome"),
        "reply": replies[-1] if replies else None,
        "awaiting_underwriter": awaiting_underwriter(graph, case_id) is not None,
    }


def create_app(
    graph: CompiledStateGraph | None = None,
    lifespan=None,
) -> FastAPI:
    """Build the FastAPI app. Without a graph, the lifespan must set app.state.graph at startup."""
    app = FastAPI(title="LoanLens", lifespan=lifespan)
    app.state.locks = InProcessCaseLocks()  # main.py's lifespan swaps in Postgres locks when deployed
    if graph is not None:
        app.state.graph = graph

    @app.exception_handler(CaseBusy)
    def case_busy(request: Request, exc: CaseBusy) -> JSONResponse:
        """Another request is changing this case: 409 with a fixed message."""
        logger.info("case busy, rejected %s %s", request.method, request.url.path)
        return JSONResponse(status_code=409, content={"detail": BUSY})

    @app.exception_handler(openai.APIError)
    def model_unavailable(request: Request, exc: openai.APIError) -> JSONResponse:
        """The model is down, rate-limited or unreachable: 503 with a fixed message.

        Never log or return str(exc): provider error text can contain parts of the prompt, i.e.
        customer data. Only the exception type is logged.
        """
        logger.error("model call failed (%s) on %s %s", type(exc).__name__, request.method, request.url.path)
        return JSONResponse(status_code=503, content={"detail": UNAVAILABLE})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/cases/{case_id}/messages", response_model=CaseView)
    def send_message(
        body: MessageIn,
        case_id: str = CaseId,
        graph: CompiledStateGraph = Depends(get_graph),
        locks: CaseLocks = Depends(get_locks),
    ) -> dict:
        """Send an applicant message into the case and return the updated view."""
        with locks.hold(case_id):
            continue_application(graph, case_id, body.text)
            return case_view(graph, case_id)

    @app.get("/cases/{case_id}", response_model=CaseView)
    def get_case(case_id: str = CaseId, graph: CompiledStateGraph = Depends(get_graph)) -> dict:
        """Read a case without running anything."""
        view = case_view(graph, case_id)
        if view is None:
            raise HTTPException(status_code=404, detail="Case not found")
        return view

    @app.get("/cases/{case_id}/review")
    def get_review(
        case_id: str = CaseId,
        graph: CompiledStateGraph = Depends(get_graph),
        locks: CaseLocks = Depends(get_locks),
    ) -> dict:
        """The review payload for a case paused at underwriter_review."""
        with locks.hold(case_id):
            review = awaiting_underwriter(graph, case_id)
        if review is None:
            raise HTTPException(status_code=404, detail="No review pending")
        return review

    @app.post("/cases/{case_id}/decision", response_model=CaseView)
    def decide(
        body: DecisionIn,
        case_id: str = CaseId,
        graph: CompiledStateGraph = Depends(get_graph),
        locks: CaseLocks = Depends(get_locks),
    ) -> dict:
        """Resume a paused case with the underwriter's decision.

        The pending-review check runs inside the lock: checking first and acting later would be a race.
        """
        with locks.hold(case_id):
            if awaiting_underwriter(graph, case_id) is None:
                raise HTTPException(status_code=409, detail="No review pending")
            submit_decision(graph, case_id, body.model_dump())
        return case_view(graph, case_id)

    return app
