"""FastAPI service: POST /ask streams an answer over SSE, GET /health.

SSE events: `token` ({"t": text}) as the model writes, then one `citations` event with the
mapped doc:page list and flags, then `done` with latency and cost.
Each request appends one JSON line to logs/requests.jsonl (ip, latency, cost, refusal).
"""

import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from answer.answer import Answer, answer_stream

log = logging.getLogger("service")
REQUEST_LOG = Path(os.getenv("REQUEST_LOG", "logs/requests.jsonl"))
TOP_K = 5


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def default_retriever():
    from ingest.db import connect
    from retrieval import get_retriever

    config = f"s{os.getenv('CHUNK_SIZE', '512')}_o{os.getenv('CHUNK_OVERLAP', '64')}"
    # ponytail: one shared connection; psycopg serializes access. Use a pool if the
    # dense query ever becomes the bottleneck (the cross-encoder dominates today).
    r = get_retriever("hybrid", connect(), config, rerank=os.getenv("RERANK", "1") == "1")
    r.search("warm up", 1)
    return r


def create_app(
    retriever=None,
    stream_fn: Callable[[str, list], Iterator[str | Answer]] = answer_stream,
    rate_limit: str | None = None,
) -> FastAPI:
    load_dotenv()
    limiter = Limiter(key_func=get_remote_address, default_limits=[])
    limit = rate_limit or os.getenv("RATE_LIMIT", "30/minute")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = retriever or default_retriever()
        yield

    app = FastAPI(title="cited-answer-engine", lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask")
    @limiter.limit(limit)
    def ask(request: Request, body: AskRequest):
        ip = get_remote_address(request)

        def events() -> Iterator[str]:
            t0 = time.perf_counter()
            final: Answer | None = None
            try:
                hits = app.state.retriever.search(body.question, TOP_K)
                for x in stream_fn(body.question, hits):
                    if isinstance(x, Answer):
                        final = x
                    else:
                        yield sse("token", {"t": x})
            except Exception as e:
                # The 200 header is already sent, so a failure has to travel in the stream.
                log.exception("ask failed")
                yield sse("error", {"error": type(e).__name__})
                return
            assert final is not None
            yield sse(
                "citations",
                {
                    "answer": final.text,
                    "refused": final.refused,
                    "uncited": final.uncited,
                    "citations": final.citations,
                },
            )
            latency = time.perf_counter() - t0
            cost = final.meta.get("cost_usd", 0.0)
            yield sse("done", {"latency_s": round(latency, 3), "cost_usd": cost})
            _log(ip, body.question, latency, cost, final)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        return JSONResponse({"error": "internal error"}, status_code=500)

    return app


def _log(ip: str, question: str, latency: float, cost: float, a: Answer) -> None:
    REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(UTC).isoformat(),
        "ip": ip,
        "question_chars": len(question),
        "latency_s": round(latency, 3),
        "cost_usd": cost,
        "refused": a.refused,
        "uncited": a.uncited,
        "n_citations": len(a.citations),
        "input_tokens": a.meta.get("input_tokens"),
        "output_tokens": a.meta.get("output_tokens"),
    }
    with REQUEST_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def app_factory() -> FastAPI:  # uvicorn --factory service.api:app_factory
    return create_app()
