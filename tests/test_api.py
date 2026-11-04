import json

from fastapi.testclient import TestClient

from answer.answer import Answer
from retrieval import Hit
from service.api import create_app


class FakeRetriever:
    def search(self, q, k):
        return [Hit("c:d:0", "d", 3, 3, "text", 1.0)]


def fake_stream(question, hits):
    yield "Hello "
    yield "world [1]"
    yield Answer(
        "Hello world [1]",
        [{"n": 1, "doc": "d", "page": 3, "chunk_id": "c:d:0", "source": "d:3"}],
        meta={"cost_usd": 0.001},
    )


def parse(text):
    events = []
    for block in text.strip().split("\n\n"):
        ev, data = block.split("\n")
        events.append((ev.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def client(**kw):
    return TestClient(create_app(FakeRetriever(), fake_stream, **kw))


def test_health():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_ask_streams_tokens_then_citations_then_done(tmp_path, monkeypatch):
    monkeypatch.setattr("service.api.REQUEST_LOG", tmp_path / "req.jsonl")
    with client() as c:
        r = c.post("/ask", json={"question": "what is hello?"})
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = parse(r.text)
    assert [e for e, _ in ev] == ["token", "token", "citations", "done"]
    assert ev[2][1]["citations"][0]["source"] == "d:3" and not ev[2][1]["refused"]
    assert ev[3][1]["cost_usd"] == 0.001
    log = json.loads((tmp_path / "req.jsonl").read_text())
    assert log["cost_usd"] == 0.001 and log["n_citations"] == 1


def test_short_question_is_rejected():
    with client() as c:
        assert c.post("/ask", json={"question": "x"}).status_code == 422


def test_rate_limit_returns_429_after_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("service.api.REQUEST_LOG", tmp_path / "req.jsonl")
    with client(rate_limit="2/minute") as c:
        codes = [c.post("/ask", json={"question": "valid question"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
