import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def client():
    app = create_app(Settings(llm_provider="mock", openai_api_key=""))
    with TestClient(app) as c:
        yield c


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        events.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return events


@pytest.fixture()
def chat(client):
    sid = client.post("/api/sessions").json()["session_id"]

    def send(message: str, simulate: str = "none"):
        r = client.post("/api/chat", json={"session_id": sid, "message": message, "simulate": simulate})
        assert r.status_code == 200, r.text
        return parse_sse(r.text)

    send.session_id = sid
    return send
