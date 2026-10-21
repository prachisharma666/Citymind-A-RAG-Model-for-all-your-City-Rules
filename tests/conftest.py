import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.pop("PINECONE_API_KEY", None)
os.environ.pop("TAVILY_API_KEY", None)

import pytest
from fastapi.testclient import TestClient

from app import api as api_mod
from app import graph, llm, store
from app.config import cfg


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    api_mod._hits.clear()
    monkeypatch.setattr(cfg, "admin_token", "")
    monkeypatch.setattr(cfg, "allow_uploads", True)
    monkeypatch.setattr(cfg, "ask_rate_per_min", 1000)


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the network call. `script` maps a keyword in the prompt -> reply text."""
    calls = []
    script = {}

    def fake_invoke(prompt, tier="main", json_mode=False):
        calls.append((tier, json_mode))
        for key, reply in script.items():
            if key in prompt:
                return reply
        return "default reply"

    monkeypatch.setattr(llm, "invoke_text", fake_invoke)
    monkeypatch.setattr(graph.L, "invoke_text", fake_invoke)
    fake_invoke.calls, fake_invoke.script = calls, script
    return fake_invoke


@pytest.fixture
def fake_store(monkeypatch):
    cities = {"mumbai_india": 12}
    monkeypatch.setattr(store, "list_cities", lambda: dict(cities))
    monkeypatch.setattr(store, "search", lambda ns, vec, k: [
        {"text": f"[Mumbai] Max building height is 70 m near the airport ({ns}).", "city": "Mumbai",
         "source": "dcpr.pdf", "url": ""}])
    monkeypatch.setattr(llm, "embed_query", lambda t: [0.1] * 8)
    return cities