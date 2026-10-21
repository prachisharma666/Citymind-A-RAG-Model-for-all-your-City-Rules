import json

from app import store


def test_health_and_status(client, fake_store):
    assert client.get("/api/health").json() == {"ok": True}
    s = client.get("/api/status").json()
    assert s["ai"] is True and s["db"] is False and s["model"] == "gemini-3.8-flash"
    assert s["fast_model"] == "gemini-3.5-flash-lite" and s["cities"][0]["name"] == "Mumbai India"


def test_ask_small_talk_skips_retrieval(client, fake_llm, fake_store):
    fake_llm.script['You route questions'] = json.dumps({"kind": "chat", "standalone": "hi", "cities": []})
    fake_llm.script['You are CityMind'] = "Hello! Ask me about city rules.\nFOLLOWUPS:\n- a\n- b\n- c"
    r = client.post("/api/ask", json={"question": "hi"}).json()
    assert r["mode"] == "chat" and r["followups"] == ["a", "b", "c"] and r["sources"] == []
    assert [s["name"] for s in r["steps"]] == ["Router", "Report"]


def test_ask_documents_path_with_citations(client, fake_llm, fake_store):
    fake_llm.script['You route questions'] = json.dumps(
        {"kind": "question", "standalone": "Max building height in Mumbai?", "cities": ["Mumbai"]})
    fake_llm.script['Do these passages'] = '{"sufficient": true}'
    fake_llm.script['You are CityMind'] = "70 m near the airport [1].\n\nFOLLOWUPS:\n- What about FSI?\n- x\n- y"
    r = client.post("/api/ask", json={"question": "height limit?"}).json()
    assert r["mode"] == "documents" and "[1]" in r["answer"]
    assert r["sources"][0]["city"] == "Mumbai"
    assert [s["name"] for s in r["steps"]] == ["Router", "Retriever", "Checker", "Report"]
    # router + checker use the fast tier with JSON mode; the report uses the main tier
    tiers = [c for c in fake_llm.calls]
    assert tiers == [("fast", True), ("fast", True), ("main", False)]


def test_ask_insufficient_docs_goes_to_fallback_general(client, fake_llm, fake_store):
    fake_llm.script['You route questions'] = json.dumps({"kind": "question", "standalone": "q", "cities": ["Mumbai"]})
    fake_llm.script['Do these passages'] = '{"sufficient": false}'
    fake_llm.script['You are CityMind'] = "Answer\nFOLLOWUPS:\n- 1\n- 2\n- 3"
    r = client.post("/api/ask", json={"question": "q"}).json()
    assert r["mode"] == "partial"           # docs exist but are not enough, and no web key
    assert [s["name"] for s in r["steps"]] == ["Router", "Retriever", "Checker", "Fallback", "Report"]


def test_ask_unknown_city_is_general(client, fake_llm, fake_store):
    fake_llm.script['You route questions'] = json.dumps({"kind": "question", "standalone": "q", "cities": ["Paris"]})
    r = client.post("/api/ask", json={"question": "Paris parking"}).json()
    assert r["mode"] == "general"


def test_ask_validation_and_missing_key(client, monkeypatch):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    monkeypatch.delenv("GOOGLE_API_KEY")
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 503


def test_ask_error_is_friendly(client, fake_llm, fake_store, monkeypatch):
    from app import graph

    def boom(*a, **k): raise RuntimeError("429 RESOURCE_EXHAUSTED")
    monkeypatch.setattr(graph.L, "invoke_text", boom)
    graph.pipeline.cache_clear()
    r = client.post("/api/ask", json={"question": "hi"})
    # router degrades gracefully to defaults, report is the one that fails
    assert r.status_code == 500 and "rate limit" in r.json()["detail"]


def test_rate_limit(client, fake_llm, fake_store, monkeypatch):
    from app.config import cfg
    monkeypatch.setattr(cfg, "ask_rate_per_min", 2)
    codes = [client.post("/api/ask", json={"question": "hi"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_ingest_requires_keys_and_admin_token(client, monkeypatch):
    from app.config import cfg
    body = {"city": "Mumbai", "text": "x" * 200}
    assert client.post("/api/ingest", json=body).status_code == 503          # no Pinecone key
    monkeypatch.setattr(cfg, "admin_token", "secret")
    assert client.post("/api/ingest", json=body).status_code == 401
    assert client.post("/api/ingest", json=body, headers={"x-admin-token": "wrong"}).status_code == 401
    assert client.delete("/api/cities/mumbai_india").status_code == 401
    assert client.get("/api/health/llm").status_code == 401


def test_ingest_success_with_fake_store(client, monkeypatch):
    from app.config import cfg
    monkeypatch.setenv("PINECONE_API_KEY", "p")
    saved = {}
    monkeypatch.setattr(store, "ensure_index", lambda: None)
    monkeypatch.setattr(store, "add_chunks", lambda ns, city, src, chunks, ids: saved.update(ns=ns, n=len(chunks), ids=ids))
    text = "\n\n".join(f"Rule {i}: " + "setback " * 40 for i in range(8))
    r = client.post("/api/ingest", json={"city": "New Delhi, India", "text": text}).json()
    assert r["ns"] == "new_delhi_india" and r["chunks"] == saved["n"] and len(set(saved["ids"])) == saved["n"]
    monkeypatch.setattr(cfg, "allow_uploads", False)
    assert client.post("/api/ingest", json={"city": "x", "text": text}).status_code == 403
    assert client.post("/api/ingest", json={"city": "!!!", "text": text}).status_code in (403, 422)


def test_delete_city(client, monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "p")
    gone = []
    monkeypatch.setattr(store, "delete_namespace", gone.append)
    assert client.delete("/api/cities/mumbai_india").json() == {"deleted": "mumbai_india"}
    assert gone == ["mumbai_india"]


def test_health_llm_probe(client, monkeypatch):
    from types import SimpleNamespace
    from app import llm

    class M:
        def invoke(self, p): return SimpleNamespace(content="ok")

    monkeypatch.setattr(llm, "build_llm", lambda *a: M())
    r = client.get("/api/health/llm").json()
    assert r["ok"] is True and [x["tier"] for x in r["results"]] == ["fast", "main"]