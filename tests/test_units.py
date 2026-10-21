from types import SimpleNamespace

import pytest

from app import llm, store
from app.config import cfg
from app.ingest import chunk_ids, split_text


def test_slug_and_pretty():
    assert store.slug("New York, USA") == "new_york_usa"
    assert store.slug("  !!! ") == ""
    assert store.pretty("new_york_usa") == "New York Usa"


def test_match_city_prefix_and_fuzzy():
    avail = {"mumbai_india": 5, "new_delhi_india": 3}
    assert store.match_city("Mumbai", avail) == "mumbai_india"
    assert store.match_city("Mumbay", avail) == "mumbai_india"      # typo
    assert store.match_city("Paris", avail) is None
    assert store.match_city("", avail) is None


def test_split_text_chunks_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    chunks = split_text(text, size=500)
    assert len(chunks) > 3 and all(len(c) <= 700 for c in chunks)
    assert split_text("tiny") == []


def test_chunk_ids_are_stable_and_unique():
    a = chunk_ids("x", ["one chunk of text", "another chunk of text"])
    assert a == chunk_ids("x", ["one chunk of text", "another chunk of text"])
    assert len(set(a)) == 2


def test_text_of_handles_blocks_and_skips_thoughts():
    r = SimpleNamespace(content=[{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Hello "}, "world"])
    assert llm.text_of(r) == "Hello world"
    assert llm.text_of(SimpleNamespace(content="  hi ")) == "hi"


def test_parse_json_strips_fences():
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('Sure! {"a": 2} done') == {"a": 2}


@pytest.mark.parametrize("model,level,expected", [
    ("gemini-3.8-flash", "minimal", "low"),          # 3.8 rejects minimal
    ("gemini-3.8-flash", "high", "high"),
    ("gemini-3.5-flash-lite", "minimal", "minimal"),
    ("gemini-3.6-flash", "minimal", "minimal"),
    ("gemini-2.5-flash", "low", None),               # legacy: no thinking_level
])
def test_thinking_for(model, level, expected):
    assert llm.thinking_for(model, level) == expected


def test_no_temperature_for_gemini3(monkeypatch):
    captured = {}

    class Fake:
        def __init__(self, **kw): captured.update(kw)

    import langchain_google_genai
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", Fake)
    llm.build_llm.cache_clear()
    llm.build_llm("gemini-3.8-flash", "low", False, 100)
    assert "temperature" not in captured and captured["thinking_level"] == "low"
    llm.build_llm.cache_clear()
    captured.clear()
    llm.build_llm("gemini-2.5-flash", None, True, 100)
    assert captured["temperature"] == 0.2 and captured["response_mime_type"] == "application/json"
    llm.build_llm.cache_clear()


def test_model_chain_dedupes(monkeypatch):
    monkeypatch.setattr(cfg, "main_model", "gemini-3.6-flash")
    monkeypatch.setattr(cfg, "fallback_models", ["gemini-3.6-flash", "gemini-2.5-flash"])
    assert llm.chain_for("main") == ["gemini-3.6-flash", "gemini-2.5-flash"]


def test_fallback_to_next_model_on_quota_and_404(monkeypatch):
    monkeypatch.setattr(cfg, "main_model", "gemini-3.8-flash")
    monkeypatch.setattr(cfg, "fallback_models", ["gemini-3.6-flash", "gemini-2.5-flash"])
    tried = []

    class Model:
        def __init__(self, name): self.name = name
        def invoke(self, msgs):
            tried.append(self.name)
            if self.name == "gemini-3.8-flash":
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
            if self.name == "gemini-3.6-flash":
                raise RuntimeError("404 models/gemini-3.6-flash is not found")
            return SimpleNamespace(content="final answer")

    monkeypatch.setattr(llm, "build_llm", lambda model, *a: Model(model))
    assert llm.invoke_text("hi") == "final answer"
    assert tried == ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash"]


def test_bad_api_key_does_not_fallback(monkeypatch):
    tried = []

    class Model:
        def invoke(self, msgs):
            tried.append(1)
            raise RuntimeError("400 API key not valid. Please pass a valid API key.")

    monkeypatch.setattr(llm, "build_llm", lambda *a: Model())
    with pytest.raises(RuntimeError):
        llm.invoke_text("hi")
    assert len(tried) == 1


def test_ask_json_returns_default_on_garbage(monkeypatch):
    monkeypatch.setattr(llm, "invoke_text", lambda *a, **k: "not json at all")
    assert llm.ask_json("x", {"d": 1}) == {"d": 1}


def test_friendly_messages():
    assert "rate limit" in llm.friendly(RuntimeError("429 quota"))
    assert "API key" in llm.friendly(RuntimeError("API key not valid"))
    assert llm.friendly(RuntimeError("boom")) == "Something went wrong. Please try again."