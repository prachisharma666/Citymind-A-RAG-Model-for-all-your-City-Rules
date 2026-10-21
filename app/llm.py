"""Gemini access: one place that knows about model quirks, fallbacks and JSON parsing."""
import json
import logging
import re
from functools import lru_cache

from langchain_core.messages import HumanMessage

from .config import cfg

log = logging.getLogger("citymind.llm")


def is_gemini3(model: str) -> bool:
    m = re.search(r"gemini-(\d+)", model or "")
    return bool(m) and int(m.group(1)) >= 3


def thinking_for(model: str, level: str):
    """Return a thinking level this model accepts (or None to use the model default).

    - Gemini 3.x+: temperature/top_p/top_k are deprecated, so we steer with thinking_level.
    - Gemini 3.8 Flash rejects 'minimal' (HTTP 400) -> use 'low' instead.
    - Older models (2.x) do not take thinking_level at all.
    """
    if not level or not is_gemini3(model):
        return None
    if level == "minimal" and "3.8" in model:
        return "low"
    return level


def text_of(response) -> str:
    """Gemini returns a string or a list of content blocks - always give back plain text."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict) and c.get("type", "text") == "text":   # skip thought blocks
                parts.append(str(c.get("text", "")))
        return "".join(parts).strip()
    return str(content).strip()


@lru_cache(maxsize=16)
def build_llm(model: str, thinking: str | None, json_mode: bool, max_tokens: int):
    from langchain_google_genai import ChatGoogleGenerativeAI

    kw = dict(model=model, max_retries=2, timeout=cfg.llm_timeout, max_output_tokens=max_tokens)
    if thinking:
        kw["thinking_level"] = thinking
    if json_mode:
        kw["response_mime_type"] = "application/json"
    if not is_gemini3(model):
        kw["temperature"] = 0.2          # only legacy models still honour it
    return ChatGoogleGenerativeAI(**kw)


def chain_for(tier: str) -> list:
    first = cfg.fast_model if tier == "fast" else cfg.main_model
    out = []
    for m in [first, *cfg.fallback_models]:
        if m not in out:
            out.append(m)
    return out


_FALLBACK_HINTS = ("429", "quota", "resource_exhausted", "503", "unavailable", "overloaded",
                   "404", "not found", "not_found", "400", "invalid_argument", "deadline", "timeout")


def should_fallback(exc: Exception) -> bool:
    low = str(exc).lower()
    if "api key" in low or "api_key" in low or "permission" in low or "401" in low or "403" in low:
        return False                      # a different model will not fix a bad key
    return any(h in low for h in _FALLBACK_HINTS)


def invoke_text(prompt: str, tier: str = "main", json_mode: bool = False) -> str:
    """Call the tier's primary model; on quota/overload/unknown-model errors try the fallbacks."""
    level = cfg.fast_thinking if tier == "fast" else cfg.main_thinking
    max_tokens = 2048 if tier == "fast" else 8192
    last = None
    for model in chain_for(tier):
        try:
            llm = build_llm(model, thinking_for(model, level), json_mode, max_tokens)
            return text_of(llm.invoke([HumanMessage(content=prompt)]))
        except Exception as exc:                      # noqa: BLE001
            last = exc
            if not should_fallback(exc):
                raise
            log.warning("model %s failed (%s) - trying next", model, str(exc)[:160])
    raise last


def parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start: end + 1])


def ask_json(prompt: str, default: dict, tier: str = "fast") -> dict:
    try:
        return parse_json(invoke_text(prompt, tier=tier, json_mode=True))
    except Exception as exc:                          # noqa: BLE001
        log.warning("ask_json fell back to default: %s", str(exc)[:160])
        return default


@lru_cache(maxsize=2)
def embedder(task_type: str):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=cfg.embed_model, output_dimensionality=cfg.embed_dim,
                                        task_type=task_type)


def embed_query(text: str) -> list:
    return embedder("RETRIEVAL_QUERY").embed_query(text)     # queries and documents use different task types


def friendly(exc: Exception) -> str:
    msg, low = str(exc), str(exc).lower()
    if "google_api_key" in low or "api key" in low or "api_key" in low:
        return "The Google API key is missing or invalid. Check GOOGLE_API_KEY on the server."
    if "429" in msg or "quota" in low or "resource_exhausted" in low:
        return "The AI service is busy (rate limit). Please wait a few seconds and try again."
    if "pinecone" in low and ("unauthorized" in low or "401" in msg):
        return "The Pinecone API key is invalid. Check PINECONE_API_KEY on the server."
    return "Something went wrong. Please try again."