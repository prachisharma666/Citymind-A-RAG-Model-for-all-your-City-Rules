import hmac
import logging
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import llm as L
from . import store as S
from .config import cfg, has_ai, has_db, has_web
from .graph import pipeline
from .ingest import chunk_ids, split_text

log = logging.getLogger("citymind.api")
api = APIRouter(prefix="/api")


# ---------- small guards ----------
_hits: dict = defaultdict(deque)


def rate_limit(request: Request):
    """Sliding 60-second window per client IP (in-memory; fine for one server)."""
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= cfg.ask_rate_per_min:
        raise HTTPException(429, "Too many questions - please wait a minute.")
    q.append(now)


def require_admin(token: str | None):
    """If ADMIN_TOKEN is set, uploads/deletes/probes need header  X-Admin-Token."""
    if cfg.admin_token and not hmac.compare_digest(token or "", cfg.admin_token):
        raise HTTPException(401, "Admin token required.")


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    history: list[dict] = Field(default_factory=list)
    city: str = ""


class IngestBody(BaseModel):
    city: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=80000)
    source: str = "upload"


# ---------- routes ----------
@api.get("/health")
def health():
    return {"ok": True}


@api.get("/status")
def status():
    cities = [{"ns": ns, "name": S.pretty(ns), "chunks": n} for ns, n in sorted(S.list_cities().items())]
    return {"ai": has_ai(), "db": has_db(), "web": has_web(), "uploads": cfg.allow_uploads and has_db(),
            "model": cfg.main_model, "fast_model": cfg.fast_model, "fallbacks": cfg.fallback_models,
            "protected": bool(cfg.admin_token), "cities": cities}


@api.get("/health/llm")
def health_llm(x_admin_token: str | None = Header(default=None)):
    """Live probe: one tiny call per configured model. Costs a few tokens - protect it with ADMIN_TOKEN."""
    require_admin(x_admin_token)
    if not has_ai():
        raise HTTPException(503, "GOOGLE_API_KEY is not set on the server yet.")
    results = []
    for tier in ("fast", "main"):
        model = L.chain_for(tier)[0]
        t0 = time.time()
        try:
            llm = L.build_llm(model, L.thinking_for(model, cfg.fast_thinking if tier == "fast" else cfg.main_thinking),
                              False, 256)
            out = L.text_of(llm.invoke("Reply with the single word: ok"))
            results.append({"tier": tier, "model": model, "ok": True, "reply": out[:40],
                            "ms": int((time.time() - t0) * 1000)})
        except Exception as exc:                       # noqa: BLE001
            results.append({"tier": tier, "model": model, "ok": False, "error": str(exc)[:300]})
    return {"ok": all(r["ok"] for r in results), "results": results}


@api.post("/ask")
def ask(body: AskBody, request: Request):
    rate_limit(request)
    if not has_ai():
        raise HTTPException(503, "GOOGLE_API_KEY is not set on the server yet.")
    history = [{"role": "user" if t.get("role") == "user" else "assistant", "content": str(t.get("content", ""))[:1500]}
               for t in body.history[-8:]]
    try:
        out = pipeline().invoke({"question": body.question.strip(), "history": history,
                                 "city_hint": body.city.strip()[:80], "steps": []})
    except Exception as exc:                           # noqa: BLE001
        log.exception("ask failed")
        raise HTTPException(500, L.friendly(exc))
    return {"answer": out["answer"], "followups": out.get("followups", []), "sources": out.get("sources", []),
            "mode": out.get("mode", "general"), "steps": out.get("steps", [])}


@api.post("/ingest")
def ingest(body: IngestBody, x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    if not cfg.allow_uploads:
        raise HTTPException(403, "Uploads are turned off on this server.")
    if not (has_ai() and has_db()):
        raise HTTPException(503, "Add GOOGLE_API_KEY and PINECONE_API_KEY on the server to store documents.")
    ns = S.slug(body.city)
    if not ns:
        raise HTTPException(422, "Please enter a city name using letters or numbers.")
    chunks = split_text(body.text)
    if not chunks:
        raise HTTPException(422, "No readable text found.")
    try:
        S.ensure_index()
        S.add_chunks(ns, body.city.strip(), body.source, chunks, chunk_ids(ns, chunks))
    except Exception as exc:                           # noqa: BLE001
        log.exception("ingest failed")
        raise HTTPException(500, L.friendly(exc))
    return {"city": body.city.strip(), "ns": ns, "chunks": len(chunks)}


@api.delete("/cities/{ns}")
def delete_city(ns: str, x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    if not cfg.allow_uploads:
        raise HTTPException(403, "Changes are turned off on this server.")
    if not has_db():
        raise HTTPException(503, "Pinecone is not configured.")
    try:
        S.delete_namespace(ns)
    except Exception as exc:                           # noqa: BLE001
        raise HTTPException(500, L.friendly(exc))
    return {"deleted": ns}