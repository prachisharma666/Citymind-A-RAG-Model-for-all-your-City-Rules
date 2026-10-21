"""Pinecone access + city/namespace helpers."""
import difflib
import os
import re
import threading
import time
from functools import lru_cache

from .config import cfg, has_db


def slug(name: str) -> str:
    """'New York, USA' -> 'new_york_usa'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_")


def pretty(ns: str) -> str:
    return " ".join(w.capitalize() for w in ns.split("_"))


@lru_cache(maxsize=1)
def pinecone_client():
    from pinecone import Pinecone

    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def ensure_index():
    from pinecone import ServerlessSpec

    pc = pinecone_client()
    if cfg.index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(name=cfg.index_name, dimension=cfg.embed_dim, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    for _ in range(40):
        if pc.describe_index(cfg.index_name).status.get("ready"):
            return
        time.sleep(2)


def vectorstore(namespace: str):
    from langchain_pinecone import PineconeVectorStore
    from .llm import embedder

    return PineconeVectorStore(index_name=cfg.index_name, embedding=embedder("RETRIEVAL_DOCUMENT"),
                               namespace=namespace)


_cache = {"at": 0.0, "data": {}}
_lock = threading.Lock()


def invalidate_cities():
    with _lock:
        _cache["at"] = 0.0


def list_cities() -> dict:
    """{namespace: chunk_count}. Cached for a few seconds - the router asks on every question."""
    if not has_db():
        return {}
    with _lock:
        if time.time() - _cache["at"] < cfg.cities_cache_ttl:
            return dict(_cache["data"])
    out = {}
    try:
        pc = pinecone_client()
        if cfg.index_name in [i.name for i in pc.list_indexes()]:
            for ns, info in pc.Index(cfg.index_name).describe_index_stats().namespaces.items():
                n = info.get("vector_count", 0) if isinstance(info, dict) else getattr(info, "vector_count", 0)
                if ns:
                    out[ns] = int(n or 0)
    except Exception:                                  # noqa: BLE001
        return {}
    with _lock:
        _cache.update(at=time.time(), data=out)
    return dict(out)


def match_city(name: str, available: dict):
    """'Mumbai' finds the 'mumbai_india' namespace."""
    s = slug(name)
    if not s:
        return None
    if s in available:
        return s
    for ns in available:
        if ns.startswith(s + "_") or s.startswith(ns + "_"):
            return ns
    # fuzzy: compare with the full namespace AND with the city part alone ('mumbai_india' -> 'mumbai'),
    # otherwise a typo like 'Mumbay' scores too low against the country suffix
    options = {}
    for ns in available:
        options[ns] = ns
        if "_" in ns:
            options.setdefault(ns.rsplit("_", 1)[0], ns)
    close = difflib.get_close_matches(s, list(options), n=1, cutoff=0.75)
    return options[close[0]] if close else None


def search(ns: str, vector: list, k: int) -> list:
    out = []
    for d in vectorstore(ns).similarity_search_by_vector(vector, k=k):
        meta = d.metadata or {}
        out.append({"text": d.page_content, "city": meta.get("city") or pretty(ns),
                    "source": meta.get("source", ""), "url": ""})
    return out


def add_chunks(ns: str, city: str, source: str, chunks: list, ids: list):
    store = vectorstore(ns)
    for i in range(0, len(chunks), 100):                # keep each request small
        part = chunks[i:i + 100]
        store.add_texts(texts=[f"[{city}] {c}" for c in part],
                        metadatas=[{"city": city, "source": source[:120]}] * len(part),
                        ids=ids[i:i + 100])
    invalidate_cities()


def delete_namespace(ns: str):
    pinecone_client().Index(cfg.index_name).delete(delete_all=True, namespace=ns)
    invalidate_cities()