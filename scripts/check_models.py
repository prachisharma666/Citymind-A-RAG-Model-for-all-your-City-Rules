#!/usr/bin/env python
"""Test your Gemini API key and pick the right model.   Usage:  python scripts/check_models.py

1. Lists the models your key can see (REST models.list)
2. Sends a tiny generateContent request to each candidate and times it (also checks JSON mode)
3. Tests the embedding model with the exact settings the app uses
4. Optionally checks Pinecone and Tavily if those keys are set
5. Prints which GEMINI_MODEL / GEMINI_FAST_MODEL / GEMINI_FALLBACKS to put in .env
Exit code 0 = everything the app needs works, 1 = something is broken.
"""
import json
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()
BASE = "https://generativelanguage.googleapis.com/v1beta"
KEY = os.getenv("GOOGLE_API_KEY", "")

# best-first candidates (checked against Google's docs, Oct 2026)
MAIN_CANDIDATES = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash"]
FAST_CANDIDATES = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
EMBED = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
DIM = int(os.getenv("EMBED_DIM", "768"))


def ok(msg): print(f"  \033[32m✔\033[0m {msg}")
def bad(msg): print(f"  \033[31m✘\033[0m {msg}")


def call(path, body=None, method="POST"):
    r = httpx.request(method, f"{BASE}/{path}", headers={"x-goog-api-key": KEY}, json=body, timeout=60)
    return r


def gen_body(model, json_mode=False):
    cfg = {"maxOutputTokens": 256}
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    # Gemini 3.x: no temperature. 3.8 Flash rejects "minimal"; 2.x takes no thinkingLevel.
    if "gemini-3" in model:
        cfg["thinkingConfig"] = {"thinkingLevel": "low" if "3.8" in model else "minimal"}
    prompt = 'Return only this JSON: {"ok": true}' if json_mode else "Reply with the single word: ok"
    return {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg}


def try_model(model):
    t0 = time.time()
    r = call(f"models/{model}:generateContent", gen_body(model))
    ms = int((time.time() - t0) * 1000)
    if r.status_code != 200:
        return False, ms, f"HTTP {r.status_code}: {r.text[:140].replace(chr(10), ' ')}"
    j = call(f"models/{model}:generateContent", gen_body(model, True))
    json_ok = False
    if j.status_code == 200:
        try:
            txt = j.json()["candidates"][0]["content"]["parts"][-1]["text"]
            json_ok = json.loads(txt).get("ok") is True
        except Exception:
            pass
    return True, ms, "JSON mode ok" if json_ok else "JSON mode NOT clean"


def main() -> int:
    print("\nCityMind - API check\n")
    if not KEY:
        bad("GOOGLE_API_KEY is not set (put it in .env)")
        return 1

    print("1) Models visible to your key")
    r = call("models?pageSize=200", method="GET")
    if r.status_code != 200:
        bad(f"models.list failed: HTTP {r.status_code} {r.text[:200]}")
        return 1
    models = {m["name"].split("/")[-1]: m.get("supportedGenerationMethods", []) for m in r.json().get("models", [])}
    ok(f"{len(models)} models listed; {sum('generateContent' in v for v in models.values())} can generate text")

    results = {}
    print("\n2) Live generateContent test")
    for m in dict.fromkeys(MAIN_CANDIDATES + FAST_CANDIDATES):
        if m not in models:
            print(f"  - {m}: not available to this key")
            continue
        good, ms, note = try_model(m)
        results[m] = (good, ms)
        (ok if good else bad)(f"{m}: {ms} ms - {note}")

    print(f"\n3) Embeddings ({EMBED}, {DIM} dims, RETRIEVAL_QUERY)")
    e = call(f"models/{EMBED}:embedContent",
             {"content": {"parts": [{"text": "building height limit"}]}, "taskType": "RETRIEVAL_QUERY",
              "outputDimensionality": DIM})
    embed_ok = e.status_code == 200 and len(e.json().get("embedding", {}).get("values", [])) == DIM
    (ok if embed_ok else bad)(f"{EMBED}: {'returned %d dims' % DIM if embed_ok else f'HTTP {e.status_code} {e.text[:160]}'}")

    if os.getenv("PINECONE_API_KEY"):
        print("\n4) Pinecone")
        try:
            from pinecone import Pinecone
            names = [i.name for i in Pinecone(api_key=os.environ["PINECONE_API_KEY"]).list_indexes()]
            ok(f"connected; indexes: {names or 'none yet (created on first upload)'}")
        except Exception as exc:
            bad(f"Pinecone failed: {str(exc)[:160]}")
    if os.getenv("TAVILY_API_KEY"):
        print("\n5) Tavily")
        t = httpx.post("https://api.tavily.com/search", headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
                       json={"query": "smart city", "max_results": 1}, timeout=20)
        (ok if t.status_code == 200 else bad)(f"HTTP {t.status_code}")

    work = [m for m, (g, _) in results.items() if g]
    main_pick = next((m for m in MAIN_CANDIDATES if m in work), None)
    fast_pick = next((m for m in FAST_CANDIDATES if m in work), None)
    fallbacks = [m for m in work if m not in (main_pick, fast_pick)][:2]
    print("\nRecommendation for your .env")
    if not main_pick:
        bad("No candidate model worked with this key - check billing/quota in Google AI Studio.")
        return 1
    print(f"  GEMINI_MODEL={main_pick}\n  GEMINI_FAST_MODEL={fast_pick or main_pick}\n  GEMINI_FALLBACKS={','.join(fallbacks)}")
    return 0 if embed_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except httpx.HTTPError as exc:
        bad(f"Network problem: {type(exc).__name__}: {exc}")
        print("  Check your internet/firewall - this script must reach generativelanguage.googleapis.com")
        sys.exit(1)