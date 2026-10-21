# 🏙️ CityMind

Smart-city RAG assistant. Pipeline: `Router → Retriever → Checker → Fallback → Report`
(small talk skips to Report; if the documents are enough, Fallback is skipped).

```
main.py            entry point (uvicorn main:app) - Render/Vercel configs unchanged
app/config.py      every setting, read from env
app/llm.py         Gemini calls, model fallback chain, JSON parsing, model quirks
app/store.py       Pinecone + city matching (cached)
app/graph.py       the 5 LangGraph nodes
app/api.py         routes, rate limit, admin token
app/ingest.py      chunking + stable ids
scripts/check_models.py   test YOUR key and print which models to use
tests/             29 tests (no network / no keys needed)
public/index.html  your existing page (API is unchanged)
```

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # add GOOGLE_API_KEY (+ PINECONE_API_KEY)
python scripts/check_models.py  # <- tests the key, tells you which models to set
pytest -q                       # offline tests
uvicorn main:app --reload
```
After deploying: `GET /api/health` (cheap) and `GET /api/health/llm` (live model probe, send `X-Admin-Token` if you set one).

## What changed vs the original
| Area | Before | Now |
|---|---|---|
| Model | `gemini-3.6-flash` for everything, `temperature=0.2` | `gemini-3.8-flash` for answers, `gemini-3.5-flash-lite` for router/checker, automatic fallback chain; no `temperature` on Gemini 3+ (deprecated) |
| JSON steps | regex-parsed free text | JSON mode (`response_mime_type`) + tolerant parser |
| Embeddings | one embedder for queries and documents | `RETRIEVAL_QUERY` for questions, `RETRIEVAL_DOCUMENT` for uploads |
| Retrieval | cities searched one by one | embed once, search cities in parallel, duplicate passages dropped |
| Pinecone | `list_cities()` hit on every question (twice) | 30 s cache, invalidated on upload/delete |
| Security | anyone could upload/delete; CORS `*` | optional `ADMIN_TOKEN`, `CORS_ORIGINS`, per-IP rate limit on `/ask`, sources wrapped as data (prompt-injection guard) |
| City matching | `Mumbay` could not find `mumbai_india` | fuzzy match also compares the city part alone |
| Ops | none | `/api/health`, `/api/health/llm`, logging, tests, `check_models.py` |

`ADMIN_TOKEN`: your current `public/index.html` does not send it, so leave it empty until you add an
`X-Admin-Token` header to the upload/delete `fetch` calls (otherwise uploads will get 401).
