"""CityMind entry point. `uvicorn main:app` keeps working exactly as before (Render / Vercel unchanged)."""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import api
from app.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="CityMind")
app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_methods=["*"], allow_headers=["*"])
app.include_router(api)

_public = Path(__file__).resolve().parent / "public"
if _public.exists():                                   # serve the page from the same server
    app.mount("/", StaticFiles(directory=str(_public), html=True), name="site")