"""All settings in one place. Everything is read from environment variables."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _list(name: str, default: str) -> list:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass
class Settings:
    # --- models (verified against Google's docs, Oct 2026) ---
    # MAIN writes the final answer; FAST does the small JSON jobs (router + checker).
    main_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    fast_model: str = os.getenv("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")
    fallback_models: list = field(default_factory=lambda: _list("GEMINI_FALLBACKS", "gemini-3.6-flash"))
    main_thinking: str = os.getenv("GEMINI_THINKING", "low")        # low | medium | high
    fast_thinking: str = os.getenv("GEMINI_FAST_THINKING", "minimal")
    embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "60"))

    # --- storage / search ---
    index_name: str = os.getenv("PINECONE_INDEX_NAME", "citymind")
    top_k: int = int(os.getenv("TOP_K", "5"))
    cities_cache_ttl: int = int(os.getenv("CITIES_CACHE_TTL", "30"))

    # --- security ---
    allow_uploads: bool = os.getenv("ALLOW_UPLOADS", "true").lower() != "false"
    admin_token: str = os.getenv("ADMIN_TOKEN", "")          # empty = open (same as before)
    cors_origins: list = field(default_factory=lambda: _list("CORS_ORIGINS", "*"))
    ask_rate_per_min: int = int(os.getenv("ASK_RATE_PER_MIN", "30"))


cfg = Settings()


def has_ai() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


def has_db() -> bool:
    return bool(os.getenv("PINECONE_API_KEY"))


def has_web() -> bool:
    return bool(os.getenv("TAVILY_API_KEY"))