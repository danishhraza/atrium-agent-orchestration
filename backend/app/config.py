"""Server settings, overridable from backend/.env (see .env.example).

Model access is bring-your-own-key: each user stores their own OpenRouter key (encrypted),
so the server itself holds no provider credentials.
"""
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")  # also exposes LANGFUSE_* to the Langfuse SDK


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATRIUM_", extra="ignore")

    data_dir: str = str(BACKEND_DIR / "data")
    database_url: str = ""  # default: SQLite file in data_dir. Set a postgres:// URL to ship.

    # --- Auth (Clerk). The publishable key is public; it is all the backend needs to verify tokens.
    clerk_publishable_key: str = ""
    dev_auth: bool = False          # local development only: skip login, act as a fixed dev user
    allowed_origins: str = "http://localhost:5173"  # comma-separated, for CORS during frontend dev

    # --- Encryption key for user API keys at rest (Fernet). Required unless dev_auth is on.
    secret_key: str = ""

    # --- Default OpenRouter model per tier. Each user can override these in Settings.
    tier_fast: str = "anthropic/claude-haiku-5.5"
    tier_balanced: str = "anthropic/claude-sonnet-5.5"
    tier_smart: str = "anthropic/claude-opus-5.5"

    # OpenRouter embedding model, or "local" for an offline lexical embedder (no key, lower quality).
    embedding_model: str = "openai/text-embedding-3-small"

    orchestrator_name: str = "Jarvis"
    max_steps: int = 8          # LLM turns per agent run
    max_handoffs: int = 4       # per task, stops ping-pong
    max_consult_depth: int = 2  # agent -> agent -> agent
    pass_score: float = 0.7     # judge score at or above this counts as a success

    @property
    def db_url(self) -> str:
        url = self.database_url or f"sqlite:///{self.data_dir}/atrium.db"
        for prefix in ("postgres://", "postgresql://"):  # what hosting providers hand out
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url

    @property
    def default_tiers(self) -> dict[str, str]:
        return {"fast": self.tier_fast, "balanced": self.tier_balanced, "smart": self.tier_smart}


settings = Settings()
TIERS = ("fast", "balanced", "smart")  # cheapest first
