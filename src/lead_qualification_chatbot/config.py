from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "gemini", "groq", "openrouter", "mistral", "cerebras", "ollama"]
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "resources"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    llm_provider: Provider
    llm_model: str
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str | None = None

    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_api_key: SecretStr = SecretStr("")

    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "lead_qualification_chatbot"
    langsmith_workspace_id: str | None = None
    langsmith_endpoint: str | None = None

    office_phone: str = "020 7946 0100"
    data_dir: Path = DEFAULT_DATA_DIR

    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0
    llm_max_attempts: int = 3

    embedding_base_url: str | None = None
    chroma_dir: Path | None = Path(".chroma")
    faq_top_k: int = 3
    faq_max_distance: float = 0.6

    max_turns: int = 60
    max_off_topic: int = 3

    api_key: SecretStr = SecretStr("")  # empty = no auth (dev only)
    cors_origins: list[str] = []  # JSON list in env, e.g. ["https://example.com"]
    rate_limit_per_minute: int = 30  # per client IP; 0 disables
    request_timeout_s: float = 90.0

    leads_file: Path = Path("leads/leads.jsonl")


settings = Settings()
