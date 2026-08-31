"""Application configuration loaded from environment variables or a local .env file."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ARXIVREC_",
        populate_by_name=True,
        extra="ignore",
    )

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    resend_api_key: SecretStr | None = Field(default=None, validation_alias="RESEND_API_KEY")
    recipient_email: str | None = Field(
        default=None, validation_alias="ARXIVREC_RECIPIENT"
    )
    email_from: str = "ArxivRec <onboarding@resend.dev>"
    research_profile_json: SecretStr | None = Field(
        default=None, validation_alias="ARXIVREC_PROFILE_JSON"
    )
    profile_model: str = "gpt-5.4-mini"
    ranking_model: str = "gpt-5.4-mini"
    embedding_model: str = "text-embedding-3-small"
    database_path: Path = Path("data/arxiv_rec.sqlite3")
    cache_dir: Path = Path("data/cache")
    delivery_state_dir: Path = Path("data/cache/delivery-state")
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    arxiv_user_agent: str = "ArxivRec/0.1 (local research tool)"
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=120)

    @property
    def live_openai_available(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())

    @property
    def email_delivery_available(self) -> bool:
        return bool(
            self.resend_api_key
            and self.resend_api_key.get_secret_value().strip()
            and self.recipient_email
            and self.recipient_email.strip()
        )
