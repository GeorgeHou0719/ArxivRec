from pathlib import Path

from arxiv_rec.settings import AppSettings


def test_settings_are_safe_without_api_key(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        openai_api_key=None,
        database_path=tmp_path / "test.sqlite3",
        cache_dir=tmp_path / "cache",
    )

    assert settings.live_openai_available is False
    assert settings.email_delivery_available is False
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.database_path == tmp_path / "test.sqlite3"


def test_email_delivery_configuration_requires_key_and_recipient(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        resend_api_key="re_test_secret",
        recipient_email="researcher@stanford.edu",
        database_path=tmp_path / "test.sqlite3",
        cache_dir=tmp_path / "cache",
    )

    assert settings.email_delivery_available is True
    assert settings.resend_api_key is not None
    assert settings.resend_api_key.get_secret_value() == "re_test_secret"
