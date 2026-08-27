import httpx
from openai import APIConnectionError

from arxiv_rec.openai_errors import openai_error_message


def test_connection_error_is_actionable_without_exposing_exception_details() -> None:
    error = APIConnectionError(
        message="low-level connection details",
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )

    message = openai_error_message("profile analysis", error)

    assert "could not connect from the app process" in message
    assert "low-level connection details" not in message
