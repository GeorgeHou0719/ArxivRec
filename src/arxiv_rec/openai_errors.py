"""Safe, actionable messages for OpenAI failures shown in the local UI."""

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)


def openai_error_message(operation: str, error: OpenAIError) -> str:
    if isinstance(error, AuthenticationError):
        return f"OpenAI {operation} authentication failed. Check OPENAI_API_KEY in .env."
    if isinstance(error, RateLimitError):
        return (
            f"OpenAI {operation} was rate-limited or the API quota was reached. "
            "Check API usage and billing, then retry."
        )
    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return (
            f"OpenAI {operation} could not connect from the app process. "
            "Check network access and retry."
        )
    return f"OpenAI {operation} failed. Retry or inspect the local server output."
