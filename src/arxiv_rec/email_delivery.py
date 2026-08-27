"""Resend-backed delivery for rendered ArxivRec digests."""

from __future__ import annotations

from dataclasses import dataclass
from email.utils import parseaddr
from typing import Any

import httpx

from arxiv_rec.digest import DigestContent

RESEND_EMAILS_URL = "https://api.resend.com/emails"


class EmailDeliveryError(RuntimeError):
    """Raised when an email provider does not accept a digest."""


@dataclass(frozen=True)
class EmailDeliveryReceipt:
    """Minimal provider receipt safe to print in local and CI logs."""

    provider: str
    message_id: str


def _validated_recipient(value: str) -> str:
    _, address = parseaddr(value.strip())
    if not address or "@" not in address or address.startswith("@") or address.endswith("@"):
        raise ValueError("ARXIVREC_RECIPIENT must be a valid email address")
    return address


def _validated_sender(value: str) -> str:
    _, address = parseaddr(value.strip())
    if not address or "@" not in address:
        raise ValueError("ARXIVREC_EMAIL_FROM must contain a valid sender address")
    return value.strip()


def _provider_error(response: httpx.Response) -> EmailDeliveryError:
    if response.status_code in {401, 403}:
        return EmailDeliveryError(
            "Resend authentication or sender validation failed. Check RESEND_API_KEY, "
            "ARXIVREC_EMAIL_FROM, and that the recipient is allowed by the sending domain."
        )
    if response.status_code == 429:
        return EmailDeliveryError("Resend rate-limited the email request; try again later.")
    if response.status_code == 409:
        return EmailDeliveryError(
            "Resend rejected reuse of an idempotency key with different email content."
        )
    return EmailDeliveryError(
        f"Resend email delivery failed with HTTP status {response.status_code}."
    )


def send_digest_via_resend(
    content: DigestContent,
    *,
    api_key: str,
    recipient: str,
    sender: str = "ArxivRec <onboarding@resend.dev>",
    idempotency_key: str | None = None,
    api_url: str = RESEND_EMAILS_URL,
    timeout_seconds: float = 30.0,
    client: httpx.Client | None = None,
) -> EmailDeliveryReceipt:
    """Send one rendered digest through Resend's transactional email API."""
    cleaned_key = api_key.strip()
    if not cleaned_key:
        raise ValueError("RESEND_API_KEY is required for email delivery")
    recipient_address = _validated_recipient(recipient)
    sender_value = _validated_sender(sender)
    headers = {
        "Authorization": f"Bearer {cleaned_key}",
        "Content-Type": "application/json",
    }
    if idempotency_key is not None:
        if not idempotency_key.strip() or len(idempotency_key) > 256:
            raise ValueError("idempotency_key must contain 1 to 256 characters")
        headers["Idempotency-Key"] = idempotency_key
    payload = {
        "from": sender_value,
        "to": [recipient_address],
        "subject": content.subject,
        "html": content.html,
        "text": content.text,
    }

    owns_client = client is None
    active_client = client or httpx.Client(timeout=timeout_seconds)
    try:
        try:
            response = active_client.post(api_url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise EmailDeliveryError(
                "Could not connect to Resend from the current process."
            ) from exc
        if not response.is_success:
            raise _provider_error(response)
        try:
            response_payload: Any = response.json()
        except ValueError as exc:
            raise EmailDeliveryError("Resend returned an unreadable success response.") from exc
        message_id = response_payload.get("id") if isinstance(response_payload, dict) else None
        if not isinstance(message_id, str) or not message_id.strip():
            raise EmailDeliveryError("Resend returned no message ID after accepting the email.")
        return EmailDeliveryReceipt(provider="resend", message_id=message_id)
    finally:
        if owns_client:
            active_client.close()
