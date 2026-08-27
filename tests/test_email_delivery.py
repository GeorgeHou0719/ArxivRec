import json

import httpx
import pytest

from arxiv_rec.digest import DigestContent
from arxiv_rec.email_delivery import EmailDeliveryError, send_digest_via_resend


def _content() -> DigestContent:
    return DigestContent(
        subject="ArxivRec daily test",
        html="<strong>Full abstract</strong>",
        text="Full abstract\n",
    )


def test_resend_delivery_sends_html_text_and_idempotency_header() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["Authorization"]
        captured["idempotency"] = request.headers["Idempotency-Key"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "email_123"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = send_digest_via_resend(
            _content(),
            api_key="re_test_secret",
            recipient="researcher@stanford.edu",
            idempotency_key="arxivrec-daily-2026-08-27",
            client=client,
        )

    assert receipt.provider == "resend"
    assert receipt.message_id == "email_123"
    assert captured["authorization"] == "Bearer re_test_secret"
    assert captured["idempotency"] == "arxivrec-daily-2026-08-27"
    assert captured["payload"] == {
        "from": "ArxivRec <onboarding@resend.dev>",
        "to": ["researcher@stanford.edu"],
        "subject": "ArxivRec daily test",
        "html": "<strong>Full abstract</strong>",
        "text": "Full abstract\n",
    }


def test_resend_authentication_error_is_actionable_and_sanitized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(403, json={"message": "provider internals"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(EmailDeliveryError) as error:
            send_digest_via_resend(
                _content(),
                api_key="re_test_secret",
                recipient="researcher@stanford.edu",
                client=client,
            )

    assert "authentication or sender validation failed" in str(error.value)
    assert "provider internals" not in str(error.value)
    assert "re_test_secret" not in str(error.value)


def test_resend_connection_error_does_not_expose_low_level_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private network details", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(EmailDeliveryError) as error:
            send_digest_via_resend(
                _content(),
                api_key="re_test_secret",
                recipient="researcher@stanford.edu",
                client=client,
            )

    assert "Could not connect to Resend" in str(error.value)
    assert "private network details" not in str(error.value)


def test_resend_idempotency_conflict_is_actionable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(409, json={"message": "conflict"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(EmailDeliveryError) as error:
            send_digest_via_resend(
                _content(),
                api_key="re_test_secret",
                recipient="researcher@stanford.edu",
                idempotency_key="same-day-key",
                client=client,
            )
    assert "idempotency key" in str(error.value)
