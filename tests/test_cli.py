import json
from pathlib import Path

from arxiv_rec.cli import main
from arxiv_rec.email_delivery import EmailDeliveryReceipt


def test_inspect_recall_cli_emits_machine_readable_diagnostics(capsys) -> None:
    exit_code = main(
        [
            "inspect-recall",
            "--fixture",
            "fixtures/cavity_qed_papers.json",
            "--threshold",
            "0.65",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(payload["items"]) == 4
    assert [item["paper"]["arxiv_id"] for item in payload["items"] if item["included"]] == [
        "2608.10001",
        "2608.10002",
    ]


def test_preview_digest_cli_writes_local_html(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "digest.html"

    exit_code = main(
        [
            "preview-digest",
            "--fixture",
            "fixtures/cavity_qed_papers.json",
            "--output",
            str(output_path),
            "--relevance-threshold",
            "40",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output_path.exists()
    assert "3 relevant papers" in output
    assert "Research worth your attention" in output_path.read_text(encoding="utf-8")


def test_verification_email_cli_can_be_run_immediately_without_live_apis(
    monkeypatch, capsys
) -> None:
    def fake_send(*args, **kwargs) -> EmailDeliveryReceipt:
        del args, kwargs
        return EmailDeliveryReceipt(provider="resend", message_id="email_fixture_test")

    monkeypatch.setattr("arxiv_rec.cli._send_digest", fake_send)

    exit_code = main(
        [
            "send-verification-email",
            "--fixture",
            "fixtures/cavity_qed_papers.json",
            "--relevance-threshold",
            "40",
        ]
    )

    assert exit_code == 0
    assert "email_fixture_test" in capsys.readouterr().out
