from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from arxiv_rec.cli import main
from arxiv_rec.delivery_state import DeliveryStateStore
from arxiv_rec.email_delivery import EmailDeliveryReceipt
from arxiv_rec.models import ArxivFetchReport, FetchedPaper, PaperUpdateKind, RunMode
from arxiv_rec.pipeline import load_fixture, run_fixture


def test_daily_checkpoint_advances_only_after_email_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile, papers, _ = load_fixture(Path("fixtures/cavity_qed_papers.json"))
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(profile.model_dump_json(), encoding="utf-8")
    state_dir = tmp_path / "delivery-state"
    settings = SimpleNamespace(delivery_state_dir=state_dir)
    fetch_report = ArxivFetchReport(
        items=(
            FetchedPaper(
                paper=papers[0],
                update_kind=PaperUpdateKind.NEW_SUBMISSION,
            ),
        ),
        categories=profile.arxiv_categories,
        cutoff=papers[0].updated_at - timedelta(days=7),
        api_total_results=1,
        scanned_count=1,
        page_size=50,
        safety_cap=500,
        complete_through_cutoff=True,
        truncated=False,
    )
    fixture_run = run_fixture(
        path=Path("fixtures/cavity_qed_papers.json"),
        candidate_threshold=0.0,
        relevance_threshold=40,
    )
    daily_run = fixture_run.model_copy(
        update={
            "mode": RunMode.LIVE,
            "fetched_count": 1,
            "fetch_report": fetch_report,
        }
    )
    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return daily_run

    def failed_send(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("simulated Resend failure")

    monkeypatch.setattr("arxiv_rec.cli.AppSettings", lambda: settings)
    monkeypatch.setattr("arxiv_rec.cli.run_live_with_profile", fake_run)
    monkeypatch.setattr("arxiv_rec.cli._send_digest", failed_send)

    command = [
        "send-daily-email",
        "--profile-file",
        str(profile_path),
        "--allow-duplicate-email",
    ]
    with pytest.raises(RuntimeError, match="simulated Resend failure"):
        main(command)

    checkpoint = DeliveryStateStore(state_dir).load(profile)
    assert checkpoint.last_successful_updated_at is None
    assert not checkpoint.path.exists()
    assert captured["published_after"] is not None
    assert captured["processed_versions"] == frozenset()

    monkeypatch.setattr(
        "arxiv_rec.cli._send_digest",
        lambda *args, **kwargs: EmailDeliveryReceipt(
            provider="resend", message_id="accepted"
        ),
    )

    assert main(command) == 0
    checkpoint = DeliveryStateStore(state_dir).load(profile)
    assert checkpoint.last_successful_updated_at == papers[0].updated_at
