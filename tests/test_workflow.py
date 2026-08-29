from pathlib import Path


def test_daily_digest_workflow_has_agreed_schedule_and_manual_verification() -> None:
    workflow = Path(".github/workflows/daily-digest.yml").read_text(encoding="utf-8")

    assert 'cron: "7 5 * * 1-5"' in workflow
    assert 'timezone: "America/Los_Angeles"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "send-verification-email" in workflow
    assert "send-daily-email" in workflow
    assert "--lookback-days 1" in workflow
    assert "--relevance-threshold 40" in workflow
    assert "Run manual live recommendation and send email" in workflow
    assert "--allow-duplicate-email" in workflow
    assert "Run scheduled daily recommendation and send email" in workflow
