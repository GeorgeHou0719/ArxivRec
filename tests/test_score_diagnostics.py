from pathlib import Path

import pytest

from arxiv_rec.feedback import FeedbackRecord, analyze_score_diagnostics, profile_hash
from arxiv_rec.pipeline import run_fixture


def test_score_diagnostics_reports_boundaries_and_user_calibration() -> None:
    result = run_fixture(path=Path("fixtures/cavity_qed_papers.json"))
    profile = result.profile
    feedback = [
        FeedbackRecord(
            profile_hash=profile_hash(profile),
            arxiv_id=item.paper.arxiv_id,
            title=item.paper.title,
            predicted_score=item.assessment.relevance_score,
            user_score=user_score,
            updated_at=result.completed_at,
        )
        for item, user_score in zip(result.ranking.items, [96, 80, 50, 10], strict=True)
    ]

    diagnostics = analyze_score_diagnostics(result.ranking.items, feedback)

    assert diagnostics.prediction_count == 4
    assert diagnostics.unique_score_count == 4
    assert diagnostics.boundary_count == 0
    assert diagnostics.feedback_count == 4
    assert diagnostics.mean_absolute_error == pytest.approx(1.75)
    assert diagnostics.mean_bias == pytest.approx(0.25)
    assert diagnostics.band_agreement == pytest.approx(1.0)
