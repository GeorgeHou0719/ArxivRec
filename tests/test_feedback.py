from pathlib import Path

from arxiv_rec.feedback import FeedbackStore
from arxiv_rec.pipeline import load_fixture


def test_feedback_store_upserts_one_user_score_per_profile_and_paper(tmp_path: Path) -> None:
    profile, papers, _ = load_fixture(Path("fixtures/cavity_qed_papers.json"))
    store = FeedbackStore(tmp_path / "feedback.sqlite3")

    store.save(
        profile=profile,
        paper=papers[0],
        predicted_score=96,
        user_score=90,
        notes="First pass",
    )
    store.save(
        profile=profile,
        paper=papers[0],
        predicted_score=96,
        user_score=94,
        notes="Reconsidered",
    )

    records = store.list_for_profile(profile)
    assert len(records) == 1
    assert records[0].user_score == 94
    assert records[0].notes == "Reconsidered"

