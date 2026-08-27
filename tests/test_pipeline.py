from pathlib import Path

from arxiv_rec.models import RunMode
from arxiv_rec.pipeline import run_fixture


def test_fixture_pipeline_preserves_intermediate_results() -> None:
    result = run_fixture(
        path=Path("fixtures/cavity_qed_papers.json"),
        candidate_threshold=0.0,
        relevance_threshold=75,
    )

    assert result.mode is RunMode.FIXTURE
    assert result.fetched_count == 4
    assert len(result.recall.items) == 4
    assert len(result.recall.candidates) == 4
    assert [item.paper.arxiv_id for item in result.ranking.selected] == [
        "2608.10001",
        "2608.10002",
    ]
    assert result.completed_at >= result.started_at

