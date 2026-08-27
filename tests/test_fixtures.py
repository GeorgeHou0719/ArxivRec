import json
from pathlib import Path

from arxiv_rec.models import Paper, ResearchProfile


def test_cavity_qed_fixture_matches_domain_schemas() -> None:
    fixture_path = Path("fixtures/cavity_qed_papers.json")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    profile = ResearchProfile.model_validate(payload["profile"])
    papers = [Paper.model_validate(item) for item in payload["papers"]]

    assert "cavity-mediated quantum interconnects" in profile.core_topics
    assert len(papers) == 4
    assert len({paper.arxiv_id for paper in papers}) == 4
    assert papers[0].version == 1

