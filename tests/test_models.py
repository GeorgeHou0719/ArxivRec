from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from arxiv_rec.models import (
    Confidence,
    Paper,
    RelevanceAssessment,
    RelevanceLabel,
    ResearchProfile,
    RunMode,
    RunRequest,
    canonicalize_arxiv_id,
    label_for_score,
)


def sample_profile() -> ResearchProfile:
    return ResearchProfile(
        source_text="I study cavity QED interfaces for remote quantum-network nodes.",
        core_topics=["cavity QED", "cavity QED"],
        required_intersections=[["optical cavity", "quantum network"]],
        arxiv_categories=["quant-ph", "physics.optics"],
    )


def test_canonicalize_arxiv_identifier_and_version() -> None:
    assert canonicalize_arxiv_id("https://arxiv.org/abs/2608.12345v2") == ("2608.12345", 2)
    assert canonicalize_arxiv_id("quant-ph/0601001v3") == ("quant-ph/0601001", 3)


def test_paper_normalizes_text_and_builds_links() -> None:
    paper = Paper(
        arxiv_id="https://arxiv.org/abs/2608.12345v2",
        version=2,
        title="  A   useful\n title ",
        abstract="An   abstract.",
        authors=["A. Researcher", "A. Researcher"],
        categories=["quant-ph", "physics.optics"],
        primary_category="quant-ph",
        published_at=datetime(2026, 8, 18, tzinfo=UTC),
        updated_at=datetime(2026, 8, 19, tzinfo=UTC),
    )

    assert paper.arxiv_id == "2608.12345"
    assert paper.version == 2
    assert paper.title == "A useful title"
    assert paper.authors == ("A. Researcher",)
    assert paper.abs_url == "https://arxiv.org/abs/2608.12345"


def test_paper_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        Paper(
            arxiv_id="2608.12345",
            title="Title",
            abstract="Abstract",
            authors=["Author"],
            categories=["quant-ph"],
            primary_category="quant-ph",
            published_at=datetime(2026, 8, 18),
            updated_at=datetime(2026, 8, 18, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (100, RelevanceLabel.ESSENTIAL),
        (90, RelevanceLabel.ESSENTIAL),
        (89, RelevanceLabel.HIGHLY_RELEVANT),
        (75, RelevanceLabel.HIGHLY_RELEVANT),
        (74, RelevanceLabel.RELATED),
        (55, RelevanceLabel.RELATED),
        (54, RelevanceLabel.PERIPHERAL),
        (35, RelevanceLabel.PERIPHERAL),
        (34, RelevanceLabel.IRRELEVANT),
        (0, RelevanceLabel.IRRELEVANT),
    ],
)
def test_relevance_boundaries(score: int, expected: RelevanceLabel) -> None:
    assert label_for_score(score) is expected


def test_assessment_rejects_inconsistent_label() -> None:
    with pytest.raises(ValidationError, match="does not match score"):
        RelevanceAssessment(
            relevance_score=92,
            label=RelevanceLabel.RELATED,
            confidence=Confidence.HIGH,
            reason="The score and label intentionally disagree for this test.",
        )


def test_profile_normalizes_and_deduplicates_topics() -> None:
    profile = sample_profile()
    assert profile.core_topics == ("cavity QED",)
    assert profile.required_intersections == (("optical cavity", "quantum network"),)


def test_fixture_request_requires_fixture_path() -> None:
    with pytest.raises(ValidationError, match="fixture_path"):
        RunRequest(profile=sample_profile(), mode=RunMode.FIXTURE)
