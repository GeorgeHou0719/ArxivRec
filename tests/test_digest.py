from datetime import date
from pathlib import Path

import pytest

from arxiv_rec.digest import build_digest, write_digest_preview
from arxiv_rec.models import ArxivFetchReport, FetchedPaper, PaperUpdateKind
from arxiv_rec.pipeline import run_fixture

FIXTURE_PATH = Path("fixtures/cavity_qed_papers.json")


@pytest.mark.parametrize(
    ("threshold", "expected_count", "expected_title"),
    [
        (100, 0, None),
        (90, 1, "Cavity-Mediated Entanglement Between Remote Quantum Network Nodes"),
        (40, 3, "Nonlinear Dynamics in a Multimode Optical Cavity"),
    ],
)
def test_digest_renders_zero_one_and_multiple_selected_papers(
    threshold: int,
    expected_count: int,
    expected_title: str | None,
) -> None:
    run = run_fixture(path=FIXTURE_PATH, relevance_threshold=threshold)

    digest = build_digest(run, digest_date=date(2026, 8, 27))

    assert f"{expected_count} relevant" in digest.subject
    assert f"{expected_count} papers at or above threshold {threshold}" in digest.text
    assert "America/Los_Angeles" in digest.html
    if expected_title is None:
        assert "No papers reached your relevance threshold today." in digest.html
    else:
        assert expected_title in digest.html


def test_digest_only_includes_papers_at_or_above_threshold() -> None:
    run = run_fixture(path=FIXTURE_PATH, relevance_threshold=40)

    digest = build_digest(run, digest_date=date(2026, 8, 27))

    assert "Cavity-Mediated Entanglement Between Remote Quantum Network Nodes" in digest.html
    assert "Fiber-Based Entanglement Distribution" in digest.html
    assert "Nonlinear Dynamics in a Multimode Optical Cavity" in digest.html
    assert "Compact Integrated Photonic Router" not in digest.html
    assert "no fixed paper limit" in digest.html


def test_write_digest_preview_creates_utf8_html(tmp_path: Path) -> None:
    run = run_fixture(path=FIXTURE_PATH, relevance_threshold=40)
    output_path = tmp_path / "nested" / "digest.html"

    digest = write_digest_preview(
        run,
        output_path,
        digest_date=date(2026, 8, 27),
    )

    assert output_path.read_text(encoding="utf-8") == digest.html
    assert digest.subject == "ArxivRec · Aug 27 · 3 relevant papers · top score 96"


def test_digest_includes_the_complete_arxiv_abstract_without_truncation() -> None:
    run = run_fixture(path=FIXTURE_PATH, relevance_threshold=40)
    first = run.ranking.items[0]
    long_abstract = " ".join(f"abstract-section-{index}" for index in range(100))
    updated_paper = first.paper.model_copy(update={"abstract": long_abstract})
    updated_first = first.model_copy(update={"paper": updated_paper})
    updated_ranking = run.ranking.model_copy(
        update={"items": (updated_first, *run.ranking.items[1:])}
    )
    updated_run = run.model_copy(update={"ranking": updated_ranking})

    digest = build_digest(updated_run, digest_date=date(2026, 8, 27))

    assert long_abstract in digest.html
    assert f"Abstract: {long_abstract}" in digest.text


@pytest.mark.parametrize("threshold", [40, 100])
def test_catchup_digest_does_not_claim_complete_daily_coverage(threshold: int) -> None:
    run = run_fixture(path=FIXTURE_PATH, relevance_threshold=threshold)
    papers = tuple(item.paper for item in run.recall.items)
    fetch = ArxivFetchReport(
        items=tuple(
            FetchedPaper(paper=paper, update_kind=PaperUpdateKind.NEW_SUBMISSION)
            for paper in papers
        ),
        categories=run.profile.arxiv_categories,
        cutoff=min(paper.updated_at for paper in papers),
        api_total_results=1200,
        scanned_count=510,
        page_size=50,
        safety_cap=500,
        complete_through_cutoff=False,
        truncated=True,
        catchup_batch=True,
    )
    digest = build_digest(run.model_copy(update={"fetch_report": fetch}))

    assert digest.subject.startswith("[Backlog catch-up]")
    assert "not a complete scan of today's updates" in digest.text
    assert "More papers remain" in digest.html
    if threshold == 100:
        assert "No papers reached your relevance threshold in this batch." in digest.html
        assert "The daily scan completed successfully." not in digest.html
