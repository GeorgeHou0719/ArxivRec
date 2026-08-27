from datetime import date
from pathlib import Path

import pytest

from arxiv_rec.digest import build_digest, write_digest_preview
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
