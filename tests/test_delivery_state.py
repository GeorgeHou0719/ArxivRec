from datetime import UTC, datetime, timedelta
from pathlib import Path

from arxiv_rec.delivery_state import DeliveryStateStore, paper_version_key
from arxiv_rec.models import ArxivFetchReport, FetchedPaper, PaperUpdateKind
from arxiv_rec.pipeline import load_fixture


def _fixture(tmp_path: Path):
    profile, papers, _ = load_fixture(Path("fixtures/cavity_qed_papers.json"))
    return profile, papers, DeliveryStateStore(tmp_path / "state")


def test_new_profile_uses_initial_backfill_and_empty_commit_does_not_advance(
    tmp_path: Path,
) -> None:
    profile, _, store = _fixture(tmp_path)
    checkpoint = store.load(profile)
    now = datetime(2026, 9, 1, 12, tzinfo=UTC)

    assert checkpoint.retrieval_cutoff(now=now, initial_lookback_days=7) == now - timedelta(
        days=7
    )
    assert checkpoint.commit(()) is checkpoint
    assert not checkpoint.path.exists()


def test_successful_commit_reloads_cursor_and_uses_overlap(tmp_path: Path) -> None:
    profile, papers, store = _fixture(tmp_path)
    checkpoint = store.load(profile).commit(tuple(papers[:2]))
    reloaded = store.load(profile)

    newest = max(paper.updated_at for paper in papers[:2])
    assert checkpoint.path.exists()
    assert reloaded.last_successful_updated_at == newest
    assert reloaded.retrieval_cutoff(
        now=datetime(2026, 9, 1, tzinfo=UTC), initial_lookback_days=7
    ) == newest - timedelta(hours=24)
    assert set(reloaded.processed_versions) == {
        paper_version_key(paper) for paper in papers[:2]
    }


def test_overlap_filters_same_versions_but_keeps_a_revision(tmp_path: Path) -> None:
    profile, papers, store = _fixture(tmp_path)
    original = papers[0]
    checkpoint = store.load(profile).commit((original,))
    revision = original.model_copy(
        update={
            "version": 2,
            "updated_at": original.updated_at + timedelta(days=1),
        }
    )
    report = ArxivFetchReport(
        items=(
            FetchedPaper(paper=revision, update_kind=PaperUpdateKind.REVISED_VERSION),
            FetchedPaper(paper=original, update_kind=PaperUpdateKind.NEW_SUBMISSION),
        ),
        categories=profile.arxiv_categories,
        cutoff=original.updated_at - timedelta(days=1),
        api_total_results=2,
        scanned_count=2,
        page_size=50,
        safety_cap=500,
        complete_through_cutoff=True,
        truncated=False,
    )

    filtered = checkpoint.filter_unseen(report)

    assert filtered.papers == (revision,)


def test_changed_profile_gets_an_independent_checkpoint(tmp_path: Path) -> None:
    profile, papers, store = _fixture(tmp_path)
    first = store.load(profile).commit((papers[0],))
    changed = profile.model_copy(update={"source_text": profile.source_text + " More detail."})
    second = store.load(changed)

    assert second.path != first.path
    assert second.last_successful_updated_at is None
    assert second.processed_versions == {}
