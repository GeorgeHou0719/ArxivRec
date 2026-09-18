from datetime import UTC, datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

import httpx
import pytest

from arxiv_rec.arxiv_client import ArxivClient, ArxivClientError
from arxiv_rec.delivery_state import DeliveryStateStore, paper_version_key
from arxiv_rec.models import Paper
from arxiv_rec.pipeline import load_fixture

BASE = datetime(2026, 9, 4, tzinfo=UTC)


def make_papers(count: int, *, tied: bool = False) -> list[Paper]:
    _, template, _ = load_fixture(Path("fixtures/cavity_qed_papers.json"))
    return [
        template[0].model_copy(
            update={
                "arxiv_id": f"2609.{index:05d}",
                "version": 1,
                "published_at": BASE + timedelta(minutes=0 if tied else index),
                "updated_at": BASE + timedelta(minutes=0 if tied else index),
            }
        )
        for index in range(count)
    ]


class MockFeed:
    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.params["sortBy"] == "lastUpdatedDate"
        assert request.url.params["sortOrder"] == "descending"
        ordered = sorted(
            self.papers,
            key=lambda paper: (paper.updated_at, paper.arxiv_id),
            reverse=True,
        )
        start = int(request.url.params["start"])
        count = int(request.url.params["max_results"])
        entries = []
        for paper in ordered[start : start + count]:
            entries.append(
                "<entry>"
                f"<id>https://arxiv.org/abs/{paper.arxiv_id}v{paper.version}</id>"
                f"<title>{escape(paper.title)}</title>"
                f"<summary>{escape(paper.abstract)}</summary>"
                f"<published>{paper.published_at.isoformat()}</published>"
                f"<updated>{paper.updated_at.isoformat()}</updated>"
                "<author><name>Test Author</name></author>"
                '<category term="quant-ph"/>'
                '<arxiv:primary_category term="quant-ph"/>'
                "</entry>"
            )
        payload = (
            '<feed xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:arxiv="http://arxiv.org/schemas/atom" '
            'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
            f"<opensearch:totalResults>{len(ordered)}</opensearch:totalResults>"
            + "".join(entries)
            + "</feed>"
        )
        return httpx.Response(200, text=payload, request=request)


def make_client(tmp_path: Path, feed: MockFeed, *, page_size: int = 50) -> ArxivClient:
    return ArxivClient(
        api_url="https://example.test/arxiv",
        user_agent="ArxivRec/local-tests",
        cache_dir=tmp_path / "cache",
        cache_ttl=timedelta(0),
        page_size=page_size,
        min_request_interval_seconds=0,
        client=httpx.Client(transport=httpx.MockTransport(feed.handle)),
    )


def test_1200_backlogged_versions_are_delivered_in_three_safe_batches(tmp_path: Path) -> None:
    profile, _, _ = load_fixture(Path("fixtures/cavity_qed_papers.json"))
    papers = make_papers(1200)
    ancient = papers[0].model_copy(
        update={
            "arxiv_id": "2501.00001",
            "published_at": BASE - timedelta(days=50),
            "updated_at": BASE - timedelta(days=50),
        }
    )
    feed = MockFeed([*papers, ancient])
    client = make_client(tmp_path, feed)
    store = DeliveryStateStore(tmp_path / "state")
    delivered: set[str] = set()
    counts: list[int] = []
    flags: list[bool] = []
    for _ in range(3):
        checkpoint = store.load(profile)
        report = client.fetch_incremental_report(
            categories=profile.arxiv_categories,
            published_after=checkpoint.retrieval_cutoff(
                now=BASE + timedelta(days=3), initial_lookback_days=7
            ),
            max_results=500,
            processed_versions=frozenset(checkpoint.processed_versions),
        )
        keys = {paper_version_key(paper) for paper in report.papers}
        assert not keys & delivered
        assert keys == {
            paper_version_key(paper)
            for paper in papers[len(delivered) : len(delivered) + len(keys)]
        }
        delivered.update(keys)
        counts.append(len(report.papers))
        flags.append(report.truncated)
        checkpoint.commit(report.papers)  # Simulate email acceptance.

    assert counts == [500, 500, 200]
    assert flags == [True, True, False]
    assert delivered == {paper_version_key(paper) for paper in papers}
    assert store.load(profile).last_successful_updated_at == papers[-1].updated_at
    assert len(feed.requests) < 120


def test_new_insertions_and_a_revision_do_not_shift_persisted_progress(tmp_path: Path) -> None:
    papers = make_papers(9)
    feed = MockFeed(list(papers))
    client = make_client(tmp_path, feed, page_size=2)
    first = client.fetch_incremental_report(
        categories=["quant-ph"], published_after=BASE, max_results=3
    )
    assert {paper.arxiv_id for paper in first.papers} == {
        paper.arxiv_id for paper in papers[:3]
    }
    delivered = {paper_version_key(paper) for paper in first.papers}
    revision = papers[0].model_copy(
        update={"version": 2, "updated_at": BASE + timedelta(days=1)}
    )
    inserted = papers[0].model_copy(
        update={
            "arxiv_id": "2609.90000",
            "published_at": BASE + timedelta(days=2),
            "updated_at": BASE + timedelta(days=2),
        }
    )
    feed.papers = [*papers[1:], revision, inserted]
    cursor = max(paper.updated_at for paper in first.papers)
    for _ in range(3):
        report = client.fetch_incremental_report(
            categories=["quant-ph"],
            published_after=cursor - timedelta(hours=24),
            max_results=3,
            processed_versions=frozenset(delivered),
        )
        keys = {paper_version_key(paper) for paper in report.papers}
        assert not keys & delivered
        delivered.update(keys)
        cursor = max(cursor, *(paper.updated_at for paper in report.papers))
    assert delivered == {
        paper_version_key(paper) for paper in [papers[0], *feed.papers]
    }


def test_timestamp_ties_can_cross_batches_without_stalling_or_skipping(tmp_path: Path) -> None:
    papers = make_papers(9, tied=True)
    feed = MockFeed(papers)
    client = make_client(tmp_path, feed, page_size=2)
    delivered: set[str] = set()
    for _ in range(3):
        report = client.fetch_incremental_report(
            categories=["quant-ph"],
            published_after=BASE - timedelta(hours=24),
            max_results=3,
            processed_versions=frozenset(delivered),
        )
        keys = {paper_version_key(paper) for paper in report.papers}
        assert len(keys) == 3
        assert not keys & delivered
        delivered.update(keys)
    assert delivered == {paper_version_key(paper) for paper in papers}


@pytest.mark.parametrize("processed_count", [650, 700])
def test_overlap_larger_than_cap_does_not_block_new_work(
    tmp_path: Path, processed_count: int
) -> None:
    papers = make_papers(700)
    feed = MockFeed(papers)
    client = make_client(tmp_path, feed)
    report = client.fetch_incremental_report(
        categories=["quant-ph"],
        published_after=BASE,
        max_results=500,
        processed_versions=frozenset(
            paper_version_key(paper) for paper in papers[:processed_count]
        ),
    )
    assert {paper_version_key(paper) for paper in report.papers} == {
        paper_version_key(paper) for paper in papers[processed_count:]
    }
    assert report.complete_through_cutoff
    assert not report.truncated


@pytest.mark.parametrize("eligible_count", [0, 1, 2, 3, 4, 5, 8])
def test_cutoff_boundaries_and_exact_batch_sizes(tmp_path: Path, eligible_count: int) -> None:
    papers = make_papers(eligible_count)
    ancient = make_papers(1)[0].model_copy(
        update={
            "arxiv_id": "2501.00001",
            "published_at": BASE - timedelta(days=50),
            "updated_at": BASE - timedelta(days=50),
        }
    )
    feed = MockFeed([*papers, ancient])
    client = make_client(tmp_path, feed, page_size=2)
    delivered: set[str] = set()
    for _ in range(6):
        report = client.fetch_incremental_report(
            categories=["quant-ph"],
            published_after=BASE,
            max_results=2,
            processed_versions=frozenset(delivered),
        )
        keys = {paper_version_key(paper) for paper in report.papers}
        assert len(keys) <= 2
        assert not keys & delivered
        delivered.update(keys)
        if not report.truncated:
            break
    assert delivered == {paper_version_key(paper) for paper in papers}


def test_changed_result_set_aborts_without_moving_delivery_state(tmp_path: Path) -> None:
    profile, _, _ = load_fixture(Path("fixtures/cavity_qed_papers.json"))
    checkpoint = DeliveryStateStore(tmp_path / "state").load(profile)
    feed = MockFeed(make_papers(10))
    original_handler = feed.handle

    def changed_handler(request: httpx.Request) -> httpx.Response:
        if len(feed.requests) == 2:
            feed.papers.pop()
        return original_handler(request)

    feed.handle = changed_handler
    client = make_client(tmp_path, feed, page_size=2)
    with pytest.raises(ArxivClientError, match="results changed"):
        client.fetch_incremental_report(
            categories=profile.arxiv_categories,
            published_after=BASE,
            max_results=3,
        )
    assert checkpoint.last_successful_updated_at is None
    assert not checkpoint.path.exists()


def test_catchup_outpaces_new_daily_arrivals_without_losing_older_papers(tmp_path: Path) -> None:
    profile, _, _ = load_fixture(Path("fixtures/cavity_qed_papers.json"))
    papers = make_papers(1400)
    feed = MockFeed(papers[:1200])
    client = make_client(tmp_path, feed)
    store = DeliveryStateStore(tmp_path / "state")
    delivered: set[str] = set()
    counts = []
    for day in range(3):
        if day:
            feed.papers = papers[: 1200 + day * 100]
        checkpoint = store.load(profile)
        report = client.fetch_incremental_report(
            categories=profile.arxiv_categories,
            published_after=checkpoint.retrieval_cutoff(
                now=BASE + timedelta(days=3 + day), initial_lookback_days=7
            ),
            max_results=500,
            processed_versions=frozenset(checkpoint.processed_versions),
        )
        keys = {paper_version_key(paper) for paper in report.papers}
        assert not keys & delivered
        delivered.update(keys)
        counts.append(len(keys))
        checkpoint.commit(report.papers)
    assert counts == [500, 500, 400]
    assert delivered == {paper_version_key(paper) for paper in papers}
    assert report.complete_through_cutoff
