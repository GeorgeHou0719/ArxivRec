from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from arxiv_rec.arxiv_client import (
    ArxivClient,
    ArxivClientError,
    ArxivParseError,
    build_category_query,
    parse_atom_feed,
)
from arxiv_rec.models import PaperUpdateKind


def fixture_payload() -> bytes:
    return Path("fixtures/arxiv_feed.xml").read_bytes()


def make_client(
    tmp_path: Path,
    handler: httpx.MockTransport,
    *,
    sleeps: list[float] | None = None,
    cache_ttl: timedelta = timedelta(hours=24),
    min_request_interval_seconds: float = 0,
    now=lambda: datetime(2026, 8, 20, 12, tzinfo=UTC),
) -> ArxivClient:
    return ArxivClient(
        api_url="https://export.arxiv.org/api/query",
        user_agent="ArxivRec/tests",
        cache_dir=tmp_path / "cache",
        client=httpx.Client(transport=handler),
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
        min_request_interval_seconds=min_request_interval_seconds,
        retry_backoff_seconds=1,
        cache_ttl=cache_ttl,
        now=now,
    )


def test_parse_atom_feed_validates_and_normalizes_papers() -> None:
    feed = parse_atom_feed(fixture_payload())

    assert feed.total_results == 2
    assert len(feed.papers) == 2
    assert feed.papers[0].arxiv_id == "2608.10001"
    assert feed.papers[0].version == 2
    assert feed.papers[0].categories == ("quant-ph", "physics.optics")


def test_parse_atom_feed_rejects_invalid_xml() -> None:
    with pytest.raises(ArxivParseError, match="valid XML"):
        parse_atom_feed(b"<not-closed>")


def test_build_category_query_deduplicates_without_losing_order() -> None:
    assert build_category_query(["quant-ph", "physics.optics", "quant-ph"]) == (
        "cat:quant-ph OR cat:physics.optics"
    )


def test_fetch_filters_cutoff_and_sends_expected_query(tmp_path: Path) -> None:
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, content=fixture_payload(), request=request)

    client = make_client(tmp_path, httpx.MockTransport(handler))
    papers = client.fetch(
        categories=["quant-ph", "physics.optics"],
        published_after=datetime(2026, 8, 17, tzinfo=UTC),
        max_results=100,
    )

    assert [paper.arxiv_id for paper in papers] == ["2608.10001"]
    params = seen_requests[0].url.params
    assert params["search_query"] == "cat:quant-ph OR cat:physics.optics"
    assert params["sortBy"] == "lastUpdatedDate"
    assert params["max_results"] == "50"
    assert len(seen_requests) == 1
    assert seen_requests[0].headers["User-Agent"] == "ArxivRec/tests"


def test_fetch_report_distinguishes_new_submissions_and_revisions(tmp_path: Path) -> None:
    client = make_client(
        tmp_path,
        httpx.MockTransport(
            lambda request: httpx.Response(200, content=fixture_payload(), request=request)
        ),
    )

    new_report = client.fetch_report(
        categories=["quant-ph"],
        published_after=datetime(2026, 8, 17, tzinfo=UTC),
        max_results=100,
    )
    assert new_report.items[0].update_kind is PaperUpdateKind.NEW_SUBMISSION
    assert new_report.new_submission_count == 1
    assert new_report.revised_version_count == 0
    assert new_report.complete_through_cutoff
    assert not new_report.truncated
    assert new_report.api_total_results == 2

    revised_payload = fixture_payload().replace(
        b"<published>2026-08-18T12:00:00Z</published>",
        b"<published>2026-08-01T12:00:00Z</published>",
    )
    revised_client = make_client(
        tmp_path / "revised",
        httpx.MockTransport(
            lambda request: httpx.Response(200, content=revised_payload, request=request)
        ),
    )
    revised_report = revised_client.fetch_report(
        categories=["quant-ph"],
        published_after=datetime(2026, 8, 17, tzinfo=UTC),
        max_results=100,
    )
    assert revised_report.items[0].update_kind is PaperUpdateKind.REVISED_VERSION
    assert revised_report.new_submission_count == 0
    assert revised_report.revised_version_count == 1


def test_fetch_report_marks_safety_cap_truncation(tmp_path: Path) -> None:
    payload = fixture_payload().replace(
        b"<opensearch:totalResults>2</opensearch:totalResults>",
        b"<opensearch:totalResults>10</opensearch:totalResults>",
    ).replace(b"2026-08-10T09:00:00Z", b"2026-08-18T09:00:00Z")
    client = make_client(
        tmp_path,
        httpx.MockTransport(
            lambda request: httpx.Response(200, content=payload, request=request)
        ),
    )

    report = client.fetch_report(
        categories=["quant-ph"],
        published_after=datetime(2026, 8, 17, tzinfo=UTC),
        max_results=2,
    )

    assert report.scanned_count == 2
    assert report.api_total_results == 10
    assert report.truncated
    assert not report.complete_through_cutoff


def test_fetch_retries_transient_server_error(tmp_path: Path) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        status = 503 if attempts == 1 else 200
        content = b"temporarily unavailable" if status == 503 else fixture_payload()
        return httpx.Response(status, content=content, request=request)

    client = make_client(tmp_path, httpx.MockTransport(handler), sleeps=sleeps)
    papers = client.fetch(
        categories=["quant-ph"],
        published_after=datetime(2026, 8, 17, tzinfo=UTC),
        max_results=10,
    )

    assert len(papers) == 1
    assert attempts == 2
    assert sleeps == [1]


def test_fetch_uses_fresh_cache(tmp_path: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, content=fixture_payload(), request=request)

    client = make_client(tmp_path, httpx.MockTransport(handler))
    arguments = {
        "categories": ["quant-ph"],
        "published_after": datetime(2026, 8, 17, tzinfo=UTC),
        "max_results": 10,
    }

    first = client.fetch(**arguments)
    second = client.fetch(**arguments)

    assert first == second
    assert attempts == 1


def test_request_interval_is_shared_across_client_instances(tmp_path: Path) -> None:
    sleeps: list[float] = []
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=fixture_payload(), request=request)

    transport = httpx.MockTransport(handler)
    first = make_client(
        tmp_path,
        transport,
        sleeps=sleeps,
        min_request_interval_seconds=3.1,
    )
    second = make_client(
        tmp_path,
        transport,
        sleeps=sleeps,
        min_request_interval_seconds=3.1,
    )

    first.fetch(
        categories=["quant-ph"],
        published_after=datetime(2026, 8, 17, tzinfo=UTC),
        max_results=10,
    )
    second.fetch(
        categories=["physics.optics"],
        published_after=datetime(2026, 8, 17, tzinfo=UTC),
        max_results=10,
    )

    assert len(requests) == 2
    assert sleeps == pytest.approx([3.1])


def test_rate_limit_response_persists_retry_after_cooldown(tmp_path: Path) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "10"},
                content=b"rate limited",
                request=request,
            )
        return httpx.Response(200, content=fixture_payload(), request=request)

    first_client = ArxivClient(
        api_url="https://export.arxiv.org/api/query",
        user_agent="ArxivRec/tests",
        cache_dir=tmp_path / "cache",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
        now=lambda: datetime(2026, 8, 20, 12, tzinfo=UTC),
        min_request_interval_seconds=3.1,
        retry_backoff_seconds=1,
        max_retries=1,
    )

    with pytest.raises(ArxivClientError, match="rate-limit"):
        first_client.fetch(
            categories=["quant-ph"],
            published_after=datetime(2026, 8, 17, tzinfo=UTC),
            max_results=10,
        )

    second_client = ArxivClient(
        api_url="https://export.arxiv.org/api/query",
        user_agent="ArxivRec/tests",
        cache_dir=tmp_path / "cache",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
        now=lambda: datetime(2026, 8, 20, 12, tzinfo=UTC),
        min_request_interval_seconds=3.1,
        max_retries=0,
    )
    papers = second_client.fetch(
        categories=["quant-ph"],
        published_after=datetime(2026, 8, 17, tzinfo=UTC),
        max_results=10,
    )

    assert len(papers) == 1
    assert attempts == 2
    assert sleeps == pytest.approx([10])


def test_fetch_requires_timezone_aware_cutoff(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=fixture_payload(), request=request)
    )
    client = make_client(tmp_path, transport)

    with pytest.raises(ValueError, match="timezone"):
        client.fetch(
            categories=["quant-ph"],
            published_after=datetime(2026, 8, 17),
            max_results=10,
        )
