"""Small, testable client for the official arXiv Atom API."""

from __future__ import annotations

import hashlib
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import BinaryIO, Final
from urllib.parse import urlencode

import httpx

from arxiv_rec.models import ArxivFetchReport, FetchedPaper, Paper, PaperUpdateKind

ATOM: Final = "http://www.w3.org/2005/Atom"
ARXIV: Final = "http://arxiv.org/schemas/atom"
OPENSEARCH: Final = "http://a9.com/-/spec/opensearch/1.1/"
NS: Final = {"atom": ATOM, "arxiv": ARXIV, "opensearch": OPENSEARCH}


class ArxivClientError(RuntimeError):
    """Base exception for arXiv retrieval failures."""


class ArxivParseError(ArxivClientError):
    """Raised when an Atom response cannot be converted into validated papers."""


class ArxivRateLimitError(ArxivClientError):
    """Raised after arXiv requests a longer cooldown."""

    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__("arXiv returned HTTP 429")
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class ParsedFeed:
    papers: tuple[Paper, ...]
    total_results: int


def _required_text(element: ET.Element, path: str) -> str:
    child = element.find(path, NS)
    if child is None or not child.text or not child.text.strip():
        raise ArxivParseError(f"missing required Atom field: {path}")
    return child.text.strip()


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArxivParseError(f"invalid Atom timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ArxivParseError(f"Atom timestamp lacks timezone: {value!r}")
    return parsed.astimezone(UTC)


def parse_atom_feed(payload: bytes | str) -> ParsedFeed:
    """Parse and validate a complete arXiv Atom response."""
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ArxivParseError("response is not valid XML") from exc

    total_node = root.find("opensearch:totalResults", NS)
    try:
        total_results = int(total_node.text) if total_node is not None and total_node.text else 0
    except ValueError as exc:
        raise ArxivParseError("opensearch:totalResults is not an integer") from exc

    papers: list[Paper] = []
    for entry in root.findall("atom:entry", NS):
        authors = [
            name.text.strip()
            for author in entry.findall("atom:author", NS)
            if (name := author.find("atom:name", NS)) is not None and name.text
        ]
        categories = [
            category.attrib["term"].strip()
            for category in entry.findall("atom:category", NS)
            if category.attrib.get("term", "").strip()
        ]
        primary_node = entry.find("arxiv:primary_category", NS)
        primary_category = (
            primary_node.attrib.get("term", "").strip() if primary_node is not None else ""
        )
        links = entry.findall("atom:link", NS)
        abs_url = next(
            (
                link.attrib.get("href")
                for link in links
                if link.attrib.get("rel") == "alternate"
            ),
            None,
        )
        pdf_url = next(
            (
                link.attrib.get("href")
                for link in links
                if link.attrib.get("title") == "pdf"
                or link.attrib.get("type") == "application/pdf"
            ),
            None,
        )

        try:
            paper = Paper(
                arxiv_id=_required_text(entry, "atom:id"),
                title=_required_text(entry, "atom:title"),
                abstract=_required_text(entry, "atom:summary"),
                authors=tuple(authors),
                categories=tuple(categories),
                primary_category=primary_category,
                published_at=_parse_datetime(_required_text(entry, "atom:published")),
                updated_at=_parse_datetime(_required_text(entry, "atom:updated")),
                abs_url=abs_url,
                pdf_url=pdf_url,
            )
        except ValueError as exc:
            entry_id = entry.findtext(f"{{{ATOM}}}id", default="unknown")
            raise ArxivParseError(f"invalid paper entry {entry_id!r}: {exc}") from exc
        papers.append(paper)

    return ParsedFeed(papers=tuple(papers), total_results=total_results)


def build_category_query(categories: Sequence[str]) -> str:
    cleaned = tuple(dict.fromkeys(category.strip() for category in categories if category.strip()))
    if not cleaned:
        raise ValueError("at least one arXiv category is required")
    return " OR ".join(f"cat:{category}" for category in cleaned)


@contextmanager
def _exclusive_lock(handle: BinaryIO) -> Iterator[None]:
    """Hold an OS-level lock on the first byte of a shared state file."""
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


class PersistentRequestGate:
    """Serialize requests and preserve the minimum interval across app runs."""

    def __init__(
        self,
        *,
        state_path: Path,
        interval_seconds: float,
        sleep: Callable[[float], None],
        now: Callable[[], datetime],
    ) -> None:
        self.state_path = state_path
        self.interval_seconds = interval_seconds
        self._sleep = sleep
        self._now = now

    @contextmanager
    def request_slot(self) -> Iterator[None]:
        if self.interval_seconds <= 0:
            yield
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("a+b") as handle, _exclusive_lock(handle):
            next_allowed = self._read_timestamp(handle)
            current = self._now().timestamp()
            wait_seconds = max(0.0, next_allowed - current)
            if wait_seconds > 0:
                self._sleep(wait_seconds)
            dispatch_time = max(self._now().timestamp(), next_allowed)
            self._write_timestamp(handle, dispatch_time + self.interval_seconds)
            # Holding the lock through the request also enforces one connection at a time.
            yield

    def defer(self, seconds: float) -> None:
        if self.interval_seconds <= 0:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("a+b") as handle, _exclusive_lock(handle):
            existing = self._read_timestamp(handle)
            deferred = self._now().timestamp() + max(0.0, seconds)
            self._write_timestamp(handle, max(existing, deferred))

    @staticmethod
    def _read_timestamp(handle: BinaryIO) -> float:
        handle.seek(0)
        try:
            return float(handle.read().decode("ascii").strip() or "0")
        except (UnicodeDecodeError, ValueError):
            return 0.0

    @staticmethod
    def _write_timestamp(handle: BinaryIO, value: float) -> None:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{value:.6f}".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())


def _retry_after_seconds(response: httpx.Response, now: datetime) -> float:
    value = response.headers.get("Retry-After", "").strip()
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is not None:
                    return max(0.0, (retry_at.astimezone(UTC) - now).total_seconds())
            except (TypeError, ValueError):
                pass
    # A conservative shared cooldown when arXiv does not provide Retry-After.
    return 60.0


class ArxivClient:
    """Retrieve recent papers with pagination, retries, caching, and deduplication."""

    def __init__(
        self,
        *,
        api_url: str,
        user_agent: str,
        cache_dir: Path,
        timeout_seconds: float = 30.0,
        page_size: int = 50,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        min_request_interval_seconds: float = 3.1,
        cache_ttl: timedelta = timedelta(hours=24),
        rate_limit_state_path: Path | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not 1 <= page_size <= 2000:
            raise ValueError("page_size must be between 1 and 2000")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.api_url = api_url
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.page_size = page_size
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self.cache_ttl = cache_ttl
        self._client = client or httpx.Client(timeout=timeout_seconds, follow_redirects=True)
        self._owns_client = client is None
        self._sleep = sleep
        self._now = now
        self._request_gate = PersistentRequestGate(
            state_path=rate_limit_state_path or cache_dir.parent / ".arxiv_api_rate_limit",
            interval_seconds=min_request_interval_seconds,
            sleep=sleep,
            now=now,
        )

    def __enter__(self) -> ArxivClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch(
        self,
        *,
        categories: Sequence[str],
        published_after: datetime,
        max_results: int,
    ) -> list[Paper]:
        return list(
            self.fetch_report(
                categories=categories,
                published_after=published_after,
                max_results=max_results,
            ).papers
        )

    def fetch_report(
        self,
        *,
        categories: Sequence[str],
        published_after: datetime,
        max_results: int,
    ) -> ArxivFetchReport:
        if published_after.tzinfo is None or published_after.utcoffset() is None:
            raise ValueError("published_after must include a timezone")
        if max_results < 1:
            raise ValueError("max_results must be positive")

        cutoff = published_after.astimezone(UTC)
        cleaned_categories = tuple(
            dict.fromkeys(category.strip() for category in categories if category.strip())
        )
        query = build_category_query(cleaned_categories)
        start = 0
        deduplicated: dict[str, FetchedPaper] = {}
        api_total_results = 0
        complete_through_cutoff = False

        while start < max_results:
            requested = min(self.page_size, max_results - start)
            params = {
                "search_query": query,
                "start": str(start),
                "max_results": str(requested),
                "sortBy": "lastUpdatedDate",
                "sortOrder": "descending",
            }
            feed = parse_atom_feed(self._get_page(params))
            api_total_results = feed.total_results
            if not feed.papers:
                complete_through_cutoff = True
                break

            reached_cutoff = False
            for paper in feed.papers:
                if paper.updated_at < cutoff:
                    reached_cutoff = True
                    continue
                update_kind = (
                    PaperUpdateKind.NEW_SUBMISSION
                    if paper.published_at >= cutoff
                    else PaperUpdateKind.REVISED_VERSION
                )
                existing = deduplicated.get(paper.arxiv_id)
                if existing is None or paper.updated_at > existing.paper.updated_at:
                    deduplicated[paper.arxiv_id] = FetchedPaper(
                        paper=paper, update_kind=update_kind
                    )

            start += len(feed.papers)
            if reached_cutoff or start >= feed.total_results or len(feed.papers) < requested:
                complete_through_cutoff = True
                break

        ordered = tuple(
            sorted(
                deduplicated.values(),
                key=lambda item: item.paper.updated_at,
                reverse=True,
            )
        )
        truncated = not complete_through_cutoff and start >= max_results
        return ArxivFetchReport(
            items=ordered,
            categories=cleaned_categories,
            cutoff=cutoff,
            api_total_results=api_total_results,
            scanned_count=start,
            page_size=self.page_size,
            safety_cap=max_results,
            complete_through_cutoff=complete_through_cutoff,
            truncated=truncated,
        )

    def _get_page(self, params: dict[str, str]) -> bytes:
        cache_path = self._cache_path(params)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._request_gate.request_slot():
                    response = self._client.get(
                        self.api_url,
                        params=params,
                        headers={
                            "User-Agent": self.user_agent,
                            "Accept": "application/atom+xml",
                        },
                    )
                if response.status_code == 429:
                    retry_after = _retry_after_seconds(response, self._now())
                    self._request_gate.defer(retry_after)
                    raise ArxivRateLimitError(retry_after)
                if response.status_code >= 500:
                    raise ArxivClientError(f"arXiv returned HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.content
                parse_atom_feed(payload)
                self._write_cache(cache_path, payload)
                return payload
            except (httpx.HTTPError, ArxivClientError, ArxivParseError) as exc:
                last_error = exc
                if isinstance(exc, ArxivRateLimitError):
                    # Do not automatically hit arXiv again after a 429. A later run will
                    # observe the persisted Retry-After cooldown.
                    break
                if attempt >= self.max_retries:
                    break
                self._sleep(self.retry_backoff_seconds * (2**attempt))
        if isinstance(last_error, ArxivRateLimitError):
            raise ArxivClientError(
                "arXiv rate-limited the request. A shared cooldown was saved; retry later."
            ) from last_error
        raise ArxivClientError(
            f"arXiv request failed after {self.max_retries + 1} attempts"
        ) from last_error

    def _cache_path(self, params: dict[str, str]) -> Path:
        key_material = f"{self.api_url}?{urlencode(sorted(params.items()))}"
        digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.atom"

    def _read_cache(self, path: Path) -> bytes | None:
        if not path.exists() or self.cache_ttl <= timedelta(0):
            return None
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if self._now() - modified_at > self.cache_ttl:
            return None
        return path.read_bytes()

    @staticmethod
    def _write_cache(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
