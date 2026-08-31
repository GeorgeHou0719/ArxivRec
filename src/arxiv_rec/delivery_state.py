"""Durable incremental-delivery checkpoints for scheduled digests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from arxiv_rec.models import ArxivFetchReport, FetchedPaper, Paper, ResearchProfile

_SCHEMA_VERSION = 1


def profile_fingerprint(profile: ResearchProfile) -> str:
    """Return a stable identifier so each research profile has its own cursor."""
    material = json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def paper_version_key(paper: Paper) -> str:
    """Identify an arXiv version, falling back to its update timestamp if needed."""
    if paper.version is not None:
        return f"{paper.arxiv_id}v{paper.version}"
    updated = paper.updated_at.astimezone(UTC).isoformat()
    return f"{paper.arxiv_id}@{updated}"


@dataclass(frozen=True)
class DeliveryCheckpoint:
    path: Path
    profile_hash: str
    last_successful_updated_at: datetime | None
    processed_versions: dict[str, datetime]

    def retrieval_cutoff(
        self,
        *,
        now: datetime,
        initial_lookback_days: int,
        overlap: timedelta = timedelta(hours=24),
    ) -> datetime:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must include a timezone")
        if not 1 <= initial_lookback_days <= 30:
            raise ValueError("initial_lookback_days must be between 1 and 30")
        reference = now.astimezone(UTC)
        if self.last_successful_updated_at is None:
            return reference - timedelta(days=initial_lookback_days)
        return self.last_successful_updated_at - overlap

    def filter_unseen(self, report: ArxivFetchReport) -> ArxivFetchReport:
        unseen: tuple[FetchedPaper, ...] = tuple(
            item
            for item in report.items
            if paper_version_key(item.paper) not in self.processed_versions
        )
        return report.model_copy(update={"items": unseen})

    def commit(
        self,
        papers: tuple[Paper, ...],
        *,
        retention: timedelta = timedelta(days=30),
    ) -> DeliveryCheckpoint:
        """Persist papers only after the caller confirms successful email delivery."""
        if not papers:
            return self

        newest = max(paper.updated_at.astimezone(UTC) for paper in papers)
        cursor = max(
            value
            for value in (self.last_successful_updated_at, newest)
            if value is not None
        )
        keep_after = cursor - retention
        processed = {
            key: timestamp
            for key, timestamp in self.processed_versions.items()
            if timestamp >= keep_after
        }
        for paper in papers:
            processed[paper_version_key(paper)] = paper.updated_at.astimezone(UTC)

        updated = DeliveryCheckpoint(
            path=self.path,
            profile_hash=self.profile_hash,
            last_successful_updated_at=cursor,
            processed_versions=processed,
        )
        updated._write()
        return updated

    def _write(self) -> None:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "profile_hash": self.profile_hash,
            "last_successful_updated_at": (
                self.last_successful_updated_at.astimezone(UTC).isoformat()
                if self.last_successful_updated_at is not None
                else None
            ),
            "processed_versions": {
                key: value.astimezone(UTC).isoformat()
                for key, value in sorted(self.processed_versions.items())
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


class DeliveryStateStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load(self, profile: ResearchProfile) -> DeliveryCheckpoint:
        fingerprint = profile_fingerprint(profile)
        path = self.directory / f"{fingerprint}.json"
        if not path.exists():
            return DeliveryCheckpoint(path, fingerprint, None, {})

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"unsupported delivery state schema in {path}")
        if payload.get("profile_hash") != fingerprint:
            raise ValueError(f"delivery state profile mismatch in {path}")

        raw_cursor = payload.get("last_successful_updated_at")
        cursor = _parse_timestamp(raw_cursor, field="last_successful_updated_at")
        raw_versions = payload.get("processed_versions")
        if not isinstance(raw_versions, dict):
            raise ValueError(f"processed_versions must be an object in {path}")
        processed = {
            str(key): _required_timestamp(value, field=f"processed_versions.{key}")
            for key, value in raw_versions.items()
        }
        return DeliveryCheckpoint(path, fingerprint, cursor, processed)


def _parse_timestamp(value: object, *, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _required_timestamp(value: object, *, field: str) -> datetime:
    parsed = _parse_timestamp(value, field=field)
    if parsed is None:
        raise ValueError(f"{field} cannot be null")
    return parsed
