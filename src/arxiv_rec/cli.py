"""Small CLI shell; pipeline commands are added at their implementation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from arxiv_rec.arxiv_client import ArxivClient
from arxiv_rec.digest import build_digest, write_digest_preview
from arxiv_rec.email_delivery import EmailDeliveryReceipt, send_digest_via_resend
from arxiv_rec.evaluation import EvaluationReport, GoldSet, ScorePrediction, evaluate_all
from arxiv_rec.models import Paper, RankingReport, RecallReport, ResearchProfile, RunMode
from arxiv_rec.pipeline import (
    RecommendationRun,
    fetch_live_report,
    run_fixture,
    run_recommendation,
)
from arxiv_rec.profile_analyzer import OpenAIProfileAnalyzer
from arxiv_rec.ranking import (
    FixtureRelevanceRanker,
    OpenAIRelevanceRanker,
    assessments_from_fixture,
)
from arxiv_rec.retrieval import HybridRetriever, OpenAITextEmbedder, StaticTextEmbedder
from arxiv_rec.settings import AppSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arxiv-rec")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check local configuration without making API calls")
    inspect_parser = subparsers.add_parser(
        "inspect-recall", help="Inspect hybrid recall signals using a deterministic fixture"
    )
    inspect_parser.add_argument("--fixture", type=Path, required=True)
    inspect_parser.add_argument("--threshold", type=float, default=0.15)
    inspect_parser.add_argument("--max-candidates", type=int, default=100)
    inspect_parser.add_argument("--lexical-only", action="store_true")
    inspect_parser.add_argument("--json", action="store_true")
    ranking_parser = subparsers.add_parser(
        "inspect-ranking", help="Inspect fixture relevance ranking and selection"
    )
    ranking_parser.add_argument("--fixture", type=Path, required=True)
    ranking_parser.add_argument("--threshold", type=int, default=75)
    ranking_parser.add_argument("--review-margin", type=int, default=5)
    ranking_parser.add_argument("--json", action="store_true")
    evaluation_parser = subparsers.add_parser(
        "evaluate-fixture", help="Compare retrieval and ranking signals against a gold set"
    )
    evaluation_parser.add_argument("--fixture", type=Path, required=True)
    evaluation_parser.add_argument("--gold", type=Path, required=True)
    evaluation_parser.add_argument("--threshold", type=int, default=75)
    evaluation_parser.add_argument("--json", action="store_true")
    live_parser = subparsers.add_parser(
        "run-live", help="Run the complete arXiv + OpenAI pipeline immediately"
    )
    description_group = live_parser.add_mutually_exclusive_group(required=True)
    description_group.add_argument("--description")
    description_group.add_argument("--description-file", type=Path)
    live_parser.add_argument("--lookback-days", type=int, default=1)
    live_parser.add_argument("--max-results", type=int, default=25)
    live_parser.add_argument("--candidate-threshold", type=float, default=0.20)
    live_parser.add_argument("--max-candidates", type=int, default=5)
    live_parser.add_argument("--relevance-threshold", type=int, default=75)
    live_parser.add_argument("--json", action="store_true")
    preview_parser = subparsers.add_parser(
        "preview-digest",
        help="Generate a local HTML email preview from a deterministic fixture",
    )
    preview_parser.add_argument("--fixture", type=Path, required=True)
    preview_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/digest-preview.html"),
    )
    preview_parser.add_argument("--relevance-threshold", type=int, default=40)
    preview_parser.add_argument("--delivery-timezone", default="America/Los_Angeles")
    analyze_parser = subparsers.add_parser(
        "analyze-profile",
        help="Analyze a research description once and save the structured profile",
    )
    analyze_description = analyze_parser.add_mutually_exclusive_group(required=True)
    analyze_description.add_argument("--description")
    analyze_description.add_argument("--description-file", type=Path)
    analyze_parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/research_profile.json"),
    )
    verification_parser = subparsers.add_parser(
        "send-verification-email",
        help="Send the deterministic fixture digest now to verify Resend and email layout",
    )
    verification_parser.add_argument("--fixture", type=Path, required=True)
    verification_parser.add_argument("--relevance-threshold", type=int, default=40)
    verification_parser.add_argument(
        "--delivery-timezone", default="America/Los_Angeles"
    )
    daily_parser = subparsers.add_parser(
        "send-daily-email",
        help="Run live arXiv retrieval and OpenAI ranking, then send the daily digest",
    )
    daily_parser.add_argument("--profile-file", type=Path)
    daily_parser.add_argument("--lookback-days", type=int, default=1)
    daily_parser.add_argument("--max-results", type=int, default=100)
    daily_parser.add_argument("--candidate-threshold", type=float, default=0.20)
    daily_parser.add_argument("--max-candidates", type=int, default=20)
    daily_parser.add_argument("--relevance-threshold", type=int, default=40)
    daily_parser.add_argument("--delivery-timezone", default="America/Los_Angeles")
    daily_parser.add_argument(
        "--allow-duplicate-email",
        action="store_true",
        help="Skip the daily idempotency key for an intentional manual verification send",
    )
    return parser


def _load_recall_fixture(
    path: Path,
) -> tuple[ResearchProfile, list[Paper], tuple[tuple[float, ...], ...] | None]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("fixture root must be a JSON object")
    payload = cast(dict[str, Any], raw)
    profile = ResearchProfile.model_validate(payload.get("profile"))
    papers_raw = payload.get("papers")
    if not isinstance(papers_raw, list):
        raise ValueError("fixture papers must be a JSON list")
    papers = [Paper.model_validate(item) for item in papers_raw]
    vectors_raw = payload.get("embedding_vectors")
    if vectors_raw is None:
        return profile, papers, None
    if not isinstance(vectors_raw, list):
        raise ValueError("fixture embedding_vectors must be a JSON list")
    vectors = tuple(tuple(float(value) for value in vector) for vector in vectors_raw)
    return profile, papers, vectors


def inspect_recall_fixture(
    path: Path,
    *,
    threshold: float,
    max_candidates: int,
    lexical_only: bool,
) -> RecallReport:
    profile, papers, vectors = _load_recall_fixture(path)
    if not lexical_only and vectors is None:
        raise ValueError("fixture has no embedding_vectors; use --lexical-only")
    embedder = None if lexical_only else StaticTextEmbedder(vectors or (), fixture_name=path.stem)
    return HybridRetriever(embedder).retrieve(
        profile,
        papers,
        candidate_threshold=threshold,
        max_candidates=max_candidates,
    )


def _print_recall_table(report: RecallReport) -> None:
    header = (
        f"{'IN':<3} {'ARXIV ID':<13} {'HYBRID':>7} {'KEYWORD':>8} {'BM25':>6} "
        f"{'EMBED':>7} {'INTER':>6} {'NEG':>6}  TITLE"
    )
    print(header)
    print("-" * len(header))
    for item in report.items:
        signals = item.signals
        embedding = "-" if signals.embedding_score is None else f"{signals.embedding_score:.3f}"
        print(
            f"{('yes' if item.included else 'no'):<3} {item.paper.arxiv_id:<13} "
            f"{signals.hybrid_score:>7.3f} {signals.keyword_score:>8.3f} "
            f"{signals.bm25_score:>6.3f} {embedding:>7} "
            f"{signals.intersection_coverage:>6.3f} {signals.negative_penalty:>6.3f}  "
            f"{item.paper.title}"
        )


def inspect_ranking_fixture(
    path: Path,
    *,
    threshold: int,
    review_margin: int,
) -> RankingReport:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("fixture root must be a JSON object")
    payload = cast(dict[str, Any], raw)
    profile, papers, vectors = _load_recall_fixture(path)
    if vectors is None:
        raise ValueError("fixture has no embedding_vectors")
    assessments_raw = payload.get("llm_assessments")
    if not isinstance(assessments_raw, dict):
        raise ValueError("fixture has no llm_assessments object")
    recall = HybridRetriever(StaticTextEmbedder(vectors, fixture_name=path.stem)).retrieve(
        profile,
        papers,
        candidate_threshold=0.0,
        max_candidates=len(papers),
    )
    ranker = FixtureRelevanceRanker(
        assessments_from_fixture(cast(dict[str, Any], assessments_raw)),
        fixture_name=path.stem,
    )
    return ranker.assess(
        profile,
        recall.candidates,
        relevance_threshold=threshold,
        review_margin=review_margin,
    )


def _print_ranking_table(report: RankingReport) -> None:
    header = f"{'PUSH':<5} {'CHECK':<5} {'ARXIV ID':<13} {'SCORE':>5} {'CONF':<6}  TITLE"
    print(header)
    print("-" * len(header))
    for item in report.items:
        print(
            f"{('yes' if item.selected else 'no'):<5} "
            f"{('yes' if item.needs_review else 'no'):<5} "
            f"{item.paper.arxiv_id:<13} {item.assessment.relevance_score:>5} "
            f"{item.assessment.confidence.value:<6}  {item.paper.title}"
        )
        assessment = item.assessment
        if (
            assessment.raw_relevance_score is not None
            and assessment.raw_relevance_score != assessment.relevance_score
            and assessment.profile_tier is not None
        ):
            print(
                f"      Calibration: raw {assessment.raw_relevance_score} -> "
                f"{assessment.relevance_score} ({assessment.profile_tier.value})"
            )
        print(f"      Why: {item.assessment.reason}")


def evaluate_fixture(path: Path, gold_path: Path, *, threshold: int) -> EvaluationReport:
    profile, papers, vectors = _load_recall_fixture(path)
    if vectors is None:
        raise ValueError("fixture has no embedding_vectors")
    recall = HybridRetriever(StaticTextEmbedder(vectors, fixture_name=path.stem)).retrieve(
        profile,
        papers,
        candidate_threshold=0.0,
        max_candidates=len(papers),
    )
    ranking = inspect_ranking_fixture(path, threshold=threshold, review_margin=5)
    gold_set = GoldSet.model_validate_json(gold_path.read_text(encoding="utf-8"))

    def predictions(signal: str) -> list[ScorePrediction]:
        values: list[ScorePrediction] = []
        for item in recall.items:
            raw = getattr(item.signals, signal)
            score = max(0.0, float(raw or 0.0)) * 100
            values.append(ScorePrediction(arxiv_id=item.paper.arxiv_id, score=score))
        return values

    systems = {
        "keyword": predictions("keyword_score"),
        "bm25": predictions("bm25_score"),
        "embedding": predictions("embedding_score"),
        "hybrid": predictions("hybrid_score"),
        "llm": [
            ScorePrediction(
                arxiv_id=item.paper.arxiv_id,
                score=float(item.assessment.relevance_score),
            )
            for item in ranking.items
        ],
    }
    return evaluate_all(gold_set=gold_set, systems=systems, threshold=threshold)


def _print_evaluation_table(report: EvaluationReport) -> None:
    header = (
        f"{'SYSTEM':<10} {'N':>3} {'MISS':>4} {'PREC':>6} {'RECALL':>6} "
        f"{'F1':>6} {'FPR':>6} {'MAE':>6} {'NDCG':>6}"
    )
    print(header)
    print("-" * len(header))
    for metric in report.metrics:
        print(
            f"{metric.system:<10} {metric.evaluated_count:>3} {metric.missing_count:>4} "
            f"{metric.precision:>6.3f} {metric.recall:>6.3f} {metric.f1:>6.3f} "
            f"{metric.false_positive_rate:>6.3f} {metric.mean_absolute_error:>6.2f} "
            f"{metric.ndcg:>6.3f}"
        )


def run_live(
    *,
    description: str,
    settings: AppSettings,
    lookback_days: int,
    max_results: int,
    candidate_threshold: float,
    max_candidates: int,
    relevance_threshold: int,
) -> RecommendationRun:
    if not settings.live_openai_available or settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not configured in the local environment")
    api_key = settings.openai_api_key.get_secret_value()
    profile = OpenAIProfileAnalyzer(
        model=settings.profile_model, api_key=api_key
    ).analyze(description).profile
    return run_live_with_profile(
        profile=profile,
        settings=settings,
        lookback_days=lookback_days,
        max_results=max_results,
        candidate_threshold=candidate_threshold,
        max_candidates=max_candidates,
        relevance_threshold=relevance_threshold,
    )


def run_live_with_profile(
    *,
    profile: ResearchProfile,
    settings: AppSettings,
    lookback_days: int,
    max_results: int,
    candidate_threshold: float,
    max_candidates: int,
    relevance_threshold: int,
) -> RecommendationRun:
    """Run daily retrieval/ranking with an already approved structured profile."""
    if not settings.live_openai_available or settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not configured in the local environment")
    api_key = settings.openai_api_key.get_secret_value()
    with ArxivClient(
        api_url=settings.arxiv_api_url,
        user_agent=settings.arxiv_user_agent,
        cache_dir=settings.cache_dir,
        timeout_seconds=settings.request_timeout_seconds,
    ) as client:
        fetch_report = fetch_live_report(
            profile=profile,
            client=client,
            lookback_days=lookback_days,
            max_results=max_results,
        )
    return run_recommendation(
        mode=RunMode.LIVE,
        profile=profile,
        papers=fetch_report.papers,
        retriever=HybridRetriever(
            OpenAITextEmbedder(model=settings.embedding_model, api_key=api_key)
        ),
        ranker=OpenAIRelevanceRanker(
            model=settings.ranking_model,
            cache_dir=settings.cache_dir / "ranking",
            api_key=api_key,
        ),
        candidate_threshold=candidate_threshold,
        max_candidates=max_candidates,
        relevance_threshold=relevance_threshold,
        fetch_report=fetch_report,
    )


def analyze_profile_to_file(
    *, description: str, output_path: Path, settings: AppSettings
) -> ResearchProfile:
    """Analyze a profile once and persist validated JSON for repeatable daily runs."""
    if not settings.live_openai_available or settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not configured in the local environment")
    profile = OpenAIProfileAnalyzer(
        model=settings.profile_model,
        api_key=settings.openai_api_key.get_secret_value(),
    ).analyze(description).profile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return profile


def _load_delivery_profile(
    profile_file: Path | None, settings: AppSettings
) -> ResearchProfile:
    if profile_file is not None:
        return ResearchProfile.model_validate_json(profile_file.read_text(encoding="utf-8"))
    if settings.research_profile_json is not None:
        return ResearchProfile.model_validate_json(
            settings.research_profile_json.get_secret_value()
        )
    raise ValueError(
        "A profile is required. Use --profile-file locally or configure "
        "ARXIVREC_PROFILE_JSON for a scheduled run."
    )


def _send_digest(
    report: RecommendationRun,
    *,
    settings: AppSettings,
    delivery_timezone: str,
    idempotency_key: str | None = None,
) -> EmailDeliveryReceipt:
    if not settings.email_delivery_available or settings.resend_api_key is None:
        raise ValueError(
            "Email delivery needs RESEND_API_KEY and ARXIVREC_RECIPIENT in the environment."
        )
    assert settings.recipient_email is not None
    content = build_digest(report, delivery_timezone=delivery_timezone)
    return send_digest_via_resend(
        content,
        api_key=settings.resend_api_key.get_secret_value(),
        recipient=settings.recipient_email,
        sender=settings.email_from,
        idempotency_key=idempotency_key,
        timeout_seconds=settings.request_timeout_seconds,
    )


def _daily_idempotency_key(
    report: RecommendationRun,
    *,
    profile: ResearchProfile,
    delivery_timezone: str,
) -> str:
    local_day = report.completed_at.astimezone(ZoneInfo(delivery_timezone)).date().isoformat()
    material = f"{local_day}\n{profile.model_dump_json()}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"arxivrec-daily-{local_day}-{digest}"


def _print_live_report(report: RecommendationRun) -> None:
    print(
        f"Fetched {report.fetched_count}; recalled {len(report.recall.candidates)}; "
        f"recommended {len(report.ranking.selected)}"
    )
    if report.fetch_report is not None:
        fetch = report.fetch_report
        status = "TRUNCATED" if fetch.truncated else "complete"
        print(
            f"Coverage: {status}; {fetch.new_submission_count} new, "
            f"{fetch.revised_version_count} revised; scanned {fetch.scanned_count} "
            f"of {fetch.api_total_results} all-time category records"
        )
    _print_ranking_table(report.ranking)


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        settings = AppSettings()
        print("ArxivRec local configuration")
        print(f"OpenAI key configured: {settings.live_openai_available}")
        print(f"Profile model: {settings.profile_model}")
        print(f"Ranking model: {settings.ranking_model}")
        print(f"Embedding model: {settings.embedding_model}")
        print(f"Email delivery configured: {settings.email_delivery_available}")
        return 0
    if args.command == "inspect-recall":
        report = inspect_recall_fixture(
            args.fixture,
            threshold=args.threshold,
            max_candidates=args.max_candidates,
            lexical_only=args.lexical_only,
        )
        if args.json:
            print(report.model_dump_json(indent=2))
        else:
            _print_recall_table(report)
        return 0
    if args.command == "inspect-ranking":
        ranking_report = inspect_ranking_fixture(
            args.fixture,
            threshold=args.threshold,
            review_margin=args.review_margin,
        )
        if args.json:
            print(ranking_report.model_dump_json(indent=2))
        else:
            _print_ranking_table(ranking_report)
        return 0
    if args.command == "evaluate-fixture":
        evaluation_report = evaluate_fixture(
            args.fixture,
            args.gold,
            threshold=args.threshold,
        )
        if args.json:
            print(evaluation_report.model_dump_json(indent=2))
        else:
            _print_evaluation_table(evaluation_report)
        return 0
    if args.command == "run-live":
        description = (
            args.description_file.read_text(encoding="utf-8")
            if args.description_file is not None
            else args.description
        )
        live_report = run_live(
            description=description,
            settings=AppSettings(),
            lookback_days=args.lookback_days,
            max_results=args.max_results,
            candidate_threshold=args.candidate_threshold,
            max_candidates=args.max_candidates,
            relevance_threshold=args.relevance_threshold,
        )
        if args.json:
            print(live_report.model_dump_json(indent=2))
        else:
            _print_live_report(live_report)
        return 0
    if args.command == "preview-digest":
        preview_run = run_fixture(
            path=args.fixture,
            candidate_threshold=0.0,
            relevance_threshold=args.relevance_threshold,
        )
        digest = write_digest_preview(
            preview_run,
            args.output,
            delivery_timezone=args.delivery_timezone,
        )
        print(digest.subject)
        print(f"HTML preview: {args.output.resolve()}")
        return 0
    if args.command == "analyze-profile":
        description = (
            args.description_file.read_text(encoding="utf-8")
            if args.description_file is not None
            else args.description
        )
        analyze_profile_to_file(
            description=description,
            output_path=args.output,
            settings=AppSettings(),
        )
        print(f"Structured profile: {args.output.resolve()}")
        return 0
    if args.command == "send-verification-email":
        settings = AppSettings()
        verification_run = run_fixture(
            path=args.fixture,
            candidate_threshold=0.0,
            relevance_threshold=args.relevance_threshold,
        )
        receipt = _send_digest(
            verification_run,
            settings=settings,
            delivery_timezone=args.delivery_timezone,
        )
        print(f"Verification email accepted by {receipt.provider}: {receipt.message_id}")
        return 0
    if args.command == "send-daily-email":
        settings = AppSettings()
        profile = _load_delivery_profile(args.profile_file, settings)
        daily_run = run_live_with_profile(
            profile=profile,
            settings=settings,
            lookback_days=args.lookback_days,
            max_results=args.max_results,
            candidate_threshold=args.candidate_threshold,
            max_candidates=args.max_candidates,
            relevance_threshold=args.relevance_threshold,
        )
        receipt = _send_digest(
            daily_run,
            settings=settings,
            delivery_timezone=args.delivery_timezone,
            idempotency_key=(
                None
                if args.allow_duplicate_email
                else _daily_idempotency_key(
                    daily_run,
                    profile=profile,
                    delivery_timezone=args.delivery_timezone,
                )
            ),
        )
        _print_live_report(daily_run)
        print(f"Daily email accepted by {receipt.provider}: {receipt.message_id}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
