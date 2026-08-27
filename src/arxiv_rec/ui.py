"""English local Streamlit interface for interactive recommendation tests."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import streamlit as st

from arxiv_rec.arxiv_client import ArxivClient
from arxiv_rec.feedback import FeedbackStore, analyze_score_diagnostics
from arxiv_rec.models import ArxivFetchReport, Recommendation, ResearchProfile, RunMode
from arxiv_rec.pipeline import (
    RecommendationRun,
    fetch_live_report,
    load_fixture,
    run_recommendation,
)
from arxiv_rec.profile_analyzer import OpenAIProfileAnalyzer
from arxiv_rec.ranking import OpenAIRelevanceRanker
from arxiv_rec.retrieval import HybridRetriever, OpenAITextEmbedder
from arxiv_rec.settings import AppSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "fixtures" / "cavity_qed_papers.json"
PROFILE_FIELD_KEYS = (
    "core_topics",
    "required_intersections",
    "primary_facets",
    "methods",
    "platforms",
    "adjacent_topics",
    "weakly_related_topics",
    "negative_topics",
    "arxiv_categories",
)


def _lines(values: Iterable[str]) -> str:
    return "\n".join(values)


def _parse_lines(value: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in value.splitlines() if line.strip())


def _parse_intersections(value: str) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for line in value.splitlines():
        concepts = tuple(part.strip() for part in line.split("+") if part.strip())
        if concepts:
            groups.append(concepts)
    return tuple(groups)


def _profile_field_values(profile: ResearchProfile) -> dict[str, str]:
    return {
        "core_topics": _lines(profile.core_topics),
        "required_intersections": _lines(
            " + ".join(group) for group in profile.required_intersections
        ),
        "primary_facets": _lines(profile.primary_facets),
        "methods": _lines(profile.methods),
        "platforms": _lines(profile.platforms),
        "adjacent_topics": _lines(profile.adjacent_topics),
        "weakly_related_topics": _lines(profile.weakly_related_topics),
        "negative_topics": _lines(profile.negative_topics),
        "arxiv_categories": _lines(profile.arxiv_categories),
    }


def _set_profile(profile: ResearchProfile) -> None:
    st.session_state["profile"] = profile
    st.session_state["analyzed_source"] = profile.source_text
    for key, value in _profile_field_values(profile).items():
        st.session_state[f"profile_{key}"] = value


def _edited_profile(source_text: str) -> ResearchProfile:
    base: ResearchProfile = st.session_state["profile"]
    values = base.model_dump()
    values.update(
        {
            "source_text": source_text.strip(),
            "core_topics": _parse_lines(st.session_state["profile_core_topics"]),
            "required_intersections": _parse_intersections(
                st.session_state["profile_required_intersections"]
            ),
            "primary_facets": _parse_lines(st.session_state["profile_primary_facets"]),
            "methods": _parse_lines(st.session_state["profile_methods"]),
            "platforms": _parse_lines(st.session_state["profile_platforms"]),
            "adjacent_topics": _parse_lines(st.session_state["profile_adjacent_topics"]),
            "weakly_related_topics": _parse_lines(
                st.session_state["profile_weakly_related_topics"]
            ),
            "negative_topics": _parse_lines(st.session_state["profile_negative_topics"]),
            "arxiv_categories": _parse_lines(st.session_state["profile_arxiv_categories"]),
        }
    )
    return ResearchProfile.model_validate(values)


def _analyze_profile(*, source_text: str, settings: AppSettings) -> ResearchProfile:
    if not settings.live_openai_available or settings.openai_api_key is None:
        raise ValueError("Live mode needs OPENAI_API_KEY in your local .env file.")
    live_analyzer = OpenAIProfileAnalyzer(
        model=settings.profile_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    return live_analyzer.analyze(source_text).profile


def _local_date_range_utc(
    start_date: date, end_date: date
) -> tuple[datetime, datetime]:
    if end_date < start_date:
        raise ValueError("The end date must be on or after the start date.")
    local_timezone = datetime.now().astimezone().tzinfo
    start_local = datetime.combine(start_date, time.min, tzinfo=local_timezone)
    # The calendar range is inclusive, while timestamp filtering uses an exclusive end.
    end_local = datetime.combine(
        end_date + timedelta(days=1), time.min, tzinfo=local_timezone
    )
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def _filter_report_to_window(
    report: ArxivFetchReport, *, start: datetime, end: datetime
) -> ArxivFetchReport:
    return report.model_copy(
        update={
            "items": tuple(
                item for item in report.items if start <= item.paper.updated_at < end
            )
        }
    )


def _execute_run(
    *,
    profile: ResearchProfile,
    settings: AppSettings,
    lookback_days: int,
    specific_date_range: tuple[date, date] | None,
    max_results: int,
    candidate_threshold: float,
    max_candidates: int,
    relevance_threshold: int,
) -> RecommendationRun:
    if not settings.live_openai_available or settings.openai_api_key is None:
        raise ValueError("Live mode needs OPENAI_API_KEY in your local .env file.")
    api_key = settings.openai_api_key.get_secret_value()
    with ArxivClient(
        api_url=settings.arxiv_api_url,
        user_agent=settings.arxiv_user_agent,
        cache_dir=settings.cache_dir,
        timeout_seconds=settings.request_timeout_seconds,
    ) as client:
        if specific_date_range is None:
            fetch_report = fetch_live_report(
                profile=profile,
                client=client,
                lookback_days=lookback_days,
                max_results=max_results,
            )
        else:
            start, end = _local_date_range_utc(*specific_date_range)
            unfiltered_report = client.fetch_report(
                categories=profile.arxiv_categories,
                published_after=start,
                max_results=max_results,
            )
            fetch_report = _filter_report_to_window(
                unfiltered_report, start=start, end=end
            )
    retriever = HybridRetriever(
        OpenAITextEmbedder(model=settings.embedding_model, api_key=api_key)
    )
    ranker = OpenAIRelevanceRanker(
        model=settings.ranking_model,
        cache_dir=settings.cache_dir / "ranking",
        api_key=api_key,
    )
    return run_recommendation(
        mode=RunMode.LIVE,
        profile=profile,
        papers=fetch_report.papers,
        retriever=retriever,
        ranker=ranker,
        candidate_threshold=candidate_threshold,
        max_candidates=max_candidates,
        relevance_threshold=relevance_threshold,
        fetch_report=fetch_report,
    )


def _render_profile_editor() -> None:
    with st.expander("Review the AI-extracted research profile", expanded=False):
        st.caption(
            "A research profile is the structured summary used to find and score papers: "
            "core topics, important topic combinations, methods, platforms, neighboring "
            "interests, exclusions, and arXiv categories. Edit one item per line. Under "
            "Required intersections, use `+` between concepts that must occur together."
        )
        first, second = st.columns(2)
        with first:
            st.text_area("Core topics", key="profile_core_topics", height=100)
            st.text_area(
                "Required intersections", key="profile_required_intersections", height=90
            )
            st.text_area("Primary facets", key="profile_primary_facets", height=100)
            st.text_area("Methods", key="profile_methods", height=80)
            st.text_area("Platforms", key="profile_platforms", height=80)
        with second:
            st.text_area("Adjacent topics", key="profile_adjacent_topics", height=100)
            st.text_area(
                "Weakly related topics", key="profile_weakly_related_topics", height=100
            )
            st.text_area("Negative topics", key="profile_negative_topics", height=100)
            st.text_area("arXiv categories", key="profile_arxiv_categories", height=80)


def _render_feedback(
    recommendation: Recommendation,
    profile: ResearchProfile,
    store: FeedbackStore,
) -> None:
    paper = recommendation.paper
    assessment = recommendation.assessment
    with st.expander("Correct this score (saved locally)", expanded=False):
        score = st.slider(
            "Your relevance score",
            min_value=0,
            max_value=100,
            value=assessment.relevance_score,
            key=f"feedback_score_{paper.arxiv_id}",
        )
        notes = st.text_input(
            "Optional note", key=f"feedback_notes_{paper.arxiv_id}"
        )
        if st.button("Save feedback", key=f"save_feedback_{paper.arxiv_id}"):
            store.save(
                profile=profile,
                paper=paper,
                predicted_score=assessment.relevance_score,
                user_score=score,
                notes=notes,
            )
            st.success("Feedback saved.")


def _render_recommendation(
    recommendation: Recommendation,
    profile: ResearchProfile,
    store: FeedbackStore,
) -> None:
    assessment = recommendation.assessment
    paper = recommendation.paper
    label = assessment.label.value.replace("_", " ").title()
    badge = "Relevant" if recommendation.selected else "Below threshold"
    st.subheader(f"{assessment.relevance_score} · {label} — {paper.title}")
    st.caption(
        f"{badge} · Confidence: {assessment.confidence.value.title()} · "
        f"arXiv:{paper.arxiv_id} · {paper.published_at.date().isoformat()}"
    )
    st.write(assessment.reason)
    if (
        assessment.raw_relevance_score is not None
        and assessment.raw_relevance_score != assessment.relevance_score
        and assessment.profile_tier is not None
    ):
        st.caption(
            f"Tier calibration: raw LLM score {assessment.raw_relevance_score} -> "
            f"{assessment.relevance_score} "
            f"({assessment.profile_tier.value.replace('_', ' ')})."
        )
    left, right = st.columns(2)
    left.markdown(
        "**Matched:** " + (", ".join(assessment.matched_facets) or "None identified")
    )
    right.markdown(
        "**Missing:** " + (", ".join(assessment.missing_facets) or "None identified")
    )
    st.markdown(f"[Abstract]({paper.abs_url}) · [PDF]({paper.pdf_url})")
    with st.expander("Abstract"):
        st.write(paper.abstract)
    _render_feedback(recommendation, profile, store)
    st.divider()


def _render_results(result: RecommendationRun, store: FeedbackStore) -> None:
    selected_count = len(result.ranking.selected)
    candidate_count = len(result.recall.candidates)
    first, second, third, fourth = st.columns(4)
    first.metric("Papers fetched", result.fetched_count)
    second.metric("Candidates", candidate_count)
    third.metric("Relevant papers", selected_count)
    fourth.metric("Threshold", result.ranking.relevance_threshold)

    if result.fetch_report is not None:
        report = result.fetch_report
        st.caption(
            f"Window coverage: {len(report.items)} papers "
            f"({report.new_submission_count} new submissions, "
            f"{report.revised_version_count} revised versions). "
            f"Scanned {report.scanned_count} of {report.api_total_results} all-time "
            "category records."
        )
        if report.truncated:
            st.warning(
                "The arXiv safety cap was reached before the lookback cutoff. "
                "This run is incomplete; increase Paper safety cap and run again."
            )
        elif report.complete_through_cutoff:
            st.success("Complete through the selected time-window cutoff.")

    if result.fetched_count == 0:
        st.info("No papers were found in the selected time window and categories.")
        return
    if candidate_count == 0:
        st.info("No paper passed the candidate-recall threshold. Review diagnostics below.")
    else:
        show_all = st.toggle(
            "Also show candidate papers that scored below the relevance threshold",
            value=False,
        )
        visible = result.ranking.items if show_all else result.ranking.selected
        if not visible:
            st.info("No candidate reached the relevance threshold.")
        for recommendation in visible:
            _render_recommendation(recommendation, result.profile, store)

    with st.expander("Score calibration", expanded=False):
        diagnostics = analyze_score_diagnostics(
            result.ranking.items, store.list_for_profile(result.profile)
        )
        score_rows = [
            {"Score": score, "Papers": count}
            for score, count in diagnostics.score_counts.items()
        ]
        region_rows = [
            {"Range": region, "Papers": count}
            for region, count in diagnostics.score_regions.items()
        ]
        st.caption(
            f"{diagnostics.unique_score_count} distinct scores across "
            f"{diagnostics.prediction_count} ranked papers. "
            f"{diagnostics.boundary_count} exact band-boundary scores "
            f"({diagnostics.boundary_rate:.0%})."
        )
        left, right = st.columns(2)
        left.dataframe(score_rows, width="stretch", hide_index=True)
        right.dataframe(region_rows, width="stretch", hide_index=True)
        if diagnostics.feedback_count:
            st.write(
                f"Against {diagnostics.feedback_count} saved user labels: "
                f"MAE {diagnostics.mean_absolute_error:.1f}, "
                f"mean bias {diagnostics.mean_bias:+.1f}, "
                f"band agreement {diagnostics.band_agreement:.0%}."
            )
        else:
            st.info(
                "No saved user labels match this exact profile yet. Correct paper scores "
                "to turn this distribution check into a calibration check."
            )

    with st.expander("Intermediate diagnostics", expanded=False):
        st.caption(
            "Recall diagnostics include every fetched paper; LLM ranking runs only on included "
            "candidates."
        )
        rows = []
        update_kinds = (
            {
                fetched.paper.arxiv_id: fetched.update_kind.value.replace("_", " ").title()
                for fetched in result.fetch_report.items
            }
            if result.fetch_report is not None
            else {}
        )
        for item in result.recall.items:
            rows.append(
                {
                    "arXiv ID": item.paper.arxiv_id,
                    "Title": item.paper.title,
                    "Update": update_kinds.get(item.paper.arxiv_id, "Unknown"),
                    "Candidate": item.included,
                    "Hybrid": round(item.signals.hybrid_score, 3),
                    "Embedding": (
                        round(item.signals.embedding_score, 3)
                        if item.signals.embedding_score is not None
                        else None
                    ),
                    "Keyword": round(item.signals.keyword_score, 3),
                    "BM25": round(item.signals.bm25_score, 3),
                    "Intersection": round(item.signals.intersection_coverage, 3),
                    "Negative penalty": round(item.signals.negative_penalty, 3),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            f"Embedding: {result.recall.embedding_provider or 'disabled'} / "
            f"{result.recall.embedding_model or 'n/a'} · Ranking: "
            f"{result.ranking.provider} / {result.ranking.model} · "
            f"Cached rankings: {result.ranking.cached_count}"
        )


def run_app() -> None:
    st.set_page_config(page_title="ArxivRec", page_icon="◌", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1120px; padding-top: 2rem;}
        h1 {letter-spacing: -0.04em;}
        [data-testid="stMetric"] {
            border: 1px solid #dfe4ea;
            padding: 0.8rem;
            border-radius: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    settings = AppSettings()
    fixture_profile, _, _ = load_fixture(FIXTURE_PATH)
    if "profile" not in st.session_state:
        _set_profile(fixture_profile)
        st.session_state["analyzed_source"] = ""

    st.title("ArxivRec")
    st.write(
        "Find arXiv papers that match your research interests. ArxivRec reads titles and "
        "abstracts from your chosen dates, scores each paper from 0 to 100, and shows papers "
        "at or above your relevance threshold."
    )
    with st.expander("How to use ArxivRec", expanded=True):
        st.markdown(
            """
            1. **Describe your research interests** and what makes a paper useful to you.
            2. Click **1. Analyze research interests**. AI will turn your description into a
               structured research profile used for searching and scoring; this step does not
               search arXiv yet.
            3. Optionally review and edit the extracted profile.
            4. Choose a search period and relevance threshold in the sidebar.
            5. Click **2. Find relevant papers**. Each result includes a relevance score,
               confidence level, and explanation.
            """
        )

    with st.sidebar:
        st.header("Search settings")
        if settings.live_openai_available:
            st.success("OpenAI key detected locally.")
        else:
            st.warning("Add OPENAI_API_KEY to .env before a live run.")
        time_mode = st.radio(
            "Search period",
            ("Lookback days", "Specific date range"),
            horizontal=True,
            help=(
                "Use a rolling number of days ending now, or select inclusive start and "
                "end dates on the calendar."
            ),
        )
        lookback_days = 1
        specific_date_range: tuple[date, date] | None = None
        if time_mode == "Lookback days":
            lookback_days = int(
                st.number_input(
                    "Lookback days",
                    1,
                    30,
                    1,
                    help=(
                        "A rolling 24-hour window when set to one; increase it after "
                        "weekends or missed runs."
                    ),
                )
            )
        else:
            today = datetime.now().astimezone().date()
            chosen_dates = st.date_input(
                "Start and end dates",
                value=(today - timedelta(days=1), today),
                max_value=today,
                help=(
                    "Both dates are included. The range uses this computer's local timezone "
                    "and each paper's arXiv update timestamp."
                ),
            )
            if (
                isinstance(chosen_dates, (tuple, list))
                and len(chosen_dates) == 2
                and all(isinstance(value, date) for value in chosen_dates)
            ):
                specific_date_range = (chosen_dates[0], chosen_dates[1])
            else:
                st.warning("Select both a start date and an end date.")
        relevance_threshold = st.slider(
            "Relevance threshold",
            0,
            100,
            40,
            help=(
                "Only papers scoring at least this value are shown as relevant. "
                "Every paper is scored from 0 to 100."
            ),
        )
        with st.expander("Advanced settings", expanded=False):
            max_results = st.number_input(
                "Paper safety cap",
                1,
                500,
                100,
                step=10,
                help=(
                    "Papers are scanned newest-update-first until the selected period is "
                    "covered. This cap prevents unexpectedly broad searches; the results "
                    "page warns if coverage is incomplete."
                ),
            )
            candidate_threshold = st.slider(
                "Candidate recall threshold",
                0.0,
                1.0,
                0.20,
                0.01,
                help=(
                    "Controls the inexpensive first-pass filter before AI scoring. Keep this "
                    "permissive so potentially relevant papers are not discarded too early."
                ),
            )
            max_candidates = st.number_input(
                "Maximum papers sent for AI scoring",
                1,
                200,
                20,
                help="This is a hard cap on individual ranking API calls per search.",
            )

    source_text = st.text_area(
        "Describe your research interests",
        value="",
        height=135,
        key="research_description",
        placeholder=(
            "Example: I work on cavity QED for quantum interconnects, especially "
            "neutral-atom-superconducting hybrid systems for microwave-to-optical "
            "transduction. I also follow cold-atom experiments and quantum algorithms."
        ),
        help=(
            "Include your main topics, experimental or theoretical methods, physical "
            "platforms, nearby interests, and subjects you do not want when useful."
        ),
    )
    st.caption(
        "First analyze this description into a structured research profile. You can review "
        "the extracted topics before searching arXiv."
    )
    analyze_col, run_col, _ = st.columns([1, 1.5, 4])
    analyze_clicked = analyze_col.button(
        "1. Analyze research interests",
        width="stretch",
        help="Creates the structured research profile used for searching and scoring.",
    )
    run_clicked = run_col.button(
        "2. Find relevant papers",
        type="primary",
        width="stretch",
        help="Searches arXiv for the selected period and scores relevant titles and abstracts.",
    )

    if analyze_clicked:
        try:
            if not source_text.strip():
                raise ValueError("Describe your research interests before analyzing them.")
            with st.spinner("Analyzing the research profile…"):
                analyzed = _analyze_profile(source_text=source_text, settings=settings)
            _set_profile(analyzed)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if run_clicked:
        try:
            if not source_text.strip():
                raise ValueError("Describe your research interests before starting a search.")
            if time_mode == "Specific date range" and specific_date_range is None:
                raise ValueError("Select both a start date and an end date.")
            profile_is_stale = source_text.strip() != st.session_state["analyzed_source"]
            if profile_is_stale:
                with st.spinner("Analyzing the updated research profile…"):
                    active_profile = _analyze_profile(
                        source_text=source_text, settings=settings
                    )
                _set_profile(active_profile)
            else:
                active_profile = _edited_profile(source_text)
            with st.spinner("Retrieving candidates and assessing relevance…"):
                st.session_state["last_run"] = _execute_run(
                    profile=active_profile,
                    settings=settings,
                    lookback_days=int(lookback_days),
                    specific_date_range=specific_date_range,
                    max_results=int(max_results),
                    candidate_threshold=float(candidate_threshold),
                    max_candidates=int(max_candidates),
                    relevance_threshold=int(relevance_threshold),
                )
        except Exception as exc:
            st.error(str(exc))

    if st.session_state["analyzed_source"]:
        _render_profile_editor()
    else:
        st.caption("No research profile has been created yet. Start with step 1 above.")
    result = st.session_state.get("last_run")
    if isinstance(result, RecommendationRun):
        st.header("Relevant papers")
        st.caption(
            "Papers at or above your relevance threshold are shown by default. Each result "
            "includes the score, confidence, and the reason it matches your interests."
        )
        store = FeedbackStore(settings.database_path)
        _render_results(result, store)
    else:
        st.info(
            "To begin: describe your research interests, analyze them in step 1, choose a "
            "search period, and find relevant papers in step 2."
        )
    st.caption(
        "Thank you to arXiv for use of its open access interoperability. "
        "This independent project is not endorsed by arXiv."
    )
