"""Mobile-friendly email digest rendering with no delivery side effects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

from arxiv_rec.models import PaperUpdateKind, Recommendation
from arxiv_rec.pipeline import RecommendationRun


@dataclass(frozen=True)
class DigestContent:
    """Delivery-ready email subject and equivalent HTML/plain-text bodies."""

    subject: str
    html: str
    text: str


def _display_label(recommendation: Recommendation) -> str:
    return recommendation.assessment.label.value.replace("_", " ").title()


def _score_color(score: int) -> str:
    if score >= 90:
        return "#9f1239"
    if score >= 75:
        return "#7c3aed"
    if score >= 55:
        return "#0f766e"
    return "#475569"


def _short_date(value: date) -> str:
    return f"{value:%b} {value.day}"


def _long_date(value: date) -> str:
    return f"{value:%A, %B} {value.day}, {value.year}"


def _paper_update_labels(run: RecommendationRun) -> dict[str, str]:
    if run.fetch_report is None:
        return {}
    labels = {
        PaperUpdateKind.NEW_SUBMISSION: "New submission",
        PaperUpdateKind.REVISED_VERSION: "Revised version",
    }
    return {
        item.paper.arxiv_id: labels[item.update_kind] for item in run.fetch_report.items
    }


def _facet_line(title: str, values: tuple[str, ...], *, empty: str) -> str:
    content = ", ".join(values) if values else empty
    return (
        '<div style="margin-top:7px;font-size:13px;line-height:1.45;color:#475569;">'
        f'<strong style="color:#334155;">{escape(title)}:</strong> {escape(content)}'
        "</div>"
    )


def _paper_card(recommendation: Recommendation, update_label: str | None) -> str:
    paper = recommendation.paper
    assessment = recommendation.assessment
    score = assessment.relevance_score
    color = _score_color(score)
    authors = ", ".join(paper.authors)
    meta_parts = [
        update_label,
        f"{_short_date(paper.published_at.date())}, {paper.published_at.year}",
        paper.primary_category,
        f"arXiv:{paper.arxiv_id}",
    ]
    meta = " · ".join(part for part in meta_parts if part)
    abs_url = escape(paper.abs_url or f"https://arxiv.org/abs/{paper.arxiv_id}", quote=True)
    pdf_url = escape(paper.pdf_url or f"https://arxiv.org/pdf/{paper.arxiv_id}", quote=True)
    return f"""
      <tr>
        <td style="padding:0 18px 16px;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
                 style="border:1px solid #e2e8f0;border-radius:14px;background:#ffffff;">
            <tr>
              <td style="padding:18px;">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                  <tr>
                    <td width="62" valign="top">
                      <div style="width:52px;height:52px;border-radius:12px;background:{color};
                                  color:#ffffff;text-align:center;line-height:52px;font-size:22px;
                                  font-weight:750;">{score}</div>
                    </td>
                    <td valign="top" style="padding-left:4px;">
                      <div style="font-size:12px;font-weight:700;letter-spacing:.04em;
                                  text-transform:uppercase;color:{color};">
                        {escape(_display_label(recommendation))}
                      </div>
                      <div style="margin-top:4px;font-size:19px;line-height:1.3;
                                  font-weight:750;color:#0f172a;">
                        {escape(paper.title)}
                      </div>
                    </td>
                  </tr>
                </table>
                <div style="margin-top:12px;font-size:13px;line-height:1.45;color:#64748b;">
                  {escape(authors)}
                </div>
                <div style="margin-top:4px;font-size:12px;line-height:1.4;color:#94a3b8;">
                  {escape(meta)} · Confidence: {escape(assessment.confidence.value.title())}
                </div>
                <div style="margin-top:15px;padding:13px 14px;border-radius:10px;
                            background:#f8fafc;border-left:3px solid {color};">
                  <div style="font-size:11px;font-weight:700;letter-spacing:.06em;
                              text-transform:uppercase;color:#64748b;">Why it matches</div>
                  <div style="margin-top:5px;font-size:14px;line-height:1.55;color:#1e293b;">
                    {escape(assessment.reason)}
                  </div>
                </div>
                {_facet_line("Matched", assessment.matched_facets, empty="None identified")}
                {_facet_line("Missing", assessment.missing_facets, empty="Nothing important")}
                <div style="margin-top:15px;font-size:13px;line-height:1.55;color:#475569;">
                  <strong style="color:#334155;">Abstract:</strong>
                  {escape(paper.abstract)}
                </div>
                <div style="margin-top:17px;">
                  <a href="{abs_url}" style="display:inline-block;padding:9px 13px;
                     border-radius:8px;background:#0f172a;color:#ffffff;text-decoration:none;
                     font-size:13px;font-weight:700;">View on arXiv</a>
                  <a href="{pdf_url}" style="display:inline-block;margin-left:8px;padding:9px 13px;
                     border-radius:8px;border:1px solid #cbd5e1;color:#334155;
                     text-decoration:none;font-size:13px;font-weight:700;">Open PDF</a>
                </div>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    """


def _empty_state(threshold: int) -> str:
    return f"""
      <tr>
        <td style="padding:0 18px 20px;">
          <div style="padding:26px 20px;border:1px solid #e2e8f0;border-radius:14px;
                      background:#ffffff;text-align:center;">
            <div style="font-size:18px;font-weight:750;color:#0f172a;">
              No papers reached your relevance threshold today.
            </div>
            <div style="margin-top:8px;font-size:14px;line-height:1.5;color:#64748b;">
              The daily scan completed successfully. Your current threshold is {threshold}.
            </div>
          </div>
        </td>
      </tr>
    """


def build_digest(
    run: RecommendationRun,
    *,
    delivery_timezone: str = "America/Los_Angeles",
    digest_date: date | None = None,
) -> DigestContent:
    """Build an email digest from one completed run without sending it anywhere."""
    timezone = ZoneInfo(delivery_timezone)
    local_date = digest_date or run.completed_at.astimezone(timezone).date()
    selected = run.ranking.selected
    selected_count = len(selected)
    candidate_count = len(run.recall.candidates)
    top_score = selected[0].assessment.relevance_score if selected else None
    count_label = "paper" if selected_count == 1 else "papers"
    subject = f"ArxivRec · {_short_date(local_date)} · {selected_count} relevant {count_label}"
    if top_score is not None:
        subject += f" · top score {top_score}"

    update_labels = _paper_update_labels(run)
    cards = "".join(
        _paper_card(item, update_labels.get(item.paper.arxiv_id)) for item in selected
    )
    if not cards:
        cards = _empty_state(run.ranking.relevance_threshold)

    preheader = (
        f"{selected_count} relevant {count_label} from {run.fetched_count} scanned; "
        f"threshold {run.ranking.relevance_threshold}."
    )
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <meta name="color-scheme" content="light">
    <title>{escape(subject)}</title>
  </head>
  <body style="margin:0;padding:0;background:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,
               'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#0f172a;">
    <div style="display:none;max-height:0;overflow:hidden;opacity:0;color:transparent;">
      {escape(preheader)}
    </div>
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
           style="background:#f1f5f9;">
      <tr>
        <td align="center" style="padding:22px 8px;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
                 style="max-width:660px;background:#f8fafc;border-radius:18px;overflow:hidden;">
            <tr>
              <td style="padding:25px 20px 18px;background:#111827;color:#ffffff;">
                <div style="font-size:12px;font-weight:750;letter-spacing:.12em;
                            text-transform:uppercase;color:#fda4af;">ArxivRec daily digest</div>
                <div style="margin-top:7px;font-size:25px;line-height:1.25;font-weight:780;">
                  Research worth your attention
                </div>
                <div style="margin-top:7px;font-size:14px;color:#cbd5e1;">
                  {_long_date(local_date)} · {escape(delivery_timezone)}
                </div>
              </td>
            </tr>
            <tr>
              <td style="padding:16px 18px;">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
                       style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;">
                  <tr>
                    <td width="33%" align="center" style="padding:14px 5px;">
                      <div style="font-size:21px;font-weight:780;color:#0f172a;">
                        {run.fetched_count}
                      </div>
                      <div style="font-size:11px;text-transform:uppercase;color:#64748b;">
                        scanned
                      </div>
                    </td>
                    <td width="34%" align="center"
                        style="padding:14px 5px;border-left:1px solid #e2e8f0;
                               border-right:1px solid #e2e8f0;">
                      <div style="font-size:21px;font-weight:780;color:#0f172a;">
                        {candidate_count}
                      </div>
                      <div style="font-size:11px;text-transform:uppercase;color:#64748b;">
                        assessed
                      </div>
                    </td>
                    <td width="33%" align="center" style="padding:14px 5px;">
                      <div style="font-size:21px;font-weight:780;color:#9f1239;">
                        {selected_count}
                      </div>
                      <div style="font-size:11px;text-transform:uppercase;color:#64748b;">
                        relevant
                      </div>
                    </td>
                  </tr>
                </table>
                <div style="margin-top:10px;font-size:12px;text-align:center;color:#64748b;">
                  Showing every paper scoring at least {run.ranking.relevance_threshold};
                  no fixed paper limit.
                </div>
              </td>
            </tr>
            {cards}
            <tr>
              <td style="padding:8px 22px 24px;text-align:center;font-size:11px;
                         line-height:1.5;color:#94a3b8;">
                Scores are relevance judgments based on titles and abstracts, not measures of
                scientific quality. Thank you to arXiv for use of its open access interoperability.
                This independent project is not endorsed by arXiv.
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

    text_lines = [
        "ArxivRec daily digest",
        f"{_long_date(local_date)} · {delivery_timezone}",
        "",
        f"{run.fetched_count} papers scanned",
        f"{candidate_count} papers assessed",
        f"{selected_count} papers at or above threshold {run.ranking.relevance_threshold}",
    ]
    if not selected:
        text_lines.extend(["", "No papers reached your relevance threshold today."])
    for item in selected:
        assessment = item.assessment
        paper = item.paper
        text_lines.extend(
            [
                "",
                "─" * 48,
                f"{assessment.relevance_score} · {_display_label(item)}",
                paper.title,
                ", ".join(paper.authors),
                f"Why it matches: {assessment.reason}",
                "Matched: " + (", ".join(assessment.matched_facets) or "None identified"),
                "Missing: " + (", ".join(assessment.missing_facets) or "Nothing important"),
                f"Abstract: {paper.abstract}",
                f"arXiv: {paper.abs_url}",
                f"PDF: {paper.pdf_url}",
            ]
        )
    return DigestContent(subject=subject, html=html, text="\n".join(text_lines) + "\n")


def write_digest_preview(
    run: RecommendationRun,
    output_path: Path,
    *,
    delivery_timezone: str = "America/Los_Angeles",
    digest_date: date | None = None,
) -> DigestContent:
    """Render a digest and write its HTML body for local browser inspection."""
    content = build_digest(
        run,
        delivery_timezone=delivery_timezone,
        digest_date=digest_date,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.html, encoding="utf-8")
    return content
