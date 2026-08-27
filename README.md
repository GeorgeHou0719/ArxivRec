# ArxivRec

ArxivRec is a local, single-user research-paper recommendation application. It turns a
free-form research description into a structured profile, retrieves recent arXiv papers,
and ranks them by semantic relevance with an explanation.

The current MVP focuses on recommendation quality and local inspection. Scheduled email delivery
is being added in explicit checkpoints; the first checkpoint generates a delivery-ready HTML and
plain-text digest locally without sending it.

## MVP scope

- English Streamlit interface
- Free-form research profile
- Title-and-abstract analysis only
- Hybrid lexical and embedding candidate retrieval
- OpenAI-assisted profile extraction and relevance ranking
- Threshold-based selection rather than a fixed number of papers
- New-submission versus revised-version labels with explicit fetch coverage status
- Score-distribution, boundary-concentration, and saved-feedback calibration diagnostics
- Fixture, dry-run, and live-test modes
- Inspectable intermediate scores and explanations

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
```

Live OpenAI calls are not required for the offline test suite. Before a live run, copy
`.env.example` to `.env` and set `OPENAI_API_KEY`. Never commit `.env`.

## Run the local app

```powershell
.\.venv\Scripts\streamlit run app.py
```

The app runs in **Live arXiv + OpenAI** mode after configuring `.env`. Describe and analyze your
research interests, then click **Find relevant papers**. The sidebar supports either a rolling
lookback window or an inclusive local calendar date range, and the default relevance threshold
is 40.

The deterministic fixture remains available to the automated test suite but is not shown in the
user interface. The structured profile is editable, recall diagnostics are available below the
results, and score corrections are stored locally in `data/arxiv_rec.sqlite3` for later evaluation
work.
The pointwise ranker returns both a semantic profile tier and a numeric score; contradictory
numbers are moved to a stable within-tier anchor, while the raw model score remains visible.

For a small immediate live smoke test from the terminal:

```powershell
.\.venv\Scripts\arxiv-rec run-live `
  --description "I work on cavity QED and quantum interconnects between remote nodes." `
  --lookback-days 7 --max-results 10 --max-candidates 3
```

`--max-candidates` is a hard cap on pointwise LLM ranking calls. Start small while testing.

## Preview the daily email locally

Generate a mobile-friendly digest from the deterministic fixture without calling arXiv, OpenAI,
or an email provider:

```powershell
.\.venv\Scripts\arxiv-rec preview-digest `
  --fixture fixtures\cavity_qed_papers.json `
  --output data\digest-preview.html `
  --relevance-threshold 40 `
  --delivery-timezone America/Los_Angeles
```

Open `data/digest-preview.html` in a browser to review the proposed email. The digest includes
every paper at or above the relevance threshold rather than imposing a fixed paper count. A
paper's complete arXiv abstract is included directly in both HTML and plain text; it is not
truncated or summarized by another model. A plain-text equivalent is built at the same time for
email-client fallback support.

The agreed personal delivery schedule for the next checkpoint is Monday through Friday at 05:00
in `America/Los_Angeles`, using a one-day lookback. Saturday and Sunday runs will be omitted.
## Send an immediate verification email

Create a Resend account with the same Stanford address that will receive the test message, then
add these values to the Git-ignored `.env` file:

```dotenv
RESEND_API_KEY=re_...
ARXIVREC_RECIPIENT=your-address@stanford.edu
ARXIVREC_EMAIL_FROM="ArxivRec <onboarding@resend.dev>"
```

Resend's `onboarding@resend.dev` test sender can send only to the email address associated with
the Resend account. This is sufficient for the current single-user deployment and does not need
a custom domain. Send the deterministic fixture immediately:

```powershell
.\.venv\Scripts\arxiv-rec send-verification-email `
  --fixture fixtures\cavity_qed_papers.json `
  --relevance-threshold 40 `
  --delivery-timezone America/Los_Angeles
```

This verifies Resend credentials, delivery to the Stanford inbox, mobile layout, links, and full
abstract rendering without calling arXiv or OpenAI. A successful command prints Resend's message
ID. It never prints API keys.

## Run and send a live digest locally

Analyze a description once and save the structured profile. Daily runs reuse this profile and do
not spend an additional profile-analysis call:

```powershell
.\.venv\Scripts\arxiv-rec analyze-profile `
  --description "I work on cavity QED for quantum interconnects." `
  --output config\research_profile.json
```

Then run the same live retrieval/ranking/email path used by the scheduler:

```powershell
.\.venv\Scripts\arxiv-rec send-daily-email `
  --profile-file config\research_profile.json `
  --lookback-days 1 `
  --max-results 100 `
  --candidate-threshold 0.20 `
  --max-candidates 20 `
  --relevance-threshold 40 `
  --delivery-timezone America/Los_Angeles
```

Unlike the fixture verification, this command makes live arXiv, OpenAI embedding/ranking, and
Resend requests.

## GitHub Actions delivery

`.github/workflows/daily-digest.yml` runs at 05:00 Monday through Friday in
`America/Los_Angeles`; Saturday and Sunday have no scheduled job. The scheduled command uses a
one-day lookback and relevance threshold 40. It also provides a manual **Run workflow** action
with two modes:

- `fixture`: send the offline fixture email immediately without OpenAI/arXiv calls.
- `live`: run the complete daily pipeline immediately, regardless of the schedule.

Configure these repository Actions secrets before using the workflow:

| Secret | Value |
| --- | --- |
| `OPENAI_API_KEY` | Existing OpenAI API key; required only by live runs |
| `RESEND_API_KEY` | Resend API key |
| `ARXIVREC_RECIPIENT` | Stanford recipient address |
| `ARXIVREC_PROFILE_JSON` | Entire validated `config/research_profile.json` content |

The fixture mode needs only the two email secrets. The scheduled/live mode needs all four.
Scheduled emails use a date-and-profile idempotency key so rerunning the same daily job does not
normally deliver a duplicate message.

## arXiv API etiquette

Live retrieval follows the official legacy-API limits: one connection at a time and no more
than one request every 3 seconds. ArxivRec uses a shared on-disk request gate across runs,
persists `Retry-After` cooldowns after HTTP 429, and caches successful pages for 24 hours.
Queries sort category matches by `lastUpdatedDate` and scan 50-result pages only until the
lookback cutoff is reached. This includes both new submissions and older papers revised in the
window. A day with 30 updates therefore normally needs one 50-record page rather than blindly
downloading the 100-paper safety cap. The UI reports the API's all-time `totalResults`, scanned
records, new/revised counts, and whether the safety cap made coverage incomplete. The local UI
defaults to a one-day lookback and a total safety cap of 100; increase the lookback manually
after missed runs.

Thank you to arXiv for use of its open access interoperability. This independent project is not
endorsed by arXiv.

## Project status

Implementation proceeds through explicit checkpoints documented in
[`docs/mvp-spec.md`](docs/mvp-spec.md). The relevance rubric and model choices remain
evaluation-driven rather than being treated as final product truths.

The completed offline and capped live smoke-test results are recorded in
[`docs/validation.md`](docs/validation.md).
