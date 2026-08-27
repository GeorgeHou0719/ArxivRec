# Local MVP specification

## Confirmed decisions

- Audience: one local user
- Interface language: English
- Primary interface: local Streamlit application
- Immediate execution: supported through **Run Recommendation Now**
- Model provider: OpenAI
- Paper content: title, abstract, authors, categories, and metadata; no PDF parsing
- Selection policy: relevance threshold, not a fixed top-k quota
- Delivery: local result inspection only for this MVP
- Deferred: email, GitHub Actions, hosted UI, authentication, and multi-user support

## Pipeline

1. Parse a free-form research description into an editable structured profile.
2. Retrieve recent papers from broad, confirmed arXiv categories.
3. Produce a high-recall candidate set with category, lexical, and embedding signals.
4. Ask an LLM to assess candidates against the complete profile using a fixed rubric.
5. Validate structured model output and separate relevance from confidence.
6. Select every paper above the configured relevance threshold.
7. Display all intermediate signals and allow the user to record feedback.

## Provisional relevance rubric

The boundaries below are test labels, not calibrated probabilities. They must be reviewed
against a human-labelled evaluation set before being treated as product defaults.

| Score | Label | Interpretation |
| ---: | --- | --- |
| 90-100 | Essential | Directly addresses the core research intersection |
| 75-89 | Highly relevant | Strongly matches a primary goal or platform |
| 55-74 | Related | Useful adjacent work with a meaningful connection |
| 35-54 | Peripheral | Shares only a method, platform, or broad field |
| 0-34 | Irrelevant | No meaningful connection to the research profile |

Confidence is reported independently as `low`, `medium`, or `high`.

## Checkpoints

1. **Foundation:** schemas, configuration, fixtures, linting, and offline tests.
2. **arXiv ingestion:** parsing, time filtering, pagination, caching, and deduplication.
3. **Profile review:** inspect the structured cavity-QED profile with the user.
4. **Recall review:** inspect lexical, embedding, and category recall diagnostics.
5. **Ranking review:** compare relevance judgements, explanations, latency, and cost.
6. **UI review:** exercise fixture, dry-run, and live modes from the English UI.
7. **End-to-end review:** inspect real recent papers and repeated-run behaviour.

Checkpoints 3, 5, and 7 require user review before proceeding.

## Privacy and safety

- API keys are read from environment variables and never stored in fixtures or source files.
- The application sends only the user profile and paper metadata required for the selected
  OpenAI operation.
- Live and fixture modes are explicit; tests must not silently make paid API calls.
- Model output is validated before it can influence a recommendation.

