# Validation record

Validated locally on 2026-08-19. Secrets are loaded only from the Git-ignored `.env` file.

## Deterministic checks

- 54 tests passed, including a one-click Streamlit fixture run.
- Ruff linting passed.
- MyPy strict type checking passed for the application package.
- The four-paper calibration fixture produced scores 96, 82, 47, and 12 and selected only
  the first two at a threshold of 75.

## Live stage checks

- The official arXiv API returned and parsed 10 recent papers from the configured category
  query.
- OpenAI profile extraction returned a validated compositional research profile.
- `text-embedding-3-small` returned three 1536-dimensional vectors. The cavity-network
  comparison scored 0.730 cosine similarity versus 0.490 for the classical integrated-
  photonics comparison.
- The pointwise ranker scored the approved direct-match calibration paper as 90, Essential,
  with high confidence.
- A capped end-to-end run fetched 10 current papers, recalled 3, and recommended 0 at the
  threshold of 75. The three assessed papers scored 35, 35, and 0. This is the expected
  threshold behavior: an empty recommendation set is valid when no recent paper is relevant.

These live checks prove connectivity and pipeline behavior, not ranking generalization. User
feedback collected through the app should be used to build a larger, real gold set before
claiming recommendation-quality metrics.

## Coverage and ranking regression

Validated on 2026-08-21 with the supplied neutral-atom/superconducting cavity-QED profile:

- The one-day fetch was complete through the cutoff: 79 papers in the window, comprising 52 new
  submissions and 27 revised versions. The scanner inspected 100 newest-update-first records and
  stopped after crossing the cutoff; the API reported 328,378 all-time records for the category
  union.
- Hybrid recall sent 20 candidates to pointwise ranking. One paper crossed the 75 threshold:
  *Programmable cavity QED with a fiber-integrated atomic array* at 78. The remaining scores
  ranged from 48 down to 5.
- No paper received 35, 55, 75, or 90, so this run had zero exact band-boundary scores and did not
  reproduce the earlier 35-point cluster.
- Profile extraction now preserves priority phrases, keeps "generally interested" topics outside
  the required core, and adds one explicit positive example per independent required intersection.
- A four-paper user-approved seed remains too small to claim calibration. Live `gpt-5.4-mini`
  judgments varied across repeated runs, especially for the weakly related cavity-optics example;
  the UI therefore reports score distributions and matched user-feedback metrics instead of
  claiming that the current scale is fully calibrated. The final `ranking-v10` snapshot scored
  the four papers 98, 18, 15, and 5 versus labels 96, 82, 47, and 12 (MAE 26.25; 50% band
  agreement), so expanding the gold set and improving calibration remains explicit follow-up work.

## arXiv rate-limit regression

Validated on 2026-08-20 after adding official API safeguards:

- A live one-day `physics.optics` query returned 14 papers with one HTTP request.
- Repeating the identical fetch immediately returned the same 14 papers with zero additional
  HTTP requests.
- Automated tests verify a shared 3.1-second cross-client request interval, serialized
  connections, newest-update-first cutoff scanning, new/revised classification, explicit
  safety-cap truncation, and persisted HTTP 429 cooldowns.
