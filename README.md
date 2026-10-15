# Indonesian Google Fact Check Tools API Snapshot

A reproducible, preliminary study of Indonesian fact-check records returned by the [Google Fact Check Tools API](https://developers.google.com/fact-check/tools/api). The project separates corpus construction, exploratory analysis, network analysis, and a bounded study of explicit AI-linked claims.

**Published reports:** <https://gstdl.github.io/google-fact-checks-api/>

## Current snapshot

The committed snapshot was extracted on **22 August 2026 at 14:54:52 UTC**. Its main analytical grain is one distinct `(review_url, claim_text)` pair.

- **7,746** reviewed-claim records and **7,673** distinct claim texts
- **112** configured retrieval queries and **11** indexed publisher sites
- **6,367** records with parseable claim dates
- `suara.com`, `tempo.co`, and `turnbackhoax.id` account for **82.6%** of all reviewed-claim records
- **14** dated, rule-confirmed AI-linked records, all observed after ChatGPT's public launch
- At keyword-edge support ≥5, `video` has the highest betweenness in the retrieval-overlap graph

These results describe this query-conditioned Google index snapshot. They do not estimate misinformation prevalence or establish interaction, coordination, affiliation, influence, transformation, or causation.

## Reports and repository layout

| Path | Purpose |
| --- | --- |
| `notebooks/00_google_fact_check_tool_id.py` | Extracts API responses, archives raw pages, validates provenance, and publishes the canonical Parquet snapshot. |
| `notebooks/01_google_fact_check_exploratory_data_analysis_id.py` | Examines metadata, dates, publishers, retrieval overlap, ratings, claimants, and mechanical text patterns. |
| `notebooks/02_google_fact_check_generative_ai_hoax_distribution_id.py` | Audits a strict AI-linked subset and compares pre/post-launch periods with balanced-window sensitivity. |
| `notebooks/03_google_fact_check_relational_network_analysis_id.py` | Analyzes keyword overlap, indexed publisher coverage, candidate-name co-mentions, and candidate–publisher coverage. |
| `data/google_fact_check_tool_id.parquet` | Canonical source shared by all analytical reports. |
| `data/google_fact_check_tool_id_snapshots/` | Immutable raw API pages, metadata manifest, and checksum-linked flattened snapshot. |
| `data/google_fact_check_ner_entities_id.parquet` | Deterministic, derived NER cache used by the network report. |
| `docs/index.html` | Hand-maintained landing page. |
| `scripts/build-static-site.sh` | Checks notebooks and atomically regenerates the seven report pages in `docs/`. |
| `scripts/refresh-ner-entities.py` | Rebuilds the pinned Indonesian NER cache. |
| `scripts/download-spike-videos.py` | Finds fact-checked election videos in the snapshot and downloads `--n` of them, live or from perma.cc archives, for the AI-detection spike test. |

## Prerequisites

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)

Install the locked environment:

```sh
uv sync
```

## Run the analytical notebooks

The three analytical notebooks use only committed local Parquet files and do not make network requests:

```sh
uv run marimo run notebooks/01_google_fact_check_exploratory_data_analysis_id.py
uv run marimo run notebooks/02_google_fact_check_generative_ai_hoax_distribution_id.py
uv run marimo run notebooks/03_google_fact_check_relational_network_analysis_id.py
```

Check all notebooks without opening the interactive interface:

```sh
uv run marimo check notebooks/*.py
```

## Refresh the API snapshot

The extraction notebook reads the committed Parquet by default. It contacts Google only when the canonical file is missing or **Re-extract from API** is selected.

Copy the local configuration template:

```sh
cp .env.example .env
```

Set the Google Cloud project and API-key display name:

```dotenv
PROJECT_ID=your-gcp-project-id
KEY_DISPLAY_NAME=google-fact-check-tools-research
```

Authenticate with Application Default Credentials, then run the extraction notebook:

```sh
gcloud auth application-default login
uv run marimo run notebooks/00_google_fact_check_tool_id.py
```

Refreshing requires permission to manage API keys in `PROJECT_ID` and access to the Fact Check Tools and API Keys APIs. The pipeline:

1. queries the API's `claims:search` endpoint using the configured vocabulary;
2. archives deterministic, compressed raw response pages;
3. expands each API `claimReview` object into one Indonesian review row;
4. de-duplicates and validates the staged dataset and provenance metadata; and
5. replaces the canonical output only after every validation succeeds.

The project does not scrape or parse Schema.org `ClaimReview` markup.

## Refresh the Indonesian NER cache

Normal notebook and site builds never download a model. Refresh the cache only when source claim texts or the pinned model configuration changes:

```sh
uv sync --extra ner
uv run --extra ner python scripts/refresh-ner-entities.py
```

The refresh uses:

- [`cahya/bert-base-indonesian-NER`](https://huggingface.co/cahya/bert-base-indonesian-NER), pinned to revision `a3a3fa494cf7555ef87f446af5e826de3ed181c0` under the MIT license;
- [`BertTokenizerFast`](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizerFast);
- 512-token windows with a 64-token overlap, `simple` entity aggregation, CPU inference, and seed 42; and
- pinned SHA-256 checksums for every downloaded model file.

The tokenizer segments claim text and preserves offsets for overlapping-window inference; the token-classification model produces the entity predictions. Model files remain under `tmp/huggingface/`. Only the deterministic derived Parquet is written to `data/`, and an unchanged rerun does not rewrite it.

The network notebook rejects a missing or stale cache by checking its source-text hash, model revision, tokenizer class, and inference settings. It does not silently fall back to live inference.

## Download spike video samples

The AI-detection spike test needs social media videos that fact-checkers reviewed. The script finds them from the snapshot Parquet rather than a fixed list:

1. selects reviews whose `claim_text` contains `--keyword` (default `video`), whose claim date falls between `--start` and `--end` (default: the 28 November 2023 – 10 February 2024 campaign period), and that an election retrieval keyword matched (`--topic-keywords`);
2. visits them oldest first and reads each fact-check article, from the Wayback Machine when the publisher refuses automated requests;
3. collects the TikTok, Instagram, Facebook, and X posts the article cites, directly or through perma.cc, in order of first mention;
4. downloads the first post that still has a video, live with `yt-dlp` or, for deleted posts, from the video stored in the perma.cc capture; and
5. stops after `--n` reviews (default 5) have a video.

Each run writes to `tmp/spike-videos/<epoch>/`, one folder per sample (`video.mp4`, plus `video.info.json` and captions for live downloads or `archive.warc.gz` for archived ones), and a `manifest.parquet` with provenance and checksums. The run folder appears only when all `--n` samples succeed; otherwise it is removed and the script exits with an error.

```sh
uv sync --extra video
uv run --extra video python scripts/download-spike-videos.py --n 5
```

Behind a TLS-inspecting proxy, add `--system-certs` to `uv` commands, export the macOS keychain certificates, and point `SSL_CERT_FILE` at them; the script then trusts that bundle instead of `certifi`:

```sh
security find-certificate -a -p /Library/Keychains/System.keychain \
  /System/Library/Keychains/SystemRootCertificates.keychain > tmp/ca.pem
SSL_CERT_FILE=tmp/ca.pem uv run --system-certs --extra video python scripts/download-spike-videos.py
```

`ai_mentioned_in_review` flags reviews whose title or claim text mentions AI or deepfakes; it is a text match, not a verified label. With the defaults, the campaign-period reviews are rated as miscontextualized or unproven real footage. Fact-checks of AI-manipulated videos sit just before the campaign, for example `--start 2023-10-01 --end 2023-11-27`.

## Network semantics

The network report keeps four edge meanings separate:

- **keyword ↔ keyword:** two configured queries retrieved the same reviewed claim;
- **publisher ↔ keyword:** an indexed reviewed claim from that publisher matched the query;
- **candidate ↔ candidate:** two model-generated or conservatively retained claimant candidates occur in the same unique claim text; and
- **candidate ↔ publisher:** the publisher reviewed a claim containing that candidate label.

These edges encode retrieval overlap, textual co-mention, or indexed review coverage—not social interaction or verified affiliation. Candidate names may contain NER errors or unresolved aliases and remain unverified mention candidates.

## Build and preview the static site

Build all generated reports from committed local data:

```sh
./scripts/build-static-site.sh
```

The script runs `marimo check`, exports every report into a temporary staging directory, verifies non-empty outputs, and replaces generated pages only after all exports succeed. A failed publication restores the previous files. The hand-maintained `docs/index.html` is not overwritten.

Generated pages:

- `docs/extraction.html`
- `docs/analysis.html`
- `docs/analysis-with-code.html`
- `docs/generative-ai.html`
- `docs/generative-ai-with-code.html`
- `docs/network-analysis.html`
- `docs/network-analysis-with-code.html`

Preview the site locally:

```sh
uv run python -m http.server 8000 --directory docs
```

Open <http://localhost:8000>. Static exports have no Python kernel, but embedded charts, client-side tables, explorers, and eager downloads remain available.

## Research transparency and AI use

Claude supported initial source discovery. The Pi agent harness—using `ChatGPT-Sol-5.6`, `Claude-Opus-5`, and `Claude-Sonnet-5` also assisted with code drafting, editorial revision, and visualization iteration. Published outputs were reviewed against the project data, source code, and cited documentation; the authors retain responsibility for the research design, analysis, interpretation, and publication.

AI-generated output was not treated as research evidence and did not determine whether a reviewed claim was true or false. Separately, the pinned NER model and tokenizer are analytical dependencies—not generative-AI writing tools. They identify unverified mention candidates rather than verified identities.

Core references and the same disclosure appear on the [published landing page](https://gstdl.github.io/google-fact-checks-api/).

## Interpretation limits

The extract is conditioned on the configured vocabulary, participating publishers, Google's index, and available metadata. The primary AI-era comparison uses **2,094** dated reviewed claims before 30 November 2022 and **4,273** on or after that date; those periods have unequal durations. The balanced three-year sensitivity uses **1,666** pre-launch and **2,998** post-launch records.

The reports preserve publisher-authored labels rather than mapping them to a universal true/false taxonomy. Centrality, communities, co-mentions, and candidate–publisher edges must not be interpreted as real-world importance, identity, interaction, coordination, endorsement, or causality.
