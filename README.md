# Google Fact Check Tools - Indonesia

Four [marimo](https://marimo.io/) notebooks for extracting and analyzing Indonesian fact-check records returned by Google Fact Check Tools.

- `notebooks/00_google_fact_check_tool_id.py` contains the complete extraction pipeline and API refresh control.
- `notebooks/01_google_fact_check_exploratory_data_analysis_id.py` performs offline exploratory data analysis across time, publishers, keywords, and text.
- `notebooks/02_google_fact_check_generative_ai_hoax_distribution_id.py` uses all reviewed claims with parseable dates for the primary pre/post comparison around ChatGPT's public launch, retains balanced three-year windows as a sensitivity check, and audits rule-confirmed AI-linked claims.
- `notebooks/03_google_fact_check_relational_network_analysis_id.py` provides descriptive, temporal, and relational network analysis of keywords, publishers, and model-identified actors.
- `data/google_fact_check_tool_id.parquet` is the source of truth shared by all reports.
- `data/google_fact_check_ner_entities_id.parquet` is the pinned, derived NER cache used by the network report.

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

Install the locked project environment:

```sh
uv sync
```

## Run the notebooks

The three analysis notebooks use only committed local data and make no network requests:

```sh
uv run marimo run notebooks/01_google_fact_check_exploratory_data_analysis_id.py
uv run marimo run notebooks/02_google_fact_check_generative_ai_hoax_distribution_id.py
uv run marimo run notebooks/03_google_fact_check_relational_network_analysis_id.py
```

The extraction notebook also requires local configuration:

```sh
cp .env.example .env
```

Set both values in `.env`:

```dotenv
PROJECT_ID=your-gcp-project-id
KEY_DISPLAY_NAME=ddf-hoax-temporal
```

Then run:

```sh
uv run marimo run notebooks/00_google_fact_check_tool_id.py
```

The extraction notebook reads the cached Parquet by default. It only accesses Google APIs when the file is missing or **Re-extract from API** is selected.

Refreshing requires Application Default Credentials, permission to manage API keys in `PROJECT_ID`, and the Fact Check Tools and API Keys APIs:

```sh
gcloud auth application-default login
```

## Refresh the Indonesian NER cache

Normal notebook and site builds never download a model. Refresh NER only when the source claim texts or pinned model configuration change:

```sh
uv sync --extra ner
uv run --extra ner python scripts/refresh-ner-entities.py
```

The optional NER environment uses CPU-only PyTorch wheels on non-macOS platforms. The refresh uses the MIT-licensed Hugging Face model [`cahya/bert-base-indonesian-NER`](https://huggingface.co/cahya/bert-base-indonesian-NER) pinned at revision `a3a3fa494cf7555ef87f446af5e826de3ed181c0`. Every downloaded model file is checked against a pinned SHA-256 digest. Model files are cached under `tmp/huggingface/`; only the deterministic derived Parquet is written to `data/`. Long inputs use overlapping tokenizer windows, and a second unchanged run exits without rewriting the cache.

The network notebook validates the cache's source-text hash, model revision, and inference settings. A missing or stale cache fails with the exact refresh command instead of silently falling back to live inference.

## Network semantics

The network report keeps four relationships separate:

- keyword ↔ keyword means two configured queries retrieved the same reviewed claim;
- publisher ↔ keyword means an indexed reviewed claim from that publisher matched the query;
- actor ↔ actor means two PERSON/ORGANIZATION or retained claimant candidates occur in the same unique claim text;
- actor ↔ publisher means the publisher reviewed a claim containing or naming that actor candidate.

These are retrieval, textual co-mention, and indexed-coverage associations. They are not evidence of diffusion, coordination, endorsement, affiliation, influence, audience exposure, or causality. The report includes threshold sensitivity, claimant exclusions, NER-span audits, temporal comparisons, and eager node/edge CSV downloads.

## Build the static site

All exports execute against committed local Parquet files. The extraction page includes source code, while all three analysis reports are available with and without code.

Run the staged build script from the repository root:

```sh
./scripts/build-static-site.sh
```

The script checks all four notebooks, exports seven generated pages into a temporary staging directory, and replaces files in `docs/` only after every export succeeds. If publication fails, it restores the prior generated pages. It leaves the hand-maintained `docs/index.html` unchanged.

Published network reports:

- `docs/network-analysis.html`
- `docs/network-analysis-with-code.html`

Preview the generated files:

```sh
uv run python -m http.server 8000 --directory docs
```

Open <http://localhost:8000>. Static exports do not have a Python kernel, so extraction refresh only works in a live marimo session. Analysis tables, charts, explorers, and embedded downloads work client-side.

## Interpretation note

The dataset is conditioned on the configured queries, participating publishers, and Google's fact-check index. It is not a census of misinformation in Indonesia. The applied study's primary temporal comparison uses all 6,367 reviewed claims with parseable dates (2,094 pre-launch and 4,273 post-launch); the balanced three-year sensitivity uses 1,666 and 2,998, respectively. Model-generated entities and structured claimants remain exploratory candidates, and centrality or community membership must not be interpreted as real-world importance or coordination.
