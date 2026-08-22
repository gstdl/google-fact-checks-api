# DDF Hoax Temporal

Two [marimo](https://marimo.io/) notebooks for extracting and analyzing Indonesian fact-check records returned by Google Fact Check Tools.

- `notebooks/00_google_fact_check_tool_id.py` contains the complete extraction pipeline and API refresh control.
- `notebooks/01_google_fact_check_analysis_id.py` performs offline temporal, publisher, keyword, and NLP analysis.
- `data/google_fact_check_tool_id.parquet` is the source of truth shared between them.

## Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

Install the locked project environment:

```sh
uv sync
```

## Run the notebooks

The analysis notebook only needs the included Parquet:

```sh
uv run marimo run notebooks/01_google_fact_check_analysis_id.py
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

## Build the static site

Both exports execute against the cached Parquet. The extraction page includes source code; the analysis page hides it.

```sh
mkdir -p site

uv run marimo export html \
  notebooks/00_google_fact_check_tool_id.py \
  --output site/extraction.html \
  --include-code \
  --force

uv run marimo export html \
  notebooks/01_google_fact_check_analysis_id.py \
  --output site/analysis.html \
  --no-include-code \
  --force
```

Preview the generated files:

```sh
uv run python -m http.server 8000 --directory site
```

Open <http://localhost:8000>. Static exports do not have a Python kernel, so extraction refresh only works in a live marimo session. The analysis tables, charts, explorer, and embedded downloads work client-side.

## Interpretation note

The dataset is conditioned on the configured queries, participating publishers, and Google's fact-check index. It is not a census of misinformation in Indonesia.
