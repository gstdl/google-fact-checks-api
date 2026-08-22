#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
TMP_ROOT="$REPO_ROOT/tmp"

command -v uv >/dev/null 2>&1 || {
  echo "error: uv is required to build the static site" >&2
  exit 1
}

cd "$REPO_ROOT"

[[ -f data/google_fact_check_tool_id.parquet ]] || {
  echo "error: missing data/google_fact_check_tool_id.parquet" >&2
  exit 1
}

[[ -f data/google_fact_check_ner_entities_id.parquet ]] || {
  echo "error: missing data/google_fact_check_ner_entities_id.parquet" >&2
  echo "refresh it with: uv run --extra ner python scripts/refresh-ner-entities.py" >&2
  exit 1
}

[[ -f docs/index.html ]] || {
  echo "error: missing docs/index.html" >&2
  exit 1
}

mkdir -p "$DOCS_DIR" "$TMP_ROOT"
STAGING_DIR="$(mktemp -d "$TMP_ROOT/static-site-build.XXXXXX")"
BACKUP_DIR="$STAGING_DIR/previous"
OUTPUTS=()
PUBLISH_STARTED=false
PUBLISH_COMPLETE=false

cleanup() {
  status=$?
  if [[ "$PUBLISH_STARTED" == true && "$PUBLISH_COMPLETE" != true ]]; then
    for output in "${OUTPUTS[@]}"; do
      if [[ -f "$BACKUP_DIR/$output" ]]; then
        cp -p "$BACKUP_DIR/$output" "$DOCS_DIR/$output"
      else
        rm -f "$DOCS_DIR/$output"
      fi
    done
  fi
  rm -rf "$STAGING_DIR"
  rm -f \
    "$REPO_ROOT/notebooks/__marimo__/session/00_google_fact_check_tool_id.py.json" \
    "$REPO_ROOT/notebooks/__marimo__/session/01_google_fact_check_exploratory_data_analysis_id.py.json" \
    "$REPO_ROOT/notebooks/__marimo__/session/02_google_fact_check_generative_ai_hoax_distribution_id.py.json" \
    "$REPO_ROOT/notebooks/__marimo__/session/03_google_fact_check_relational_network_analysis_id.py.json"
  rmdir "$REPO_ROOT/notebooks/__marimo__/session" 2>/dev/null || true
  rmdir "$REPO_ROOT/notebooks/__marimo__" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT

NOTEBOOKS=(
  notebooks/00_google_fact_check_tool_id.py
  notebooks/01_google_fact_check_exploratory_data_analysis_id.py
  notebooks/02_google_fact_check_generative_ai_hoax_distribution_id.py
  notebooks/03_google_fact_check_relational_network_analysis_id.py
)

uv run marimo check "${NOTEBOOKS[@]}"

uv run marimo export html \
  notebooks/00_google_fact_check_tool_id.py \
  --output "$STAGING_DIR/extraction.html" \
  --include-code \
  --force

uv run marimo export html \
  notebooks/01_google_fact_check_exploratory_data_analysis_id.py \
  --output "$STAGING_DIR/analysis.html" \
  --no-include-code \
  --force

uv run marimo export html \
  notebooks/01_google_fact_check_exploratory_data_analysis_id.py \
  --output "$STAGING_DIR/analysis-with-code.html" \
  --include-code \
  --force

uv run marimo export html \
  notebooks/03_google_fact_check_relational_network_analysis_id.py \
  --output "$STAGING_DIR/network-analysis.html" \
  --no-include-code \
  --force

uv run marimo export html \
  notebooks/03_google_fact_check_relational_network_analysis_id.py \
  --output "$STAGING_DIR/network-analysis-with-code.html" \
  --include-code \
  --force

uv run marimo export html \
  notebooks/02_google_fact_check_generative_ai_hoax_distribution_id.py \
  --output "$STAGING_DIR/generative-ai.html" \
  --no-include-code \
  --force

uv run marimo export html \
  notebooks/02_google_fact_check_generative_ai_hoax_distribution_id.py \
  --output "$STAGING_DIR/generative-ai-with-code.html" \
  --include-code \
  --force

OUTPUTS=(
  extraction.html
  analysis.html
  analysis-with-code.html
  network-analysis.html
  network-analysis-with-code.html
  generative-ai.html
  generative-ai-with-code.html
)

for output in "${OUTPUTS[@]}"; do
  [[ -s "$STAGING_DIR/$output" ]] || {
    echo "error: build produced an empty $output" >&2
    exit 1
  }
done

mkdir -p "$BACKUP_DIR"
for output in "${OUTPUTS[@]}"; do
  if [[ -f "$DOCS_DIR/$output" ]]; then
    cp -p "$DOCS_DIR/$output" "$BACKUP_DIR/$output"
  fi
done

PUBLISH_STARTED=true
for output in "${OUTPUTS[@]}"; do
  mv -f "$STAGING_DIR/$output" "$DOCS_DIR/$output"
done
PUBLISH_COMPLETE=true

printf 'Static site built in %s\n' "$DOCS_DIR"
