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

[[ -f docs/index.html ]] || {
  echo "error: missing docs/index.html" >&2
  exit 1
}

mkdir -p "$DOCS_DIR" "$TMP_ROOT"
STAGING_DIR="$(mktemp -d "$TMP_ROOT/static-site-build.XXXXXX")"

cleanup() {
  rm -rf "$STAGING_DIR"
  rm -f \
    "$REPO_ROOT/notebooks/__marimo__/session/00_google_fact_check_tool_id.py.json" \
    "$REPO_ROOT/notebooks/__marimo__/session/01_google_fact_check_exploratory_data_analysis_id.py.json" \
    "$REPO_ROOT/notebooks/__marimo__/session/02_google_fact_check_generative_ai_hoax_distribution_id.py.json"
  rmdir "$REPO_ROOT/notebooks/__marimo__/session" 2>/dev/null || true
  rmdir "$REPO_ROOT/notebooks/__marimo__" 2>/dev/null || true
}
trap cleanup EXIT

NOTEBOOKS=(
  notebooks/00_google_fact_check_tool_id.py
  notebooks/01_google_fact_check_exploratory_data_analysis_id.py
  notebooks/02_google_fact_check_generative_ai_hoax_distribution_id.py
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
  generative-ai.html
  generative-ai-with-code.html
)

for output in "${OUTPUTS[@]}"; do
  [[ -s "$STAGING_DIR/$output" ]] || {
    echo "error: build produced an empty $output" >&2
    exit 1
  }
done

for output in "${OUTPUTS[@]}"; do
  mv -f "$STAGING_DIR/$output" "$DOCS_DIR/$output"
done

printf 'Static site built in %s\n' "$DOCS_DIR"
