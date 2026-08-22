import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def introduce_notebook(mo):
    mo.md("""
    # Google Fact Check Tools — data extraction

    Reproducible, keyword-driven extraction of Indonesian fact-checks for the
    companion analysis notebook.

    `data/google_fact_check_tool_id.parquet` is the source of truth. The notebook
    reads it on load and only calls the API when that file is missing or the
    refresh button is pressed.

    **What the API can give.** The Indonesian index holds roughly 7.6k unique
    claim reviews from 11 publishers, so a fixed 1,000-samples-per-keyword
    target is out of reach — only `video` clears it. Each keyword is extracted
    to exhaustion instead, and section 5 explains how.
    """)
    return


@app.cell(hide_code=True)
def explain_setup(mo):
    mo.md("""
    ## 1. Setup

    Everything environmental is resolved here: packages, the `.env`
    configuration, the two API endpoints, and where the extract lives.
    Nothing on this path touches the network — credentials are only requested
    later, and only if the extract has to be rebuilt.

    Copy `.env.example` to `.env` before the first run. A missing key raises
    `KeyError` immediately rather than silently pointing at the wrong project.
    """)
    return


@app.cell
def setup_environment():
    import os
    import time

    import google.auth
    import google.auth.transport.requests
    import marimo as mo
    import polars as pl
    import requests
    import urllib3
    from dotenv import load_dotenv

    REPO_ROOT = mo.notebook_dir().parent
    load_dotenv(REPO_ROOT / ".env")

    # Deployment-specific settings live in .env; see .env.example.
    PROJECT_ID = os.environ["PROJECT_ID"]
    KEY_DISPLAY_NAME = os.environ["KEY_DISPLAY_NAME"]

    API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    APIKEYS_API = "https://apikeys.googleapis.com/v2"
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    EXTRACT_PATH = REPO_ROOT / "data" / "google_fact_check_tool_id.parquet"
    mo.md(f"""
    | setting | value |
    |---|---|
    | Google Cloud project configured | yes |
    | API key display name configured | yes |
    | extract | `{EXTRACT_PATH.relative_to(REPO_ROOT)}` |
    | extract present | {"yes" if EXTRACT_PATH.exists() else "no — the API will be called"} |
    | EDA mode | offline, deterministic, recomputed in memory |
    """)
    return (
        API,
        APIKEYS_API,
        EXTRACT_PATH,
        KEY_DISPLAY_NAME,
        PROJECT_ID,
        SCOPES,
        google,
        mo,
        os,
        pl,
        requests,
        time,
        urllib3,
    )


@app.cell(hide_code=True)
def explain_keyword_vocabulary(mo):
    mo.md("""
    ## 2. Keywords

    The search vocabulary. Every keyword below returned Indonesian claims when
    probed against the API, grouped by theme so gaps are easy to spot. These
    drive the whole extract: one `query` search per keyword.
    """)
    return


@app.cell
def define_keyword_vocabulary():
    # Indonesian hoax keywords, each verified to return claims for languageCode="id".
    KEYWORDS = [
        # Politics and elections
        "pemilu", "kpu", "jokowi", "prabowo", "gibran", "anies", "ganjar",
        "megawati", "luhut", "sri mulyani", "presiden", "menteri", "demo",
        "ijazah",
        # Government, money and scams
        "korupsi", "kpk", "bansos", "subsidi", "pajak", "rupiah", "bank",
        "uang", "gaji", "penipuan", "undian", "lowongan kerja", "kartu",
        "beras", "minyak goreng", "sembako",
        # Health
        "vaksin", "covid", "imunisasi", "dokter", "rumah sakit", "obat",
        "kanker", "diabetes", "stunting", "bpjs", "bpom", "kemenkes",
        "telur", "susu", "gula", "mie instan", "kopi",
        # Disaster, environment and energy
        "gempa", "banjir", "tsunami", "erupsi", "gunung", "kebakaran",
        "cuaca", "hujan", "listrik", "pertamina", "tambang", "nikel",
        # Security, crime and society
        "polisi", "tni", "narkoba", "penculikan", "sekolah", "guru",
        "mahasiswa", "artis", "ular", "harimau", "buaya", "kiamat",
        # Religion and identity
        "islam", "masjid", "gereja", "ramadan", "haji", "natal", "mudik",
        "nu", "muhammadiyah",
        # Technology and media manipulation
        "hoaks", "video", "foto", "deepfake", "ai", "chip", "5g",
        # Transport and infrastructure
        "kereta", "pesawat", "garuda", "tol", "ojek", "nusantara", "ibu kota",
        # Places
        "jakarta", "surabaya", "bandung", "medan", "yogyakarta", "makassar",
        "bali", "aceh", "papua", "kalimantan", "sumatera",
        # International angles
        "china", "israel", "palestina", "ukraina", "amerika", "malaysia",
        "singapura",
    ]
    return (KEYWORDS,)


@app.cell(hide_code=True)
def explain_http_plumbing(mo):
    mo.md("""
    ## 3. HTTP plumbing

    `HTTP` is a pooled session with retries. This matters more than it looks:
    opening a fresh TLS connection per request made long sweeps fail
    intermittently with `SSLEOFError`, and connection reuse removed it.

    `read_json` turns a failed response into an error carrying the server's own
    message, and `authorized_session` attaches an Application Default
    Credentials token for the API Keys service.
    """)
    return


@app.cell
def configure_http_client(
    APIKEYS_API,
    PROJECT_ID,
    SCOPES,
    google,
    requests,
    time,
    urllib3,
):
    HTTP = requests.Session()
    HTTP.mount("https://", requests.adapters.HTTPAdapter(
        max_retries=urllib3.util.Retry(total=4, backoff_factor=0.5,
                                       status_forcelist=[429, 500, 502, 503, 504],
                                       allowed_methods=["GET"])))


    def read_json(response: requests.Response) -> dict:
        """Return the JSON body, raising with the server message on failure."""
        if not response.ok:
            raise RuntimeError(f"{response.request.method} {response.url} -> "
                               f"{response.status_code}: {response.text}")
        return response.json()


    def authorized_session() -> requests.Session:
        """Session carrying an Application Default Credentials bearer token."""
        credentials, _ = google.auth.default(scopes=SCOPES)
        credentials.refresh(google.auth.transport.requests.Request())

        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {credentials.token}",
            "x-goog-user-project": PROJECT_ID,
        })
        return session


    def wait_for_operation(session: requests.Session, operation: dict,
                           timeout_s: int = 60) -> dict:
        """Poll a long-running API Keys operation until it reports a result."""
        deadline = time.monotonic() + timeout_s
        while not operation.get("done"):
            if time.monotonic() > deadline:
                raise TimeoutError(f"operation {operation.get('name')} did not finish "
                                   f"within {timeout_s}s")
            time.sleep(1)
            operation = read_json(session.get(f"{APIKEYS_API}/{operation['name']}"))

        if "error" in operation:
            raise RuntimeError(f"operation failed: {operation['error']}")
        return operation["response"]

    return HTTP, authorized_session, read_json, wait_for_operation


@app.cell(hide_code=True)
def explain_api_key_management(mo):
    mo.md("""
    ## 4. API key from ADC

    `claims:search` rejects OAuth bearer tokens with `400 INVALID_ARGUMENT` — it
    only accepts an API key. So ADC is used against the **API Keys API** to mint
    one, restricted to `factchecktools.googleapis.com`.

    The call is idempotent: an existing key named `KEY_DISPLAY_NAME` is reused,
    so repeated runs never pile up duplicate keys.
    """)
    return


@app.cell
def define_api_key_manager(
    APIKEYS_API,
    KEY_DISPLAY_NAME,
    PROJECT_ID,
    authorized_session,
    read_json,
    wait_for_operation,
):
    def ensure_api_key() -> str:
        """Return the Fact Check Tools API key string, minted through ADC.

        Reuses the key named KEY_DISPLAY_NAME in PROJECT_ID when it already
        exists, so repeated runs never create duplicate keys.
        """
        session = authorized_session()
        parent = f"{APIKEYS_API}/projects/{PROJECT_ID}/locations/global"

        existing = read_json(session.get(f"{parent}/keys", params={"pageSize": 300}))
        for key in existing.get("keys", []):
            if key.get("displayName") == KEY_DISPLAY_NAME:
                return read_json(session.get(f"{APIKEYS_API}/{key['name']}/keyString"))["keyString"]

        operation = read_json(session.post(f"{parent}/keys", json={
            "displayName": KEY_DISPLAY_NAME,
            "restrictions": {"apiTargets": [{"service": "factchecktools.googleapis.com"}]},
        }))
        created = wait_for_operation(session, operation)
        return read_json(session.get(f"{APIKEYS_API}/{created['name']}/keyString"))["keyString"]

    return (ensure_api_key,)


@app.cell(hide_code=True)
def explain_claim_search(mo):
    mo.md("""
    ## 5. Searching claims

    `search_claims` is one keyword search that follows `nextPageToken`.

    `fetch_all_claims` exists because the two paging strategies disagree: token
    paging and offset paging truncate at different points, and each returns
    claims the other misses. Both are walked and merged, which is how `video`
    goes from 988 to just over 1,000 rows.
    """)
    return


@app.cell
def define_claim_search(API, HTTP, read_json):
    def claim_key(claim: dict) -> tuple[str, str]:
        """Stable identity for a claim: its text plus its first review URL."""
        first_review = (claim.get("claimReview") or [{}])[0]
        return (claim.get("text") or "", first_review.get("url") or "")


    def search_claims(
        api_key: str,
        query: str,
        language_code: str | None = "id",
        page_size: int = 100,
        max_pages: int = 40,
        offset: int | None = None,
    ) -> list[dict]:
        """Return claims matching a keyword search, following `nextPageToken`.

        Args:
            api_key: Key string from `ensure_api_key`.
            query: Search keyword, e.g. "vaksin".
            language_code: BCP-47 review language; None returns every language.
            page_size: Claims per page; the API caps this at 100.
            max_pages: Hard stop on pages fetched, keeping a run bounded.
            offset: Optional start position, an alternative to token paging.
        """
        params = {"key": api_key, "query": query, "pageSize": page_size}
        if language_code:
            params["languageCode"] = language_code
        if offset is not None:
            params["offset"] = offset

        claims: list[dict] = []
        page_token: str | None = None
        for _ in range(max_pages):
            if page_token:
                params["pageToken"] = page_token

            payload = read_json(HTTP.get(API, params=params, timeout=30))
            claims.extend(payload.get("claims", []))

            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        return claims


    def fetch_all_claims(
        api_key: str,
        query: str,
        language_code: str | None = "id",
        page_size: int = 100,
        max_offset: int = 6000,
    ) -> list[dict]:
        """Return every claim the API will surface for one keyword.

        Token paging and offset paging each truncate at different points, so both
        are walked and merged. The offset walk stops after two consecutive empty
        pages, which is where the result set genuinely ends.
        """
        found = {claim_key(c): c for c
                 in search_claims(api_key, query, language_code, page_size)}

        offset, empty_pages = 0, 0
        while offset < max_offset:
            page = search_claims(api_key, query, language_code, page_size,
                                 max_pages=1, offset=offset)
            if page:
                empty_pages = 0
                found.update({claim_key(c): c for c in page})
            else:
                empty_pages += 1
                if empty_pages >= 2:
                    break
            offset += page_size

        return list(found.values())

    return (fetch_all_claims,)


@app.cell(hide_code=True)
def explain_extract_pipeline(mo):
    mo.md("""
    ## 6. Building the extract

    Claims are flattened to one row per Indonesian review, de-duplicated and
    sorted so a rebuild is reproducible, then written atomically via a temp file
    and `os.replace` — an interrupted refresh cannot corrupt the existing file.

    `load_or_extract` enforces the cache-first contract: read the parquet, and
    only call the API when it is missing or a refresh is explicitly requested.
    """)
    return


@app.cell
def define_extract_pipeline(
    EXTRACT_PATH,
    ensure_api_key,
    fetch_all_claims,
    os,
    pl,
):
    CLAIM_SCHEMA = {
        "keyword": pl.Utf8,
        "claim_text": pl.Utf8,
        "claimant": pl.Utf8,
        "claim_date": pl.Utf8,
        "publisher_name": pl.Utf8,
        "publisher_site": pl.Utf8,
        "review_url": pl.Utf8,
        "review_title": pl.Utf8,
        "textual_rating": pl.Utf8,
        "review_date": pl.Utf8,
        "language_code": pl.Utf8,
    }


    def flatten_claims(keyword: str, claims: list[dict]) -> list[dict]:
        """Expand claims into one row per Indonesian review of that claim."""
        rows = []
        for claim in claims:
            for review in claim.get("claimReview", []):
                if review.get("languageCode") != "id":
                    continue
                publisher = review.get("publisher") or {}
                rows.append({
                    "keyword": keyword,
                    "claim_text": claim.get("text"),
                    "claimant": claim.get("claimant"),
                    "claim_date": claim.get("claimDate"),
                    "publisher_name": publisher.get("name"),
                    "publisher_site": publisher.get("site"),
                    "review_url": review.get("url"),
                    "review_title": review.get("title"),
                    "textual_rating": review.get("textualRating"),
                    "review_date": review.get("reviewDate"),
                    "language_code": review.get("languageCode"),
                })
        return rows


    def extract_keywords(api_key: str, keywords: list[str]) -> pl.DataFrame:
        """Extract every Indonesian claim the API returns for each keyword.

        Each keyword is walked with both paging strategies, so this collects
        the maximum the API exposes rather than a fixed sample size.
        Rows are de-duplicated and sorted, so a re-run yields the same table.
        """
        rows = []
        for keyword in keywords:
            claims = fetch_all_claims(api_key, keyword)
            rows.extend(flatten_claims(keyword, claims))

        return (pl.DataFrame(rows, schema=CLAIM_SCHEMA)
                .unique(subset=["keyword", "review_url", "claim_text"])
                .sort(["keyword", "review_url", "claim_text"]))


    def write_extract(frame: pl.DataFrame, path=EXTRACT_PATH) -> str:
        """Write the extract to parquet, replacing the file only on success."""
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(".parquet.tmp")
        frame.write_parquet(staging)
        os.replace(staging, path)
        return str(path)


    def load_or_extract(keywords: list[str], refresh: bool = False,
                        path=EXTRACT_PATH) -> tuple[pl.DataFrame, str]:
        """Return the extract and where it came from.

        The parquet is the source of truth: the API is only called when the file
        is missing or `refresh` is True. Nothing is fetched, and no API key is
        minted, while a cached extract exists.
        """
        if path.exists() and not refresh:
            return pl.read_parquet(path), f"parquet ({path.name})"

        frame = extract_keywords(ensure_api_key(), keywords)
        write_extract(frame, path)
        return frame, "Google Fact Check Tools API"

    return (load_or_extract,)


@app.cell(hide_code=True)
def explain_refresh_control(mo):
    mo.md("""
    ## 7. Refresh control

    In a live marimo session, this button is the only action that spends API quota:
    it re-walks every keyword and atomically overwrites the Parquet. A static HTML
    export has no Python kernel, so the control is intentionally labeled
    **live session only**; the exported report and its eager downloads remain fully
    usable without it.
    """)
    return


@app.cell
def render_refresh_control(mo):
    refresh_button = mo.ui.run_button(label="Re-extract from API (live session only; overwrites parquet)")
    refresh_button
    return (refresh_button,)

@app.cell
def render_extract_result(
    EXTRACT_PATH,
    KEYWORDS,
    load_or_extract,
    mo,
    pl,
    refresh_button,
):
    extract_frame, extract_source = load_or_extract(
        KEYWORDS, refresh=refresh_button.value
    )
    extract_summary = (
        extract_frame
        .select(
            pl.len().alias("keyword_match_rows"),
            pl.col("keyword").n_unique().alias("keywords"),
            pl.col("review_url").n_unique().alias("distinct_review_urls"),
            pl.col("publisher_site").n_unique().alias("publisher_sites"),
        )
    )
    raw_csv_bytes = extract_frame.write_csv().encode("utf-8")
    raw_parquet_bytes = EXTRACT_PATH.read_bytes()
    csv_download = mo.download(
        data=raw_csv_bytes,
        filename="google_fact_check_tool_id_raw.csv",
        mimetype="text/csv",
        label=f"Download raw CSV ({len(raw_csv_bytes) / 1024 / 1024:.2f} MiB)",
    )
    parquet_download = mo.download(
        data=raw_parquet_bytes,
        filename=EXTRACT_PATH.name,
        mimetype="application/vnd.apache.parquet",
        label=f"Download raw Parquet ({len(raw_parquet_bytes) / 1024:.0f} KiB)",
    )
    mo.vstack([
        mo.callout(
            mo.md(
                f"Loaded **{extract_frame.height:,} keyword-match rows** from "
                f"**{extract_source}** into `{EXTRACT_PATH.relative_to(mo.notebook_dir().parent)}`."
            ),
            kind="success",
            title="Extract ready",
        ),
        mo.ui.table(
            extract_summary,
            selection=None,
            pagination=False,
            show_column_summaries=False,
        ),
        mo.md("### Raw data downloads"),
        mo.hstack(
            [csv_download, parquet_download],
            justify="start",
            wrap=True,
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
