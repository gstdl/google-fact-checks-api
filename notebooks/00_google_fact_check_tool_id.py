import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def introduce_notebook(mo):
    mo.md("""
    # Google Fact Check Tools — data extraction

    Reproducible, keyword-driven extraction of Indonesian fact-checks for the
    companion exploratory data analysis notebook.

    `data/google_fact_check_tool_id.parquet` remains the canonical source of
    truth for every analysis notebook. This notebook reads it on load and only
    calls the API when that file is missing or the refresh button is pressed.
    A successful live refresh first publishes an immutable snapshot containing
    raw response pages, provenance metadata, and the flattened Parquet.

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
    import datetime
    import fcntl
    import gzip
    import hashlib
    import io
    import json
    import os
    import re
    import shutil
    import tempfile
    import time
    import warnings

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
    SNAPSHOT_ROOT = REPO_ROOT / "data" / "google_fact_check_tool_id_snapshots"
    TMP_ROOT = REPO_ROOT / "tmp"
    NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "00_google_fact_check_tool_id.py"
    mo.md(f"""
    | setting | value |
    |---|---|
    | Google Cloud project configured | yes |
    | API key display name configured | yes |
    | canonical extract | `{EXTRACT_PATH.relative_to(REPO_ROOT)}` |
    | extract present | {"yes" if EXTRACT_PATH.exists() else "no — the API will be called"} |
    | immutable snapshots | `{SNAPSHOT_ROOT.relative_to(REPO_ROOT)}/` |
    | cached execution | offline and read-only |
    """)
    return (
        API,
        APIKEYS_API,
        EXTRACT_PATH,
        KEY_DISPLAY_NAME,
        NOTEBOOK_PATH,
        PROJECT_ID,
        REPO_ROOT,
        SCOPES,
        SNAPSHOT_ROOT,
        TMP_ROOT,
        datetime,
        fcntl,
        google,
        gzip,
        hashlib,
        io,
        json,
        mo,
        os,
        pl,
        re,
        requests,
        shutil,
        tempfile,
        time,
        urllib3,
        warnings,
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
            # Do not include response.url: API-key query parameters belong there.
            method = getattr(response.request, "method", "request")
            raise RuntimeError(f"{method} response -> {response.status_code}: "
                               f"{response.text}")
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

    `search_claims` follows `nextPageToken`; `search_claims_by_offset` walks
    numeric offsets until two consecutive pages are empty. The strategies expose
    different records, so `fetch_all_claims` preserves and merges both.

    Every successful response is written under `./tmp` immediately as
    deterministic gzip (`mtime=0`). Audit records contain checksums, byte counts,
    and sanitized request parameters: API keys and raw request page tokens are
    never copied into provenance metadata. A walk that reaches either safety cap
    is reported as a warning, not mislabeled as exhaustion.
    """)
    return


@app.cell
def define_claim_search(
    API,
    HTTP,
    gzip,
    hashlib,
    io,
    re,
    read_json,
    warnings,
):
    def sha256_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()


    def deterministic_gzip(value: bytes) -> bytes:
        """Compress bytes with a stable header and timestamp."""
        buffer = io.BytesIO()
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=buffer,
            compresslevel=9,
            mtime=0,
        ) as gzip_file:
            gzip_file.write(value)
        return buffer.getvalue()


    def keyword_directory(keyword_index: int, query: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", query.casefold()).strip("-")
        return f"{keyword_index:03d}-{slug or 'query'}"


    def claim_key(claim: dict) -> tuple[str, str]:
        """Stable identity for a claim: its text plus its first review URL."""
        first_review = (claim.get("claimReview") or [{}])[0]
        return (claim.get("text") or "", first_review.get("url") or "")


    def _fetch_claim_page(
        api_key: str,
        query: str,
        capture_root,
        keyword_index: int,
        strategy: str,
        page_index: int,
        language_code: str | None,
        page_size: int,
        offset: int | None = None,
        page_token: str | None = None,
    ) -> tuple[list[dict], str | None, dict]:
        """Fetch, archive, and audit one successful claims:search page."""
        params = {"key": api_key, "query": query, "pageSize": page_size}
        if language_code:
            params["languageCode"] = language_code
        if offset is not None:
            params["offset"] = offset
        if page_token:
            params["pageToken"] = page_token

        response = HTTP.get(API, params=params, timeout=30)
        payload = read_json(response)
        response_bytes = response.content
        compressed_bytes = deterministic_gzip(response_bytes)

        keyword_path = keyword_directory(keyword_index, query)
        if strategy == "token":
            filename = f"token-{page_index:04d}.json.gz"
        else:
            filename = f"offset-{offset:06d}.json.gz"
        response_file = f"raw/{keyword_path}/{filename}"
        destination = capture_root / response_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(compressed_bytes)

        sanitized_params = {
            "query": query,
            "pageSize": page_size,
            "pageTokenPresent": bool(page_token),
        }
        if language_code:
            sanitized_params["languageCode"] = language_code
        if offset is not None:
            sanitized_params["offset"] = offset
        if page_token:
            sanitized_params["pageTokenSha256"] = sha256_bytes(
                page_token.encode("utf-8")
            )

        next_token = payload.get("nextPageToken")
        claims = payload.get("claims", [])
        page_audit = {
            "keyword": query,
            "keyword_index": keyword_index,
            "strategy": strategy,
            "page_index": page_index,
            "offset": offset,
            "request_parameters": sanitized_params,
            "response_file": response_file,
            "response_sha256": sha256_bytes(response_bytes),
            "response_bytes": len(response_bytes),
            "compressed_sha256": sha256_bytes(compressed_bytes),
            "compressed_bytes": len(compressed_bytes),
            "claim_count": len(claims),
            "next_token_present": bool(next_token),
            "next_token_sha256": (
                sha256_bytes(next_token.encode("utf-8"))
                if next_token
                else None
            ),
        }
        return claims, next_token, page_audit


    def search_claims(
        api_key: str,
        query: str,
        capture_root,
        keyword_index: int,
        language_code: str | None = "id",
        page_size: int = 100,
        max_pages: int = 40,
    ) -> tuple[list[dict], dict]:
        """Walk token pagination and return claims plus a complete audit."""
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")

        claims: list[dict] = []
        pages: list[dict] = []
        page_token: str | None = None
        termination_reason = "max_pages"
        for page_index in range(1, max_pages + 1):
            page, next_token, page_audit = _fetch_claim_page(
                api_key=api_key,
                query=query,
                capture_root=capture_root,
                keyword_index=keyword_index,
                strategy="token",
                page_index=page_index,
                language_code=language_code,
                page_size=page_size,
                page_token=page_token,
            )
            claims.extend(page)
            pages.append(page_audit)
            page_token = next_token
            if not page_token:
                termination_reason = "exhausted"
                break

        return claims, {
            "strategy": "token",
            "page_count": len(pages),
            "termination_reason": termination_reason,
            "terminated_by_cap": termination_reason == "max_pages",
            "pages": pages,
        }


    def search_claims_by_offset(
        api_key: str,
        query: str,
        capture_root,
        keyword_index: int,
        language_code: str | None = "id",
        page_size: int = 100,
        max_offset: int = 6000,
    ) -> tuple[list[dict], dict]:
        """Walk offsets until two empty pages or the configured cap."""
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        if max_offset < 1:
            raise ValueError("max_offset must be at least 1")

        claims: list[dict] = []
        pages: list[dict] = []
        offset = 0
        empty_pages = 0
        termination_reason = "max_offset"
        while offset < max_offset:
            page, _, page_audit = _fetch_claim_page(
                api_key=api_key,
                query=query,
                capture_root=capture_root,
                keyword_index=keyword_index,
                strategy="offset",
                page_index=len(pages) + 1,
                language_code=language_code,
                page_size=page_size,
                offset=offset,
            )
            claims.extend(page)
            pages.append(page_audit)
            if page:
                empty_pages = 0
            else:
                empty_pages += 1
                if empty_pages == 2:
                    termination_reason = "two_empty_pages"
                    break
            offset += page_size

        return claims, {
            "strategy": "offset",
            "page_count": len(pages),
            "termination_reason": termination_reason,
            "terminated_by_cap": termination_reason == "max_offset",
            "pages": pages,
        }


    def fetch_all_claims(
        api_key: str,
        query: str,
        capture_root,
        keyword_index: int,
        language_code: str | None = "id",
        page_size: int = 100,
        max_pages: int = 40,
        max_offset: int = 6000,
    ) -> tuple[list[dict], dict]:
        """Merge token and offset results without changing claim semantics."""
        token_claims, token_audit = search_claims(
            api_key,
            query,
            capture_root,
            keyword_index,
            language_code,
            page_size,
            max_pages,
        )
        offset_claims, offset_audit = search_claims_by_offset(
            api_key,
            query,
            capture_root,
            keyword_index,
            language_code,
            page_size,
            max_offset,
        )

        found = {claim_key(claim): claim for claim in token_claims}
        found.update({claim_key(claim): claim for claim in offset_claims})

        cap_warnings = []
        if token_audit["terminated_by_cap"]:
            cap_warnings.append(
                f"{query!r}: token pagination reached max_pages={max_pages}"
            )
        if offset_audit["terminated_by_cap"]:
            cap_warnings.append(
                f"{query!r}: offset pagination reached max_offset={max_offset}"
            )
        for message in cap_warnings:
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        return list(found.values()), {
            "keyword": query,
            "keyword_index": keyword_index,
            "claims_from_token": len(token_claims),
            "claims_from_offset": len(offset_claims),
            "claims_after_strategy_merge": len(found),
            "token": token_audit,
            "offset": offset_audit,
            "cap_warnings": cap_warnings,
        }

    return deterministic_gzip, fetch_all_claims, sha256_bytes


@app.cell(hide_code=True)
def explain_extract_pipeline(mo):
    mo.md("""
    ## 6. Building the extract

    Claims are flattened to one row per Indonesian review, de-duplicated, and
    sorted under `./tmp`. The complete raw-page archive, metadata, and derived
    Parquet are checksum-validated before an immutable snapshot is atomically
    moved into `data/google_fact_check_tool_id_snapshots/`.

    Only after that snapshot exists does `os.replace` atomically publish the same
    bytes as the canonical Parquet. If replacement fails, the new snapshot is
    removed and the prior canonical file is retained. Cache-first execution only
    reads existing files: it makes no network call and creates no snapshot.
    """)
    return


@app.cell
def define_extract_pipeline(
    API,
    EXTRACT_PATH,
    NOTEBOOK_PATH,
    REPO_ROOT,
    SNAPSHOT_ROOT,
    TMP_ROOT,
    datetime,
    deterministic_gzip,
    ensure_api_key,
    fetch_all_claims,
    fcntl,
    gzip,
    json,
    os,
    pl,
    sha256_bytes,
    shutil,
    tempfile,
):
    PROVENANCE_SCHEMA_VERSION = "1.0"
    LANGUAGE_CODE = "id"
    PAGE_SIZE = 100
    MAX_TOKEN_PAGES = 40
    MAX_OFFSET = 6000
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


    def sha256_file(path) -> str:
        return sha256_bytes(path.read_bytes())


    def vocabulary_sha256(keywords: list[str]) -> str:
        serialized = json.dumps(
            keywords,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256_bytes(serialized)


    def utc_timestamp(value: datetime.datetime) -> str:
        return value.astimezone(datetime.timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )


    def flatten_claims(keyword: str, claims: list[dict]) -> list[dict]:
        """Expand claims into one row per Indonesian review of that claim."""
        rows = []
        for claim in claims:
            for review in claim.get("claimReview", []):
                if review.get("languageCode") != LANGUAGE_CODE:
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


    def extract_keywords(
        api_key: str,
        keywords: list[str],
        capture_root,
    ) -> tuple[pl.DataFrame, list[dict], list[dict], dict]:
        """Extract, flatten, de-duplicate, and return provenance components."""
        rows = []
        keyword_audits = []
        page_records = []
        claims_before_flattening = 0

        for keyword_index, keyword in enumerate(keywords, start=1):
            claims, audit = fetch_all_claims(
                api_key=api_key,
                query=keyword,
                capture_root=capture_root,
                keyword_index=keyword_index,
                language_code=LANGUAGE_CODE,
                page_size=PAGE_SIZE,
                max_pages=MAX_TOKEN_PAGES,
                max_offset=MAX_OFFSET,
            )
            claims_before_flattening += len(claims)
            rows.extend(flatten_claims(keyword, claims))
            page_records.extend(audit["token"]["pages"])
            page_records.extend(audit["offset"]["pages"])
            keyword_audits.append({
                "keyword": keyword,
                "keyword_index": keyword_index,
                "claims_from_token": audit["claims_from_token"],
                "claims_from_offset": audit["claims_from_offset"],
                "claims_after_strategy_merge": audit[
                    "claims_after_strategy_merge"
                ],
                "token_page_count": audit["token"]["page_count"],
                "token_termination_reason": audit["token"][
                    "termination_reason"
                ],
                "token_terminated_by_cap": audit["token"][
                    "terminated_by_cap"
                ],
                "offset_page_count": audit["offset"]["page_count"],
                "offset_termination_reason": audit["offset"][
                    "termination_reason"
                ],
                "offset_terminated_by_cap": audit["offset"][
                    "terminated_by_cap"
                ],
                "cap_warnings": audit["cap_warnings"],
            })

        before_deduplication = pl.DataFrame(rows, schema=CLAIM_SCHEMA)
        frame = (
            before_deduplication
            .unique(subset=["keyword", "review_url", "claim_text"])
            .sort(["keyword", "review_url", "claim_text"])
        )
        counts = {
            "claims_before_flattening": claims_before_flattening,
            "rows_before_deduplication": before_deduplication.height,
            "rows_after_deduplication": frame.height,
        }
        return frame, keyword_audits, page_records, counts


    def validate_snapshot(
        snapshot_directory,
        api_key=None,
        expected_metadata=None,
        path=EXTRACT_PATH,
        snapshot_root=SNAPSHOT_ROOT,
        require_current_source: bool = False,
    ) -> dict:
        """Validate the staged metadata artifact and every referenced byte."""
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(f"invalid extraction snapshot: {message}")

        metadata_path = snapshot_directory / "metadata.json"
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        if expected_metadata is not None:
            require(
                metadata == expected_metadata,
                "metadata file differs from the in-memory record",
            )

        require(
            metadata["provenance_schema_version"]
            == PROVENANCE_SCHEMA_VERSION,
            "unexpected provenance schema version",
        )
        require(
            vocabulary_sha256(metadata["query_vocabulary"]["ordered"])
            == metadata["query_vocabulary"]["sha256"],
            "query vocabulary checksum mismatch",
        )
        require(
            metadata["api"] == {
                "endpoint": API,
                "language_code": LANGUAGE_CODE,
                "page_size": PAGE_SIZE,
                "token_page_cap": MAX_TOKEN_PAGES,
                "offset_cap": MAX_OFFSET,
            },
            "API configuration mismatch",
        )
        require(
            metadata["notebook_source"]["path"]
            == NOTEBOOK_PATH.relative_to(REPO_ROOT).as_posix(),
            "notebook source path mismatch",
        )
        source_hash = metadata["notebook_source"]["sha256"]
        require(
            len(source_hash) == 64
            and all(character in "0123456789abcdef" for character in source_hash),
            "invalid notebook source checksum",
        )
        if require_current_source:
            require(
                source_hash == sha256_file(NOTEBOOK_PATH),
                "notebook source checksum mismatch",
            )

        started_text = metadata["extraction"]["started_at_utc"]
        completed_text = metadata["extraction"]["completed_at_utc"]
        require(
            started_text.endswith("Z") and completed_text.endswith("Z"),
            "extraction timestamps are not UTC",
        )
        started_at = datetime.datetime.fromisoformat(
            started_text.replace("Z", "+00:00")
        )
        completed_at = datetime.datetime.fromisoformat(
            completed_text.replace("Z", "+00:00")
        )
        require(
            started_at.utcoffset() == datetime.timedelta(0)
            and completed_at.utcoffset() == datetime.timedelta(0)
            and started_at <= completed_at,
            "invalid extraction timestamp interval",
        )

        vocabulary = metadata["query_vocabulary"]["ordered"]
        require(
            len(metadata["keyword_pagination"]) == len(vocabulary),
            "keyword audit count mismatch",
        )
        final_snapshot = snapshot_root / metadata["snapshot_identifier"]
        require(
            metadata["paths"] == {
                "snapshot_directory": final_snapshot.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "raw_directory": (final_snapshot / "raw").relative_to(
                    REPO_ROOT
                ).as_posix(),
                "metadata": (final_snapshot / "metadata.json").relative_to(
                    REPO_ROOT
                ).as_posix(),
                "flattened_parquet": (
                    final_snapshot / "flattened.parquet"
                ).relative_to(REPO_ROOT).as_posix(),
                "canonical_parquet": path.relative_to(REPO_ROOT).as_posix(),
            },
            "published path metadata mismatch",
        )
        if api_key:
            require(
                api_key.encode("utf-8") not in metadata_bytes,
                "API key found in metadata",
            )

        recorded_files = []
        walk_claims = {}
        allowed_parameters = {
            "query",
            "languageCode",
            "pageSize",
            "offset",
            "pageTokenPresent",
            "pageTokenSha256",
        }
        for page in metadata["pages"]:
            keyword_index = page["keyword_index"]
            require(
                1 <= keyword_index <= len(vocabulary),
                "page keyword index outside vocabulary",
            )
            require(
                page["keyword"] == vocabulary[keyword_index - 1],
                "page keyword does not match ordered vocabulary",
            )
            require(
                page["strategy"] in {"token", "offset"},
                "invalid page strategy",
            )
            parameters = page["request_parameters"]
            require(
                set(parameters).issubset(allowed_parameters),
                "unsanitized request parameters",
            )
            require(
                parameters["query"] == page["keyword"]
                and parameters["languageCode"] == LANGUAGE_CODE
                and parameters["pageSize"] == PAGE_SIZE,
                "sanitized request parameters mismatch",
            )
            if page["strategy"] == "token":
                require(
                    page["offset"] is None and "offset" not in parameters,
                    "token page records an offset",
                )
            else:
                require(
                    page["offset"] == parameters["offset"]
                    and parameters["pageTokenPresent"] is False
                    and "pageTokenSha256" not in parameters,
                    "offset page parameters mismatch",
                )
            require("key" not in parameters, "API key parameter recorded")
            require("pageToken" not in parameters, "raw page token recorded")

            response_file = page["response_file"]
            require(
                response_file.startswith("raw/")
                and response_file.endswith(".json.gz"),
                "invalid raw response path",
            )
            response_path = snapshot_directory / response_file
            require(
                response_path.resolve().is_relative_to(
                    snapshot_directory.resolve()
                ),
                "raw response path escapes snapshot",
            )
            require(response_path.is_file(), f"missing {response_file}")
            compressed = response_path.read_bytes()
            require(
                sha256_bytes(compressed) == page["compressed_sha256"],
                f"compressed checksum mismatch for {response_file}",
            )
            require(
                len(compressed) == page["compressed_bytes"],
                f"compressed byte count mismatch for {response_file}",
            )
            response_bytes = gzip.decompress(compressed)
            require(
                compressed == deterministic_gzip(response_bytes),
                f"non-deterministic gzip encoding for {response_file}",
            )
            require(
                sha256_bytes(response_bytes) == page["response_sha256"],
                f"response checksum mismatch for {response_file}",
            )
            require(
                len(response_bytes) == page["response_bytes"],
                f"response byte count mismatch for {response_file}",
            )
            if api_key:
                require(
                    api_key.encode("utf-8") not in response_bytes,
                    f"API key echoed in {response_file}",
                )
            payload = json.loads(response_bytes)
            require(isinstance(payload, dict), "response body is not an object")
            page_claims = payload.get("claims", [])
            require(
                isinstance(page_claims, list),
                f"claims value is not a list in {response_file}",
            )
            require(
                len(page_claims) == page["claim_count"],
                f"claim count mismatch for {response_file}",
            )
            next_token = payload.get("nextPageToken")
            require(
                bool(next_token) == page["next_token_present"],
                f"next-token presence mismatch for {response_file}",
            )
            require(
                (
                    sha256_bytes(next_token.encode("utf-8"))
                    if next_token
                    else None
                ) == page["next_token_sha256"],
                f"next-token checksum mismatch for {response_file}",
            )
            walk_claims.setdefault(
                (keyword_index, page["strategy"]), []
            ).extend(page_claims)
            recorded_files.append(response_path.relative_to(snapshot_directory))

        actual_files = sorted(
            path.relative_to(snapshot_directory)
            for path in snapshot_directory.glob("raw/**/*.json.gz")
        )
        require(
            len(recorded_files) == len(set(recorded_files)),
            "duplicate raw response path",
        )
        require(
            sorted(recorded_files) == actual_files,
            "raw response file inventory mismatch",
        )

        def archived_claim_key(claim: dict) -> tuple[str, str]:
            first_review = (claim.get("claimReview") or [{}])[0]
            return (
                claim.get("text") or "",
                first_review.get("url") or "",
            )

        rebuilt_rows = []
        rebuilt_claim_count = 0
        for expected_index, (query, keyword) in enumerate(
            zip(vocabulary, metadata["keyword_pagination"]),
            start=1,
        ):
            require(
                keyword["keyword_index"] == expected_index
                and keyword["keyword"] == query,
                "keyword audit does not match ordered vocabulary",
            )
            token_pages = [
                page for page in metadata["pages"]
                if page["keyword_index"] == keyword["keyword_index"]
                and page["strategy"] == "token"
            ]
            offset_pages = [
                page for page in metadata["pages"]
                if page["keyword_index"] == keyword["keyword_index"]
                and page["strategy"] == "offset"
            ]
            require(
                len(token_pages) == keyword["token_page_count"],
                "token page count mismatch",
            )
            require(
                len(offset_pages) == keyword["offset_page_count"],
                "offset page count mismatch",
            )
            require(
                [page["page_index"] for page in token_pages]
                == list(range(1, len(token_pages) + 1)),
                "token page indexes are not contiguous",
            )
            require(
                [page["page_index"] for page in offset_pages]
                == list(range(1, len(offset_pages) + 1)),
                "offset page indexes are not contiguous",
            )
            require(
                [page["offset"] for page in offset_pages]
                == [PAGE_SIZE * index for index in range(len(offset_pages))],
                "offset indexes are not contiguous",
            )
            require(
                keyword["token_termination_reason"]
                in {"exhausted", "max_pages"},
                "invalid token termination reason",
            )
            require(
                keyword["offset_termination_reason"]
                in {"two_empty_pages", "max_offset"},
                "invalid offset termination reason",
            )
            require(
                keyword["token_terminated_by_cap"]
                == (keyword["token_termination_reason"] == "max_pages"),
                "token cap flag mismatch",
            )
            require(
                keyword["offset_terminated_by_cap"]
                == (keyword["offset_termination_reason"] == "max_offset"),
                "offset cap flag mismatch",
            )

            previous_token_hash = None
            for page_index, page in enumerate(token_pages):
                parameters = page["request_parameters"]
                if page_index == 0:
                    require(
                        parameters["pageTokenPresent"] is False,
                        "first token request unexpectedly has a token",
                    )
                else:
                    require(
                        parameters["pageTokenPresent"] is True,
                        "continued token request lacks a token",
                    )
                    require(
                        parameters["pageTokenSha256"] == previous_token_hash,
                        "page-token continuity checksum mismatch",
                    )
                previous_token_hash = page["next_token_sha256"]
            require(
                bool(token_pages[-1]["next_token_present"])
                == keyword["token_terminated_by_cap"],
                "token termination does not match final response",
            )
            if keyword["token_termination_reason"] == "max_pages":
                require(
                    len(token_pages) == MAX_TOKEN_PAGES,
                    "token cap termination has the wrong page count",
                )
            if keyword["offset_termination_reason"] == "two_empty_pages":
                require(
                    len(offset_pages) >= 2
                    and offset_pages[-2]["claim_count"] == 0
                    and offset_pages[-1]["claim_count"] == 0,
                    "offset exhaustion lacks two empty pages",
                )
            else:
                require(
                    offset_pages[-1]["offset"] + PAGE_SIZE >= MAX_OFFSET,
                    "offset cap termination occurs before the cap",
                )

            token_claims = walk_claims[(expected_index, "token")]
            offset_claims = walk_claims[(expected_index, "offset")]
            require(
                len(token_claims) == keyword["claims_from_token"],
                "token claim total mismatch",
            )
            require(
                len(offset_claims) == keyword["claims_from_offset"],
                "offset claim total mismatch",
            )

            merged_claims = {
                archived_claim_key(claim): claim for claim in token_claims
            }
            merged_claims.update({
                archived_claim_key(claim): claim for claim in offset_claims
            })
            require(
                len(merged_claims) == keyword["claims_after_strategy_merge"],
                "strategy-merged claim total mismatch",
            )
            rebuilt_claim_count += len(merged_claims)
            rebuilt_rows.extend(flatten_claims(query, list(merged_claims.values())))

            expected_keyword_warnings = []
            if keyword["token_terminated_by_cap"]:
                expected_keyword_warnings.append(
                    f"{query!r}: token pagination reached "
                    f"max_pages={MAX_TOKEN_PAGES}"
                )
            if keyword["offset_terminated_by_cap"]:
                expected_keyword_warnings.append(
                    f"{query!r}: offset pagination reached "
                    f"max_offset={MAX_OFFSET}"
                )
            require(
                keyword["cap_warnings"] == expected_keyword_warnings,
                "keyword cap warning mismatch",
            )

        expected_cap_warnings = [
            warning
            for keyword in metadata["keyword_pagination"]
            for warning in keyword["cap_warnings"]
        ]
        require(
            metadata["termination"]["cap_warnings"]
            == expected_cap_warnings,
            "cap warning inventory mismatch",
        )
        require(
            metadata["termination"]["max_pages"]
            == any(
                keyword["token_terminated_by_cap"]
                for keyword in metadata["keyword_pagination"]
            ),
            "top-level max_pages flag mismatch",
        )
        require(
            metadata["termination"]["max_offset"]
            == any(
                keyword["offset_terminated_by_cap"]
                for keyword in metadata["keyword_pagination"]
            ),
            "top-level max_offset flag mismatch",
        )

        rebuilt_before_deduplication = pl.DataFrame(
            rebuilt_rows,
            schema=CLAIM_SCHEMA,
        )
        rebuilt_frame = (
            rebuilt_before_deduplication
            .unique(subset=["keyword", "review_url", "claim_text"])
            .sort(["keyword", "review_url", "claim_text"])
        )
        require(
            metadata["counts"] == {
                "claims_before_flattening": rebuilt_claim_count,
                "rows_before_deduplication": (
                    rebuilt_before_deduplication.height
                ),
                "rows_after_deduplication": rebuilt_frame.height,
            },
            "flattening or deduplication counts mismatch",
        )

        parquet_path = snapshot_directory / "flattened.parquet"
        require(parquet_path.is_file(), "missing flattened.parquet")
        require(
            sha256_file(parquet_path) == metadata["output"]["parquet_sha256"],
            "Parquet checksum mismatch",
        )
        require(
            parquet_path.stat().st_size == metadata["output"]["parquet_bytes"],
            "Parquet byte count mismatch",
        )
        frame = pl.read_parquet(parquet_path)
        require(
            frame.height == metadata["output"]["row_count"],
            "Parquet row count mismatch",
        )
        require(
            frame.height == metadata["counts"]["rows_after_deduplication"],
            "post-deduplication count mismatch",
        )
        require(
            frame.equals(rebuilt_frame),
            "flattened Parquet does not match archived responses",
        )
        require(
            metadata["snapshot_identifier"].endswith(
                f"-{metadata['output']['parquet_sha256'][:12]}"
            ),
            "snapshot identifier hash prefix mismatch",
        )
        final_schema = [
            {"name": name, "dtype": str(dtype)}
            for name, dtype in frame.schema.items()
        ]
        require(
            final_schema == metadata["output"]["schema"],
            "Parquet schema mismatch",
        )
        return metadata


    def publish_validated_snapshot(
        staging_snapshot,
        canonical_candidate,
        metadata: dict,
        path=EXTRACT_PATH,
        snapshot_root=SNAPSHOT_ROOT,
        replace_file=None,
    ):
        """Publish snapshot first, then atomically replace the canonical file."""
        replace_file = replace_file or os.replace
        snapshot_root.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        published_snapshot = snapshot_root / metadata["snapshot_identifier"]
        if published_snapshot.exists():
            raise FileExistsError(
                f"immutable snapshot already exists: {published_snapshot}"
            )

        backup = canonical_candidate.parent / "previous-canonical.parquet"
        previous_hash = None
        if path.exists():
            previous_hash = sha256_file(path)
            shutil.copy2(path, backup)

        snapshot_published = False
        try:
            replace_file(staging_snapshot, published_snapshot)
            snapshot_published = True
            if (
                sha256_file(published_snapshot / "flattened.parquet")
                != metadata["output"]["parquet_sha256"]
            ):
                raise ValueError("published snapshot checksum mismatch")

            replace_file(canonical_candidate, path)
            if sha256_file(path) != metadata["output"]["parquet_sha256"]:
                raise ValueError("canonical Parquet checksum mismatch")
        except Exception as publication_error:
            rollback_error = None
            try:
                current_hash = sha256_file(path) if path.exists() else None
                candidate_hash = metadata["output"]["parquet_sha256"]
                if (
                    previous_hash is not None
                    and current_hash == candidate_hash
                ):
                    os.replace(backup, path)
                elif previous_hash is None and current_hash == candidate_hash:
                    path.unlink()
                if snapshot_published and published_snapshot.exists():
                    shutil.rmtree(published_snapshot)
            except Exception as error:
                rollback_error = error
            if rollback_error is not None:
                raise RuntimeError(
                    "snapshot publication and rollback both failed"
                ) from publication_error
            raise

        return published_snapshot


    def refresh_extract(
        keywords: list[str],
        path=EXTRACT_PATH,
        snapshot_root=SNAPSHOT_ROOT,
        tmp_root=TMP_ROOT,
        replace_file=None,
    ) -> tuple[pl.DataFrame, dict]:
        """Build and atomically publish a traced live extraction."""
        tmp_root.mkdir(parents=True, exist_ok=True)
        workspace = tempfile.mkdtemp(
            prefix="google-fact-check-refresh.",
            dir=tmp_root,
        )
        workspace_path = tmp_root / os.path.basename(workspace)
        staging_snapshot = workspace_path / "snapshot"
        staging_snapshot.mkdir()
        started = datetime.datetime.now(datetime.timezone.utc)

        try:
            api_key = ensure_api_key()
            frame, keyword_audits, page_records, counts = extract_keywords(
                api_key,
                keywords,
                staging_snapshot,
            )
            flattened_path = staging_snapshot / "flattened.parquet"
            frame.write_parquet(flattened_path)
            parquet_hash = sha256_file(flattened_path)
            completed = datetime.datetime.now(datetime.timezone.utc)
            snapshot_timestamp = started.strftime("%Y%m%dT%H%M%S.%fZ")
            snapshot_identifier = (
                f"{snapshot_timestamp}-{parquet_hash[:12]}"
            )
            final_snapshot = snapshot_root / snapshot_identifier
            cap_warnings = [
                warning
                for audit in keyword_audits
                for warning in audit["cap_warnings"]
            ]
            metadata = {
                "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
                "snapshot_identifier": snapshot_identifier,
                "extraction": {
                    "started_at_utc": utc_timestamp(started),
                    "completed_at_utc": utc_timestamp(completed),
                },
                "api": {
                    "endpoint": API,
                    "language_code": LANGUAGE_CODE,
                    "page_size": PAGE_SIZE,
                    "token_page_cap": MAX_TOKEN_PAGES,
                    "offset_cap": MAX_OFFSET,
                },
                "query_vocabulary": {
                    "ordered": keywords,
                    "sha256": vocabulary_sha256(keywords),
                },
                "notebook_source": {
                    "path": NOTEBOOK_PATH.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(NOTEBOOK_PATH),
                },
                "pages": page_records,
                "keyword_pagination": keyword_audits,
                "termination": {
                    "max_pages": any(
                        audit["token_terminated_by_cap"]
                        for audit in keyword_audits
                    ),
                    "max_offset": any(
                        audit["offset_terminated_by_cap"]
                        for audit in keyword_audits
                    ),
                    "cap_warnings": cap_warnings,
                },
                "counts": counts,
                "output": {
                    "schema": [
                        {"name": name, "dtype": str(dtype)}
                        for name, dtype in frame.schema.items()
                    ],
                    "row_count": frame.height,
                    "parquet_sha256": parquet_hash,
                    "parquet_bytes": flattened_path.stat().st_size,
                },
                "paths": {
                    "snapshot_directory": final_snapshot.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "raw_directory": (final_snapshot / "raw").relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "metadata": (final_snapshot / "metadata.json").relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "flattened_parquet": (
                        final_snapshot / "flattened.parquet"
                    ).relative_to(REPO_ROOT).as_posix(),
                    "canonical_parquet": path.relative_to(REPO_ROOT).as_posix(),
                },
            }
            metadata_path = staging_snapshot / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            canonical_candidate = workspace_path / "canonical.parquet"
            shutil.copyfile(flattened_path, canonical_candidate)
            validate_snapshot(
                staging_snapshot,
                api_key=api_key,
                expected_metadata=metadata,
                path=path,
                snapshot_root=snapshot_root,
                require_current_source=True,
            )
            lock_path = tmp_root / "google-fact-check-publication.lock"
            with lock_path.open("a+", encoding="utf-8") as publication_lock:
                fcntl.flock(publication_lock.fileno(), fcntl.LOCK_EX)
                try:
                    publish_validated_snapshot(
                        staging_snapshot,
                        canonical_candidate,
                        metadata,
                        path=path,
                        snapshot_root=snapshot_root,
                        replace_file=replace_file,
                    )
                finally:
                    fcntl.flock(publication_lock.fileno(), fcntl.LOCK_UN)
            return pl.read_parquet(path), metadata
        finally:
            shutil.rmtree(workspace_path, ignore_errors=True)


    def resolve_provenance(path, frame: pl.DataFrame) -> dict:
        """Match the canonical bytes to snapshot metadata by SHA-256."""
        canonical_hash = sha256_file(path)
        matching_metadata = []
        if SNAPSHOT_ROOT.exists():
            for metadata_path in sorted(SNAPSHOT_ROOT.glob("*/metadata.json")):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata["output"]["parquet_sha256"] == canonical_hash:
                    matching_metadata.append((metadata_path.parent, metadata))

        if matching_metadata:
            snapshot_directory, metadata = matching_metadata[-1]
            try:
                metadata = validate_snapshot(
                    snapshot_directory,
                    path=path,
                    snapshot_root=SNAPSHOT_ROOT,
                )
            except Exception as error:
                return {
                    "status": "traced-invalid",
                    "snapshot_identifier": metadata["snapshot_identifier"],
                    "started_at_utc": metadata["extraction"]["started_at_utc"],
                    "completed_at_utc": metadata["extraction"][
                        "completed_at_utc"
                    ],
                    "canonical_parquet_sha256": canonical_hash,
                    "row_count": frame.height,
                    "raw_page_count": None,
                    "raw_responses_available": False,
                    "cap_warnings": [],
                    "integrity_error": str(error),
                    "metadata": None,
                }
            return {
                "status": "traced",
                "snapshot_identifier": metadata["snapshot_identifier"],
                "started_at_utc": metadata["extraction"]["started_at_utc"],
                "completed_at_utc": metadata["extraction"]["completed_at_utc"],
                "canonical_parquet_sha256": canonical_hash,
                "row_count": frame.height,
                "raw_page_count": len(metadata["pages"]),
                "raw_responses_available": True,
                "cap_warnings": metadata["termination"]["cap_warnings"],
                "integrity_error": None,
                "metadata": metadata,
            }

        return {
            "status": "legacy-untraced",
            "snapshot_identifier": None,
            "started_at_utc": None,
            "completed_at_utc": None,
            "canonical_parquet_sha256": canonical_hash,
            "row_count": frame.height,
            "raw_page_count": None,
            "raw_responses_available": False,
            "cap_warnings": [],
            "integrity_error": None,
            "metadata": None,
        }


    def load_or_extract(
        keywords: list[str],
        refresh: bool = False,
        path=EXTRACT_PATH,
    ) -> tuple[pl.DataFrame, str, dict]:
        """Read the cache, or publish a traced snapshot during a live refresh."""
        if path.exists() and not refresh:
            frame = pl.read_parquet(path)
            return (
                frame,
                f"canonical Parquet ({path.name})",
                resolve_provenance(path, frame),
            )

        frame, _ = refresh_extract(keywords, path=path)
        return frame, "Google Fact Check Tools API", resolve_provenance(path, frame)

    return (load_or_extract,)


@app.cell(hide_code=True)
def explain_refresh_control(mo):
    mo.md("""
    ## 7. Refresh control

    In a live marimo session, this button is the only action that spends API
    quota. It re-walks every keyword, validates an immutable raw-response
    snapshot, publishes that snapshot, and only then replaces the canonical
    Parquet. A static HTML export has no Python kernel, so the control is
    intentionally labeled **live session only**; cached runs and static builds
    remain offline and read-only.
    """)
    return


@app.cell
def render_refresh_control(mo):
    refresh_button = mo.ui.run_button(
        label="Re-extract from API (live session only; publishes snapshot)"
    )
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
    extract_frame, extract_source, extract_provenance = load_or_extract(
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
    flattened_csv_bytes = extract_frame.write_csv().encode("utf-8")
    canonical_parquet_bytes = EXTRACT_PATH.read_bytes()
    csv_download = mo.download(
        data=flattened_csv_bytes,
        filename="google_fact_check_tool_id_raw.csv",
        mimetype="text/csv",
        label=(
            "Download flattened CSV "
            f"({len(flattened_csv_bytes) / 1024 / 1024:.2f} MiB)"
        ),
    )
    parquet_download = mo.download(
        data=canonical_parquet_bytes,
        filename=EXTRACT_PATH.name,
        mimetype="application/vnd.apache.parquet",
        label=(
            "Download canonical Parquet "
            f"({len(canonical_parquet_bytes) / 1024:.0f} KiB)"
        ),
    )

    unknown_time = "unknown — cannot be reconstructed"
    provenance_table = pl.DataFrame({
        "provenance field": [
            "snapshot status",
            "snapshot identifier",
            "extraction started (UTC)",
            "extraction completed (UTC)",
            "canonical Parquet SHA-256",
            "canonical row count",
            "archived raw pages",
        ],
        "value": [
            extract_provenance["status"],
            extract_provenance["snapshot_identifier"] or "unknown",
            extract_provenance["started_at_utc"] or unknown_time,
            extract_provenance["completed_at_utc"] or unknown_time,
            extract_provenance["canonical_parquet_sha256"],
            f"{extract_provenance['row_count']:,}",
            (
                f"{extract_provenance['raw_page_count']:,}"
                if extract_provenance["raw_page_count"] is not None
                else "unknown — raw responses unavailable"
            ),
        ],
    })

    if extract_provenance["status"] == "legacy-untraced":
        provenance_status = mo.callout(
            mo.md(
                "**The current canonical Parquet is a legacy snapshot.** It "
                "predates raw-response archiving, and no matching provenance "
                "snapshot exists for its checksum. Its original API responses "
                "and extraction timestamps cannot be reconstructed. Filesystem "
                "modification times and Git history are not substitutes for "
                "extraction evidence."
            ),
            kind="warn",
            title="Legacy snapshot · lineage incomplete",
        )
        pagination_components = []
    elif extract_provenance["status"] == "traced-invalid":
        provenance_status = mo.callout(
            mo.md(
                "The canonical checksum matches snapshot metadata, but the "
                "snapshot failed integrity validation. Archived raw responses "
                "must not be treated as available until the snapshot is "
                "repaired or replaced.\n\n"
                f"Validation detail: `{extract_provenance['integrity_error']}`"
            ),
            kind="warn",
            title="Traced snapshot · integrity failure",
        )
        pagination_components = []
    else:
        provenance_status = mo.callout(
            mo.md(
                f"Canonical bytes match immutable snapshot "
                f"`{extract_provenance['snapshot_identifier']}`."
            ),
            kind="success",
            title="Traced snapshot",
        )
        pagination_rows = [
            {
                "keyword": audit["keyword"],
                "token pages": audit["token_page_count"],
                "token termination": audit["token_termination_reason"],
                "offset pages": audit["offset_page_count"],
                "offset termination": audit["offset_termination_reason"],
                "cap warning": "yes" if audit["cap_warnings"] else "no",
            }
            for audit in extract_provenance["metadata"]["keyword_pagination"]
        ]
        pagination_components = [
            mo.md("### Per-keyword pagination audit"),
            mo.ui.table(
                pl.DataFrame(pagination_rows),
                selection=None,
                show_column_summaries=False,
            ),
        ]
        if extract_provenance["cap_warnings"]:
            warning_list = "\n".join(
                f"- {warning}" for warning in extract_provenance["cap_warnings"]
            )
            pagination_components.insert(
                0,
                mo.callout(
                    mo.md(warning_list),
                    kind="warn",
                    title="Pagination safety cap reached",
                ),
            )

    mo.vstack([
        mo.callout(
            mo.md(
                f"Loaded **{extract_frame.height:,} keyword-match rows** from "
                f"**{extract_source}** into "
                f"`{EXTRACT_PATH.relative_to(mo.notebook_dir().parent)}`."
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
        mo.md("## 8. Extraction provenance"),
        provenance_status,
        mo.ui.table(
            provenance_table,
            selection=None,
            pagination=False,
            show_column_summaries=False,
        ),
        *pagination_components,
        mo.md("### Flattened data downloads"),
        mo.hstack(
            [csv_download, parquet_download],
            justify="start",
            wrap=True,
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
