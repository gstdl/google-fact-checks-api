import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def introduce_question(mo):
    mo.md("""
    # How has the distribution of hoaxes transformed with generative AI?

    This notebook compares the Indonesian fact-check corpus before and after the
    public launch of ChatGPT on **30 November 2022**, then profiles the small set
    of records explicitly linked to AI or deepfakes.

    The comparison is descriptive. A temporal breakpoint does not show that
    generative AI caused a change: Google indexing, the configured queries,
    publisher coverage, missing dates, and changing events all shape this extract.
    The source contains fact-checked claims with heterogeneous publisher ratings;
    it does not provide a universal binary “hoax” label.
    """)
    return


@app.cell(hide_code=True)
def render_direct_answer(
    BREAKPOINT,
    ai_linked_frame,
    lexical_declining,
    lexical_rising,
    media_js_divergence,
    media_largest_shift,
    mo,
    prevalence_change_pp,
    prevalence_post,
    prevalence_pre,
    publisher_largest_shift,
    review_frame,
    sensitivity_agrees,
    sensitivity_change_pp,
    theme_declining,
    theme_rising,
):
    direct_candidate_count = int(review_frame["ai_candidate"].sum())
    direct_dated_count = int(review_frame["claim_dt"].is_not_null().sum())
    direct_media_direction = "toward" if media_largest_shift["change_pp"] > 0 else "away from"
    direct_sensitivity_label = "supports the same direction" if sensitivity_agrees else "does not support the same direction"

    direct_answer = mo.vstack([
        mo.md("""
        ## 1. Direct answer

        [Identification](#2-how-ai-linked-records-were-identified) ·
        [Emergence timeline](#3-when-do-ai-linked-records-emerge) ·
        [Corpus-wide changes](#4-what-else-changed-across-the-whole-corpus) ·
        [Methods and records](#5-methods-limitations-records-and-download)
        """),
        mo.hstack([
            mo.stat(f"{direct_candidate_count:,}", label="AI/deepfake candidates", bordered=True),
            mo.stat(f"{ai_linked_frame.height:,}", label="Rule-confirmed records", bordered=True),
            mo.stat(f"{prevalence_pre['ai_linked_claims']:,}", label="Primary pre period", bordered=True),
            mo.stat(f"{prevalence_post['ai_linked_claims']:,}", label="Primary post period", bordered=True),
            mo.stat(f"{direct_dated_count:,}", label="Records with claim dates", bordered=True),
        ], widths="equal", wrap=True),
        mo.callout(
            mo.md(
                f"Rule-confirmed AI-linked reviewed claims move from "
                f"**{prevalence_pre['ai_linked_claims']:,} of {prevalence_pre['all_reviewed_claims']:,} "
                f"({prevalence_pre['ai_linked_pct']:.3f}%)** before the breakpoint to "
                f"**{prevalence_post['ai_linked_claims']:,} of {prevalence_post['all_reviewed_claims']:,} "
                f"({prevalence_post['ai_linked_pct']:.3f}%)** after it, a "
                f"**{prevalence_change_pp:+.3f} percentage-point** change. The zero pre-period "
                "numerator supports observed emergence, not a finite growth ratio."
            ),
            kind="info",
            title="Observed emergence across full dated history",
        ),
        mo.md(
            f"Across all unique claim texts, the largest mentioned-format movement is "
            f"**{direct_media_direction} {media_largest_shift['modality']}** "
            f"(**{media_largest_shift['change_pp']:+.2f} pp**; JSD **{media_js_divergence:.4f}**). "
            f"The strongest co-theme rise is **{theme_rising['theme']}** "
            f"(**{theme_rising['change_pp']:+.2f} pp**) and the strongest decline is "
            f"**{theme_declining['theme']}** (**{theme_declining['change_pp']:+.2f} pp**). "
            f"In wording, **{lexical_rising['term']}** rises most and "
            f"**{lexical_declining['term']}** declines most. Balanced-window sensitivity "
            f"**{direct_sensitivity_label}** ({sensitivity_change_pp:+.3f} pp)."
        ),
        mo.callout(
            mo.md(
                f"The analysis cannot attribute broader post-{BREAKPOINT.date()} corpus changes "
                f"to generative AI. Google indexing, fixed queries, missing dates, current events, "
                f"and publisher composition—including the largest shift at "
                f"`{publisher_largest_shift['publisher_site']}` "
                f"({publisher_largest_shift['share_change_pp']:+.2f} pp)—all shape the comparison."
            ),
            kind="warn",
            title="Temporal association is not a causal effect",
        ),
    ])
    direct_answer
    return


@app.cell(hide_code=True)
def explain_method(mo):
    method_explanation = mo.md("""
    ### Detailed comparison design

    `data/google_fact_check_tool_id.parquet` is read directly; this notebook makes
    no network calls and needs no cloud credentials.

    Two grains are used:

    1. one `(review_url, claim_text)` record for prevalence and publisher coverage;
    2. one `(era, claim_text)` row for content analysis, so several publishers
       reviewing the same wording do not overweight it.

    The primary comparison uses every reviewed claim with a parseable claim date,
    split at ChatGPT's public launch:

    - **Pre:** all dated records before 2022-11-30;
    - **Post:** all dated records from 2022-11-30 onward.

    Because the available pre- and post-launch histories have unequal durations,
    balanced three-year windows are retained as a sensitivity check. The full
    dated corpus supplies the primary pre/post distribution because this extract
    contains no rule-confirmed AI-linked record before launch. AI-linked records
    are therefore treated as an emerging post-launch subset, not as a valid
    two-era distribution of their own.
    """)
    return (method_explanation,)


@app.cell
def setup_environment():
    import re
    from datetime import datetime, timezone

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from sklearn.feature_extraction.text import CountVectorizer

    REPO_ROOT = mo.notebook_dir().parent
    EXTRACT_PATH = REPO_ROOT / "data" / "google_fact_check_tool_id.parquet"
    BREAKPOINT = datetime(2022, 11, 30, tzinfo=timezone.utc)
    BALANCED_PRE_START = datetime(2019, 11, 30, tzinfo=timezone.utc)
    BALANCED_POST_END = datetime(2025, 11, 30, tzinfo=timezone.utc)

    if not EXTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EXTRACT_PATH}; run notebooks/00_google_fact_check_tool_id.py first."
        )

    AI_TERM_PATTERNS = [
        ("standalone AI", r"(?i)(?<![\w])ai(?![\w])"),
        ("deepfake", r"(?i)\bdeep[\s-]?fake\b"),
        ("kecerdasan buatan", r"(?i)\bkecerdasan\s+buatan\b"),
        ("artificial intelligence", r"(?i)\bartificial\s+intelligence\b"),
        ("AI generatif", r"(?i)\b(?:ai\s+generatif|generative\s+ai)\b"),
        ("ChatGPT", r"(?i)\bchat\s*gpt\b"),
        ("OpenAI", r"(?i)\bopen\s*ai\b"),
        ("Gemini", r"(?i)\bgemini\b"),
        ("Midjourney", r"(?i)\bmidjourney\b"),
        ("DALL-E", r"(?i)\bdall[\s-]?e\b"),
        ("Sora", r"(?i)\bsora\b"),
    ]
    MEDIA_TERM_PATTERNS = {
        "image": r"(?i)\b(?:foto|gambar|citra|image|poster|screenshot|tangkapan\s+layar)\b",
        "video": r"(?i)\b(?:video|footage|klip|rekaman\s+video)\b",
        "audio": r"(?i)\b(?:audio|suara|voice|rekaman\s+suara)\b",
        "text/document": r"(?i)\b(?:teks|dokumen|surat|artikel|pesan|narasi|document|message)\b",
    }
    THEME_KEYWORDS = {
        "politics/elections": {
            "pemilu", "kpu", "jokowi", "prabowo", "gibran", "anies", "ganjar",
            "megawati", "luhut", "sri mulyani", "presiden", "menteri", "demo", "ijazah",
        },
        "scams/economy": {
            "korupsi", "kpk", "bansos", "subsidi", "pajak", "rupiah", "bank", "uang",
            "gaji", "penipuan", "undian", "lowongan kerja", "kartu", "beras",
            "minyak goreng", "sembako",
        },
        "health": {
            "vaksin", "covid", "imunisasi", "dokter", "rumah sakit", "obat", "kanker",
            "diabetes", "stunting", "bpjs", "bpom", "kemenkes", "telur", "susu", "gula",
            "mie instan", "kopi",
        },
        "disasters/environment": {
            "gempa", "banjir", "tsunami", "erupsi", "gunung", "kebakaran", "cuaca",
            "hujan", "listrik", "pertamina", "tambang", "nikel",
        },
        "security/society": {
            "polisi", "tni", "narkoba", "penculikan", "sekolah", "guru", "mahasiswa",
            "artis", "ular", "harimau", "buaya", "kiamat",
        },
        "religion/identity": {
            "islam", "masjid", "gereja", "ramadan", "haji", "natal", "mudik", "nu",
            "muhammadiyah",
        },
        "technology/media": {"hoaks", "video", "foto", "deepfake", "ai", "chip", "5g"},
        "transport/infrastructure": {
            "kereta", "pesawat", "garuda", "tol", "ojek", "nusantara", "ibu kota",
        },
        "places": {
            "jakarta", "surabaya", "bandung", "medan", "yogyakarta", "makassar", "bali",
            "aceh", "papua", "kalimantan", "sumatera",
        },
        "international": {
            "china", "israel", "palestina", "ukraina", "amerika", "malaysia", "singapura",
        },
    }
    MODALITY_ORDER = ["image", "video", "audio", "text/document", "multimodal", "unspecified"]
    ERA_ORDER = ["Pre", "Post"]

    ai_term_compiled = [(label, re.compile(pattern)) for label, pattern in AI_TERM_PATTERNS]
    media_term_compiled = {
        label: re.compile(pattern) for label, pattern in MEDIA_TERM_PATTERNS.items()
    }

    def find_ai_terms(text: str) -> list[str]:
        return [label for label, pattern in ai_term_compiled if pattern.search(text)]

    def classify_modality(text: str) -> str:
        matches = [label for label, pattern in media_term_compiled.items() if pattern.search(text)]
        if not matches:
            return "unspecified"
        if len(matches) == 1:
            return matches[0]
        return "multimodal"

    def classify_themes(keywords: list[str]) -> list[str]:
        keyword_set = set(keywords)
        return [
            theme for theme, vocabulary in THEME_KEYWORDS.items()
            if keyword_set.intersection(vocabulary)
        ]

    ai_term_dictionary = pl.DataFrame({
        "confirmation_term": [label for label, _ in AI_TERM_PATTERNS],
        "regular_expression": [pattern for _, pattern in AI_TERM_PATTERNS],
    })
    media_term_dictionary = pl.DataFrame({
        "modality": list(MEDIA_TERM_PATTERNS),
        "regular_expression": list(MEDIA_TERM_PATTERNS.values()),
    })
    theme_dictionary = pl.DataFrame({
        "theme": list(THEME_KEYWORDS),
        "configured_keywords": [" | ".join(sorted(values)) for values in THEME_KEYWORDS.values()],
    })

    return (
        BREAKPOINT,
        CountVectorizer,
        ERA_ORDER,
        EXTRACT_PATH,
        MODALITY_ORDER,
        BALANCED_POST_END,
        BALANCED_PRE_START,
        StopWordRemoverFactory,
        THEME_KEYWORDS,
        ai_term_dictionary,
        alt,
        classify_modality,
        classify_themes,
        find_ai_terms,
        media_term_dictionary,
        mo,
        np,
        pl,
        theme_dictionary,
    )


@app.cell
def build_analytical_frames(
    BREAKPOINT,
    EXTRACT_PATH,
    BALANCED_POST_END,
    BALANCED_PRE_START,
    MODALITY_ORDER,
    classify_modality,
    classify_themes,
    find_ai_terms,
    mo,
    pl,
):
    raw_frame = pl.read_parquet(EXTRACT_PATH)
    record_key_columns = ["review_url", "claim_text"]
    metadata_fields = [
        "claimant",
        "claim_date",
        "publisher_name",
        "publisher_site",
        "review_title",
        "textual_rating",
        "review_date",
        "language_code",
    ]
    review_frame = (
        raw_frame
        .sort([*record_key_columns, "keyword"])
        .group_by(record_key_columns, maintain_order=True)
        .agg(
            pl.col("keyword").unique().sort().alias("matched_keywords"),
            *[
                pl.col(field).drop_nulls().first().alias(field)
                for field in metadata_fields
            ],
        )
        .with_columns(
            pl.concat_str(record_key_columns, separator=" ⟂ ").alias("record_id"),
            pl.col("claim_date").str.to_datetime(strict=False, time_zone="UTC").alias("claim_dt"),
            pl.col("review_date").str.to_datetime(strict=False, time_zone="UTC").alias("review_dt"),
            pl.col("publisher_site").fill_null("missing publisher").alias("publisher_site"),
        )
        .sort(record_key_columns)
    )

    ai_candidates = []
    matched_ai_terms = []
    record_modalities = []
    record_themes = []
    for record in review_frame.iter_rows(named=True):
        combined_text = " ".join([
            record.get("claim_text") or "",
            record.get("review_title") or "",
        ])
        keywords = record["matched_keywords"]
        ai_candidates.append(bool({"ai", "deepfake"}.intersection(keywords)))
        matched_ai_terms.append(find_ai_terms(combined_text))
        record_modalities.append(classify_modality(record.get("claim_text") or ""))
        record_themes.append(classify_themes(keywords))

    review_frame = (
        review_frame
        .with_columns(
            pl.Series("ai_candidate", ai_candidates, dtype=pl.Boolean),
            pl.Series("matched_ai_terms", matched_ai_terms, dtype=pl.List(pl.Utf8)),
            pl.Series("mentioned_modality", record_modalities, dtype=pl.Utf8),
            pl.Series("matched_themes", record_themes, dtype=pl.List(pl.Utf8)),
        )
        .with_columns(
            (pl.col("ai_candidate") & (pl.col("matched_ai_terms").list.len() > 0))
            .alias("is_ai_linked"),
            pl.when(pl.col("claim_dt") < BREAKPOINT)
            .then(pl.lit("Pre"))
            .when(pl.col("claim_dt") >= BREAKPOINT)
            .then(pl.lit("Post"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("comparison_era"),
            pl.when(
                (pl.col("claim_dt") >= BALANCED_PRE_START) &
                (pl.col("claim_dt") < BREAKPOINT)
            )
            .then(pl.lit("Pre"))
            .when(
                (pl.col("claim_dt") >= BREAKPOINT) &
                (pl.col("claim_dt") < BALANCED_POST_END)
            )
            .then(pl.lit("Post"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("balanced_window_era"),
            (
                pl.col("claim_date").is_not_null() & pl.col("claim_dt").is_null()
            ).alias("claim_date_parse_failed"),
        )
    )

    primary_frame = review_frame.filter(pl.col("comparison_era").is_not_null())
    dated_frame = review_frame.filter(pl.col("claim_dt").is_not_null())
    ai_linked_frame = review_frame.filter(pl.col("is_ai_linked"))
    rejected_ai_candidates = review_frame.filter(
        pl.col("ai_candidate") & ~pl.col("is_ai_linked")
    )

    content_groups = {}
    for record in primary_frame.iter_rows(named=True):
        claim_text = record.get("claim_text") or ""
        if not claim_text:
            continue
        content_key = (record["comparison_era"], claim_text)
        if content_key not in content_groups:
            content_groups[content_key] = {
                "comparison_era": record["comparison_era"],
                "claim_text": claim_text,
                "matched_keywords": set(),
                "is_ai_linked": False,
            }
        content_groups[content_key]["matched_keywords"].update(record["matched_keywords"])
        content_groups[content_key]["is_ai_linked"] = (
            content_groups[content_key]["is_ai_linked"] or record["is_ai_linked"]
        )

    content_rows = []
    for content_group in content_groups.values():
        content_keywords = sorted(content_group["matched_keywords"])
        content_rows.append({
            "comparison_era": content_group["comparison_era"],
            "claim_text": content_group["claim_text"],
            "matched_keywords": content_keywords,
            "is_ai_linked": content_group["is_ai_linked"],
            "mentioned_modality": classify_modality(content_group["claim_text"]),
            "matched_themes": classify_themes(content_keywords),
        })
    content_frame = pl.DataFrame(content_rows).sort(["comparison_era", "claim_text"])

    assert review_frame.height == review_frame.unique(record_key_columns).height
    assert primary_frame.height == dated_frame.height
    assert primary_frame["claim_dt"].null_count() == 0
    assert review_frame.filter(
        pl.col("balanced_window_era").is_not_null()
        & (pl.col("balanced_window_era") != pl.col("comparison_era"))
    ).height == 0
    assert ai_linked_frame.filter(~pl.col("ai_candidate")).height == 0
    assert ai_linked_frame.filter(pl.col("matched_ai_terms").list.len() == 0).height == 0
    assert set(content_frame["mentioned_modality"].unique()) <= set(MODALITY_ORDER)
    assert primary_frame.filter(pl.col("comparison_era") == "Pre").height > 0
    assert primary_frame.filter(pl.col("comparison_era") == "Post").height > 0
    assert primary_frame.filter(
        (pl.col("comparison_era") == "Pre") & (pl.col("claim_dt") >= BREAKPOINT)
    ).height == 0
    assert primary_frame.filter(
        (pl.col("comparison_era") == "Post") & (pl.col("claim_dt") < BREAKPOINT)
    ).height == 0

    grain_summary = pl.DataFrame([
        {
            "frame": "raw_frame",
            "grain": "keyword × reviewed claim",
            "rows": raw_frame.height,
            "purpose": "source and retrieval audit",
        },
        {
            "frame": "review_frame",
            "grain": "distinct (review URL, claim text)",
            "rows": review_frame.height,
            "purpose": "prevalence and publishers",
        },
        {
            "frame": "content_frame",
            "grain": "distinct (comparison era, claim text)",
            "rows": content_frame.height,
            "purpose": "modality, themes, and language",
        },
        {
            "frame": "ai_linked_frame",
            "grain": "rule-confirmed AI-linked reviewed claim",
            "rows": ai_linked_frame.height,
            "purpose": "AI emergence and audit",
        },
    ])
    date_quality_summary = pl.DataFrame([
        {
            "date status": "parseable claim date",
            "reviewed claims": dated_frame.height,
            "temporal treatment": "primary comparison denominator",
        },
        {
            "date status": "missing claim date",
            "reviewed claims": review_frame["claim_date"].null_count(),
            "temporal treatment": "retained for audit; excluded from time comparisons",
        },
        {
            "date status": "unparseable non-null claim date",
            "reviewed claims": review_frame.filter(pl.col("claim_date_parse_failed")).height,
            "temporal treatment": "retained for audit; excluded from time comparisons",
        },
        {
            "date status": "inside balanced windows",
            "reviewed claims": review_frame.filter(
                pl.col("balanced_window_era").is_not_null()
            ).height,
            "temporal treatment": "balanced-window sensitivity denominator",
        },
        {
            "date status": "parseable but outside balanced windows",
            "reviewed claims": dated_frame.filter(
                pl.col("balanced_window_era").is_null()
            ).height,
            "temporal treatment": "included in primary; excluded from sensitivity",
        },
    ])

    frame_section = mo.vstack([
        mo.md(
            f"The extract contains **{raw_frame.height:,} keyword links** and "
            f"**{review_frame.height:,} reviewed-claim records**. The conservative "
            f"AI filter retains **{ai_linked_frame.height:,} records** from "
            f"**{review_frame.filter(pl.col('ai_candidate')).height:,} candidates**."
        ),
        mo.ui.table(
            grain_summary,
            selection=None,
            pagination=False,
            show_column_summaries=False,
        ),
        mo.md("### Claim-date eligibility"),
        mo.ui.table(
            date_quality_summary,
            selection=None,
            pagination=False,
            show_column_summaries=False,
        ),
    ])
    return (
        ai_linked_frame,
        content_frame,
        dated_frame,
        frame_section,
        primary_frame,
        rejected_ai_candidates,
        review_frame,
    )


@app.cell(hide_code=True)
def explain_ai_identification(mo):
    mo.md("""
    ## 2. How AI-linked records were identified

    Identification is deliberately conservative. A record must first have been
    retrieved by the configured `ai` or `deepfake` query and must then contain an
    explicit AI term in its claim text or review title. Query candidates without
    textual confirmation are rejected; records never retrieved by either query
    cannot enter the subset.

    “Rule-confirmed” does not mean manually adjudicated. Names such as `Gemini`,
    `Sora`, or `Ai` can still be ambiguous even after query filtering. The rule
    favors precision over recall, and every matched term is retained so possible
    false positives can be inspected rather than hidden.
    """)
    return


@app.cell
def render_ai_identification(
    ai_linked_frame,
    ai_term_dictionary,
    mo,
    pl,
    rejected_ai_candidates,
    review_frame,
):
    ai_candidate_count = review_frame.filter(pl.col("ai_candidate")).height
    ai_confirmed_dated = ai_linked_frame.filter(pl.col("claim_dt").is_not_null()).height
    ai_confirmed_undated = ai_linked_frame.height - ai_confirmed_dated
    ai_pre_primary_count = ai_linked_frame.filter(pl.col("comparison_era") == "Pre").height
    ai_post_primary_count = ai_linked_frame.filter(pl.col("comparison_era") == "Post").height

    ai_flow = pl.DataFrame([
        {"stage": "ai/deepfake query candidates", "records": ai_candidate_count},
        {"stage": "rule-confirmed by explicit text", "records": ai_linked_frame.height},
        {"stage": "rejected candidates", "records": rejected_ai_candidates.height},
        {"stage": "rule-confirmed with parseable claim date", "records": ai_confirmed_dated},
        {"stage": "rule-confirmed without parseable claim date", "records": ai_confirmed_undated},
        {"stage": "rule-confirmed in primary pre period", "records": ai_pre_primary_count},
        {"stage": "rule-confirmed in primary post period", "records": ai_post_primary_count},
    ])

    ai_audit_section = mo.vstack([
        mo.callout(
            mo.md(
                f"The strict filter finds **{ai_pre_primary_count} rule-confirmed AI-linked "
                f"dated records before the breakpoint** and **{ai_post_primary_count} after it** "
                "across the full dated history. With no pre-period AI-linked baseline, "
                "their own content distribution cannot be compared across eras; the "
                "corpus-wide comparison below is used instead."
            ),
            kind="warn",
            title="Why the scope changed to corpus-wide pre/post",
        ),
        mo.ui.table(ai_flow, selection=None, pagination=False, show_column_summaries=False),
        mo.accordion({
            "Confirmation dictionary and ambiguous terms": mo.ui.table(
                ai_term_dictionary,
                selection=None,
                pagination=False,
                show_column_summaries=False,
            ),
            f"Rule-confirmed AI-linked records ({ai_linked_frame.height:,})": mo.ui.table(
                ai_linked_frame.select(
                    "claim_dt", "matched_ai_terms", "matched_keywords", "claim_text",
                    "review_title", "publisher_site", "review_url",
                ),
                selection=None,
                pagination=True,
                page_size=15,
                wrapped_columns=["claim_text", "review_title"],
            ),
            f"Rejected keyword candidates ({rejected_ai_candidates.height:,})": mo.ui.table(
                rejected_ai_candidates.select(
                    "claim_dt", "matched_keywords", "claim_text", "review_title",
                    "publisher_site", "review_url",
                ),
                selection=None,
                pagination=True,
                page_size=15,
                wrapped_columns=["claim_text", "review_title"],
            ),
        }, multiple=False),
    ])
    ai_audit_section
    return ai_post_primary_count, ai_pre_primary_count


@app.cell(hide_code=True)
def explain_prevalence(mo):
    mo.md("""
    ## 3. When do AI-linked records emerge?

    Counts use reviewed-claim records. The denominator for each rate is every
    reviewed claim with a claim date in the same month or era—not only AI query
    results. A zero pre-period numerator makes a conventional prevalence ratio
    infinite, so the report states emergence and the percentage-point change
    instead of emitting an invalid finite ratio.
    """)
    return


@app.cell
def analyze_prevalence(
    BREAKPOINT,
    ERA_ORDER,
    alt,
    dated_frame,
    mo,
    pl,
    primary_frame,
):
    prevalence_timeline = dated_frame
    prevalence_monthly = (
        prevalence_timeline
        .with_columns(pl.col("claim_dt").dt.truncate("1mo").alias("month_start"))
        .with_columns(
            pl.when(
                (pl.col("claim_dt") >= BREAKPOINT) &
                (pl.col("month_start") < BREAKPOINT)
            )
            .then(pl.lit(BREAKPOINT))
            .otherwise(pl.col("month_start"))
            .alias("month_segment_start")
        )
        .group_by("month_segment_start")
        .agg(
            pl.len().alias("all_reviewed_claims"),
            pl.col("is_ai_linked").sum().alias("ai_linked_claims"),
        )
        .with_columns(
            (100 * pl.col("ai_linked_claims") / pl.col("all_reviewed_claims"))
            .round(3)
            .alias("ai_linked_pct")
        )
        .sort("month_segment_start")
    )
    prevalence_quarterly = (
        prevalence_timeline
        .with_columns(pl.col("claim_dt").dt.truncate("1q").alias("quarter_start"))
        .with_columns(
            pl.when(
                (pl.col("claim_dt") >= BREAKPOINT) &
                (pl.col("quarter_start") < BREAKPOINT)
            )
            .then(pl.lit(BREAKPOINT))
            .otherwise(pl.col("quarter_start"))
            .alias("quarter_segment_start")
        )
        .group_by("quarter_segment_start")
        .agg(
            pl.len().alias("all_reviewed_claims"),
            pl.col("is_ai_linked").sum().alias("ai_linked_claims"),
        )
        .with_columns(
            (100 * pl.col("ai_linked_claims") / pl.col("all_reviewed_claims"))
            .round(3)
            .alias("ai_linked_pct")
        )
        .sort("quarter_segment_start")
    )

    primary_prevalence_rows = []
    for prevalence_era in ERA_ORDER:
        prevalence_slice = primary_frame.filter(pl.col("comparison_era") == prevalence_era)
        prevalence_ai_count = prevalence_slice.filter(pl.col("is_ai_linked")).height
        primary_prevalence_rows.append({
            "era": prevalence_era,
            "all_reviewed_claims": prevalence_slice.height,
            "ai_linked_claims": prevalence_ai_count,
            "ai_linked_pct": 100 * prevalence_ai_count / prevalence_slice.height,
        })
    primary_prevalence = pl.DataFrame(primary_prevalence_rows)

    sensitivity_rows = []
    for sensitivity_era in ERA_ORDER:
        sensitivity_slice = dated_frame.filter(
            pl.col("balanced_window_era") == sensitivity_era
        )
        assert sensitivity_slice.height > 0
        sensitivity_ai_count = sensitivity_slice.filter(pl.col("is_ai_linked")).height
        sensitivity_rows.append({
            "era": sensitivity_era,
            "all_reviewed_claims": sensitivity_slice.height,
            "ai_linked_claims": sensitivity_ai_count,
            "ai_linked_pct": 100 * sensitivity_ai_count / sensitivity_slice.height,
        })
    sensitivity_prevalence = pl.DataFrame(sensitivity_rows)

    prevalence_pre = primary_prevalence.row(0, named=True)
    prevalence_post = primary_prevalence.row(1, named=True)
    sensitivity_pre = sensitivity_prevalence.row(0, named=True)
    sensitivity_post = sensitivity_prevalence.row(1, named=True)
    prevalence_change_pp = prevalence_post["ai_linked_pct"] - prevalence_pre["ai_linked_pct"]
    sensitivity_change_pp = sensitivity_post["ai_linked_pct"] - sensitivity_pre["ai_linked_pct"]
    prevalence_ratio_label = (
        "not finite (zero pre-period baseline)"
        if prevalence_pre["ai_linked_pct"] == 0
        else f"{prevalence_post['ai_linked_pct'] / prevalence_pre['ai_linked_pct']:.2f}×"
    )
    sensitivity_agrees = (
        (prevalence_change_pp > 0 and sensitivity_change_pp > 0) or
        (prevalence_change_pp < 0 and sensitivity_change_pp < 0) or
        (prevalence_change_pp == 0 and sensitivity_change_pp == 0)
    )

    breakpoint_rule = alt.Chart(pl.DataFrame({"breakpoint": [BREAKPOINT]})).mark_rule(
        color="#E45756", strokeDash=[6, 4], strokeWidth=2
    ).encode(x="breakpoint:T")
    prevalence_count_chart = (
        alt.Chart(prevalence_monthly)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("month_segment_start:T", title="Claim month / launch split"),
            y=alt.Y("ai_linked_claims:Q", title="Rule-confirmed AI-linked records"),
            tooltip=[
                alt.Tooltip("month_segment_start:T", title="Period start", format="%Y-%m-%d"),
                "ai_linked_claims:Q", "all_reviewed_claims:Q",
            ],
        )
        .properties(width="container", height=250, title="Monthly AI-linked reviewed claims")
    )
    prevalence_rate_chart = (
        alt.Chart(prevalence_quarterly)
        .mark_line(point=True, color="#F58518", strokeWidth=2)
        .encode(
            x=alt.X("quarter_segment_start:T", title="Claim quarter / launch split"),
            y=alt.Y("ai_linked_pct:Q", title="AI-linked share of reviewed claims (%)"),
            tooltip=[
                alt.Tooltip("quarter_segment_start:T", title="Period start", format="%Y-%m-%d"),
                alt.Tooltip("ai_linked_pct:Q", format=".3f"),
                "ai_linked_claims:Q", "all_reviewed_claims:Q",
            ],
        )
        .properties(width="container", height=250, title="Quarterly AI-linked prevalence (launch period split)")
    )

    prevalence_section = mo.vstack([
        mo.md(
            f"Across the full dated history, explicit AI-linked records move from "
            f"**{prevalence_pre['ai_linked_claims']:,} of {prevalence_pre['all_reviewed_claims']:,} "
            f"({prevalence_pre['ai_linked_pct']:.3f}%)** before launch to "
            f"**{prevalence_post['ai_linked_claims']:,} of {prevalence_post['all_reviewed_claims']:,} "
            f"({prevalence_post['ai_linked_pct']:.3f}%)** after launch—a "
            f"**{prevalence_change_pp:+.3f} percentage-point** change. The prevalence "
            f"ratio is **{prevalence_ratio_label}**."
        ),
        mo.md(
            "The timeline includes every parseable claim date. November and Q4 2022 are "
            "split exactly at the launch timestamp, so post-launch records are not plotted "
            "inside a pre-launch bin."
        ),
        prevalence_count_chart + breakpoint_rule,
        mo.ui.tabs({
            "Quarterly prevalence": prevalence_rate_chart + breakpoint_rule,
            "Full-history result": mo.ui.table(
                primary_prevalence,
                selection=None,
                pagination=False,
                show_column_summaries=False,
            ),
            "Balanced-window sensitivity": mo.ui.table(
                sensitivity_prevalence,
                selection=None,
                pagination=False,
                show_column_summaries=False,
            ),
        }),
        mo.callout(
            mo.md(
                "The pre-period numerator is zero. The report therefore states emergence and "
                "a percentage-point change rather than a finite prevalence ratio."
            ),
            kind="warn",
            title="Zero-baseline interpretation",
        ),
    ])
    prevalence_section
    return (
        prevalence_change_pp,
        prevalence_post,
        prevalence_pre,
        sensitivity_agrees,
        sensitivity_change_pp,
    )


@app.cell(hide_code=True)
def explain_media_distribution(ai_linked_frame, mo):
    media_explanation = mo.md("""
    ### Mentioned media modality

    The classifier assigns the visible format mentioned in claim wording—not how the
    content was produced. Shares use unique claim texts in each full-history era and sum
    to 100%; Jensen–Shannon divergence describes distributional distance without
    implying statistical significance.
    """)
    corpus_wide_intro = mo.vstack([
        mo.md("## 4. What else changed across the whole corpus?"),
        mo.callout(
            mo.md(
                f"These tabs compare **all eligible claim texts**, not distributions "
                f"within the **{ai_linked_frame.height}-record** rule-confirmed AI subset."
            ),
            kind="warn",
            title="Corpus-wide comparison",
        ),
    ])
    corpus_wide_intro
    return (media_explanation,)


@app.cell
def analyze_media_distribution(
    ERA_ORDER,
    MODALITY_ORDER,
    alt,
    content_frame,
    media_term_dictionary,
    media_explanation,
    mo,
    np,
    pl,
):
    media_rows = []
    media_long_rows = []
    media_totals = {
        media_era: content_frame.filter(pl.col("comparison_era") == media_era).height
        for media_era in ERA_ORDER
    }
    assert all(media_totals[media_era] > 0 for media_era in ERA_ORDER)

    for media_category in MODALITY_ORDER:
        media_summary_row = {"modality": media_category}
        for media_era in ERA_ORDER:
            media_count = content_frame.filter(
                (pl.col("comparison_era") == media_era) &
                (pl.col("mentioned_modality") == media_category)
            ).height
            media_share = 100 * media_count / media_totals[media_era]
            media_summary_row[f"{media_era.lower()}_claim_texts"] = media_count
            media_summary_row[f"{media_era.lower()}_share_pct"] = media_share
            media_long_rows.append({
                "modality": media_category,
                "era": media_era,
                "claim_texts": media_count,
                "share_pct": media_share,
            })
        media_summary_row["change_pp"] = (
            media_summary_row["post_share_pct"] - media_summary_row["pre_share_pct"]
        )
        media_rows.append(media_summary_row)

    media_summary = pl.DataFrame(media_rows)
    media_long = pl.DataFrame(media_long_rows)
    assert abs(media_summary["pre_share_pct"].sum() - 100) < 1e-9
    assert abs(media_summary["post_share_pct"].sum() - 100) < 1e-9

    media_pre_probabilities = media_summary["pre_share_pct"].to_numpy() / 100
    media_post_probabilities = media_summary["post_share_pct"].to_numpy() / 100
    media_midpoint = (media_pre_probabilities + media_post_probabilities) / 2

    def media_kl_divergence(distribution, midpoint):
        positive = distribution > 0
        return float(np.sum(distribution[positive] * np.log2(distribution[positive] / midpoint[positive])))

    media_js_divergence = 0.5 * (
        media_kl_divergence(media_pre_probabilities, media_midpoint) +
        media_kl_divergence(media_post_probabilities, media_midpoint)
    )
    media_largest_shift = media_summary.sort(
        pl.col("change_pp").abs(), descending=True
    ).row(0, named=True)

    media_chart = (
        alt.Chart(media_long)
        .mark_bar()
        .encode(
            x=alt.X("share_pct:Q", title="Share of unique claim texts (%)"),
            y=alt.Y("modality:N", title=None, sort=MODALITY_ORDER),
            color=alt.Color("era:N", title="Era", sort=ERA_ORDER),
            yOffset="era:N",
            tooltip=["era:N", "modality:N", "claim_texts:Q", alt.Tooltip("share_pct:Q", format=".2f")],
        )
        .properties(width="container", height=300, title="Mentioned media format before and after launch")
    )
    media_delta_chart = (
        alt.Chart(media_summary)
        .mark_bar()
        .encode(
            x=alt.X("change_pp:Q", title="Post − pre change (percentage points)"),
            y=alt.Y("modality:N", title=None, sort=MODALITY_ORDER),
            color=alt.condition(
                alt.datum.change_pp >= 0,
                alt.value("#54A24B"),
                alt.value("#E45756"),
            ),
            tooltip=["modality:N", alt.Tooltip("change_pp:Q", format="+.2f")],
        )
        .properties(width="container", height=300, title="Direction and size of modality shifts")
    )

    media_section = mo.vstack([
        media_explanation,
        mo.md(
            f"The largest absolute modality change is **{media_largest_shift['modality']}** "
            f"at **{media_largest_shift['change_pp']:+.2f} percentage points**. "
            f"The complete modality distribution has Jensen–Shannon divergence "
            f"**{media_js_divergence:.4f}**."
        ),
        media_chart,
        mo.accordion({
            "Shift chart": media_delta_chart,
            "Modality summary": mo.ui.table(
                media_summary,
                selection=None,
                pagination=False,
                show_column_summaries=False,
            ),
            "Auditable modality lexicon": mo.ui.table(
                media_term_dictionary,
                selection=None,
                pagination=False,
                show_column_summaries=False,
            ),
        }, multiple=False),
    ])
    return media_js_divergence, media_largest_shift, media_section


@app.cell(hide_code=True)
def explain_theme_distribution(mo):
    theme_explanation = mo.md("""
    ### Co-occurring retrieval themes

    The configured extraction vocabulary is grouped into broad, multi-label retrieval
    themes, so percentages can sum above 100%. These rates describe changing query
    coverage—not a universal taxonomy of hoaxes.
    """)
    return (theme_explanation,)


@app.cell
def analyze_theme_distribution(
    ERA_ORDER,
    THEME_KEYWORDS,
    alt,
    content_frame,
    mo,
    pl,
    theme_dictionary,
    theme_explanation,
):
    theme_totals = {
        theme_era: content_frame.filter(pl.col("comparison_era") == theme_era).height
        for theme_era in ERA_ORDER
    }
    theme_rows = []
    theme_long_rows = []
    for theme_name in THEME_KEYWORDS:
        theme_summary_row = {"theme": theme_name}
        for theme_era in ERA_ORDER:
            theme_count = content_frame.filter(
                (pl.col("comparison_era") == theme_era) &
                pl.col("matched_themes").list.contains(theme_name)
            ).height
            theme_share = 100 * theme_count / theme_totals[theme_era]
            theme_summary_row[f"{theme_era.lower()}_claim_texts"] = theme_count
            theme_summary_row[f"{theme_era.lower()}_coverage_pct"] = theme_share
            theme_long_rows.append({
                "theme": theme_name,
                "era": theme_era,
                "claim_texts": theme_count,
                "coverage_pct": theme_share,
            })
        theme_summary_row["change_pp"] = (
            theme_summary_row["post_coverage_pct"] - theme_summary_row["pre_coverage_pct"]
        )
        theme_rows.append(theme_summary_row)

    theme_summary = pl.DataFrame(theme_rows)
    theme_long = pl.DataFrame(theme_long_rows)
    theme_rising = theme_summary.sort("change_pp", descending=True).row(0, named=True)
    theme_declining = theme_summary.sort("change_pp").row(0, named=True)
    theme_order = theme_summary.sort("post_coverage_pct", descending=True)["theme"].to_list()

    theme_chart = (
        alt.Chart(theme_long)
        .mark_bar()
        .encode(
            x=alt.X("coverage_pct:Q", title="Unique claim texts covered (%)"),
            y=alt.Y("theme:N", title=None, sort=theme_order),
            color=alt.Color("era:N", title="Era", sort=ERA_ORDER),
            yOffset="era:N",
            tooltip=["era:N", "theme:N", "claim_texts:Q", alt.Tooltip("coverage_pct:Q", format=".2f")],
        )
        .properties(width="container", height=380, title="Multi-label theme coverage by era")
    )
    theme_delta_chart = (
        alt.Chart(theme_summary)
        .mark_bar()
        .encode(
            x=alt.X("change_pp:Q", title="Post − pre change (percentage points)"),
            y=alt.Y("theme:N", title=None, sort=theme_order),
            color=alt.condition(
                alt.datum.change_pp >= 0,
                alt.value("#54A24B"),
                alt.value("#E45756"),
            ),
            tooltip=["theme:N", alt.Tooltip("change_pp:Q", format="+.2f")],
        )
        .properties(width="container", height=380, title="Theme coverage shifts")
    )

    theme_section = mo.vstack([
        theme_explanation,
        mo.md(
            f"The strongest rising co-theme is **{theme_rising['theme']}** "
            f"(**{theme_rising['change_pp']:+.2f} pp**), while the strongest decline "
            f"is **{theme_declining['theme']}** (**{theme_declining['change_pp']:+.2f} pp**)."
        ),
        theme_chart,
        mo.accordion({
            "Coverage shifts": theme_delta_chart,
            "Theme summary": mo.ui.table(
                theme_summary,
                selection=None,
                pagination=False,
                show_column_summaries=False,
            ),
            "Theme-to-keyword mapping": mo.ui.table(
                theme_dictionary,
                selection=None,
                pagination=True,
                page_size=10,
                wrapped_columns=["configured_keywords"],
            ),
        }, multiple=False),
    ])
    return theme_declining, theme_rising, theme_section


@app.cell(hide_code=True)
def explain_lexical_shift(mo):
    lexical_explanation = mo.md("""
    ### Lexical shift in claim wording

    Each unique claim text contributes once per era, and terms are compared by
    document prevalence rather than raw repetition. These are descriptive language
    signals that can reflect current events, retrieval coverage, or publisher wording.
    """)
    return (lexical_explanation,)


@app.cell
def analyze_lexical_shift(
    CountVectorizer,
    StopWordRemoverFactory,
    alt,
    content_frame,
    lexical_explanation,
    mo,
    np,
    pl,
):
    LEXICAL_BOILERPLATE = {
        "akun", "artikel", "benar", "benarkah", "beredar", "cek", "diklaim",
        "disebut", "fakta", "faktanya", "hoaks", "hoax", "informasi", "kabar",
        "keliru", "klaim", "menyesatkan", "narasi", "palsu", "postingan", "salah",
        "unggahan",
    }
    lexical_stopwords = sorted(
        set(StopWordRemoverFactory().get_stop_words()).union(LEXICAL_BOILERPLATE)
    )
    lexical_texts = content_frame["claim_text"].to_list()
    lexical_eras = np.asarray(content_frame["comparison_era"].to_list())
    lexical_vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=lexical_stopwords,
        ngram_range=(1, 2),
        min_df=5,
        binary=True,
        token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][\w-]{2,}\b",
    )
    lexical_matrix = lexical_vectorizer.fit_transform(lexical_texts)
    lexical_terms = np.asarray(lexical_vectorizer.get_feature_names_out())
    lexical_pre_mask = lexical_eras == "Pre"
    lexical_post_mask = lexical_eras == "Post"
    assert lexical_pre_mask.sum() > 0 and lexical_post_mask.sum() > 0

    lexical_pre_rates = (
        np.asarray(lexical_matrix[lexical_pre_mask].sum(axis=0)).ravel() /
        lexical_pre_mask.sum() * 100
    )
    lexical_post_rates = (
        np.asarray(lexical_matrix[lexical_post_mask].sum(axis=0)).ravel() /
        lexical_post_mask.sum() * 100
    )
    lexical_shift = (
        pl.DataFrame({
            "term": lexical_terms.tolist(),
            "pre_document_pct": lexical_pre_rates.tolist(),
            "post_document_pct": lexical_post_rates.tolist(),
            "change_pp": (lexical_post_rates - lexical_pre_rates).tolist(),
        })
        .with_columns(
            pl.when(pl.col("term").str.contains(" "))
            .then(pl.lit("bigram"))
            .otherwise(pl.lit("unigram"))
            .alias("term_type")
        )
    )
    lexical_gains = lexical_shift.sort("change_pp", descending=True).head(20)
    lexical_declines = lexical_shift.sort("change_pp").head(20)
    lexical_rising = lexical_gains.row(0, named=True)
    lexical_declining = lexical_declines.row(0, named=True)
    lexical_chart_frame = pl.concat([
        lexical_gains.with_columns(pl.lit("Largest gains").alias("direction")),
        lexical_declines.with_columns(pl.lit("Largest declines").alias("direction")),
    ])

    lexical_chart = (
        alt.Chart(lexical_chart_frame)
        .mark_bar()
        .encode(
            x=alt.X("change_pp:Q", title="Post − pre document prevalence (percentage points)"),
            y=alt.Y("term:N", title=None, sort="-x"),
            color=alt.condition(
                alt.datum.change_pp >= 0,
                alt.value("#54A24B"),
                alt.value("#E45756"),
            ),
            tooltip=[
                "term:N", "term_type:N",
                alt.Tooltip("pre_document_pct:Q", format=".2f"),
                alt.Tooltip("post_document_pct:Q", format=".2f"),
                alt.Tooltip("change_pp:Q", format="+.2f"),
            ],
        )
        .properties(width="container", height=620, title="Largest wording shifts")
    )

    lexical_section = mo.vstack([
        lexical_explanation,
        mo.md(
            f"The largest retained lexical gain is **{lexical_rising['term']}** "
            f"(**{lexical_rising['change_pp']:+.2f} pp**); the largest decline is "
            f"**{lexical_declining['term']}** (**{lexical_declining['change_pp']:+.2f} pp**)."
        ),
        lexical_chart,
        mo.accordion({
            "Largest gains and declines": mo.hstack([
                mo.ui.table(lexical_gains, selection=None, pagination=False, show_column_summaries=False),
                mo.ui.table(lexical_declines, selection=None, pagination=False, show_column_summaries=False),
            ], widths="equal", align="start", wrap=True),
        }),
    ])
    return lexical_declining, lexical_rising, lexical_section


@app.cell(hide_code=True)
def explain_publisher_composition(mo):
    publisher_explanation = mo.md("""
    ### Publisher composition and coverage sensitivity

    Publisher mix can change the aggregate distribution even when the underlying
    claim population does not. Publisher shares and within-publisher AI-linked rates
    expose coverage differences without treating publishers as a random sample.
    """)
    return (publisher_explanation,)


@app.cell
def analyze_publisher_composition(
    ERA_ORDER,
    alt,
    mo,
    pl,
    primary_frame,
    publisher_explanation,
):
    publisher_grouped = (
        primary_frame
        .group_by("comparison_era", "publisher_site")
        .agg(
            pl.len().alias("records"),
            pl.col("is_ai_linked").sum().alias("ai_linked_records"),
        )
        .with_columns(
            (100 * pl.col("records") / pl.col("records").sum().over("comparison_era"))
            .alias("corpus_share_pct"),
            (100 * pl.col("ai_linked_records") / pl.col("records"))
            .alias("within_publisher_ai_pct"),
        )
    )
    publisher_names = sorted(publisher_grouped["publisher_site"].unique().to_list())
    publisher_rows = []
    for publisher_name in publisher_names:
        publisher_row = {"publisher_site": publisher_name}
        for publisher_era in ERA_ORDER:
            publisher_slice = publisher_grouped.filter(
                (pl.col("publisher_site") == publisher_name) &
                (pl.col("comparison_era") == publisher_era)
            )
            if publisher_slice.height:
                publisher_values = publisher_slice.row(0, named=True)
                publisher_row[f"{publisher_era.lower()}_records"] = publisher_values["records"]
                publisher_row[f"{publisher_era.lower()}_share_pct"] = publisher_values["corpus_share_pct"]
                publisher_row[f"{publisher_era.lower()}_ai_linked_records"] = publisher_values["ai_linked_records"]
                publisher_row[f"{publisher_era.lower()}_within_ai_pct"] = publisher_values["within_publisher_ai_pct"]
            else:
                publisher_row[f"{publisher_era.lower()}_records"] = 0
                publisher_row[f"{publisher_era.lower()}_share_pct"] = 0.0
                publisher_row[f"{publisher_era.lower()}_ai_linked_records"] = 0
                publisher_row[f"{publisher_era.lower()}_within_ai_pct"] = 0.0
        publisher_row["share_change_pp"] = (
            publisher_row["post_share_pct"] - publisher_row["pre_share_pct"]
        )
        publisher_rows.append(publisher_row)
    publisher_summary = pl.DataFrame(publisher_rows).sort("post_records", descending=True)
    publisher_largest_shift = publisher_summary.sort(
        pl.col("share_change_pp").abs(), descending=True
    ).row(0, named=True)
    publisher_top_names = (
        publisher_summary
        .with_columns((pl.col("pre_records") + pl.col("post_records")).alias("total_records"))
        .sort("total_records", descending=True)
        .head(12)["publisher_site"]
        .to_list()
    )
    publisher_chart_frame = publisher_grouped.filter(
        pl.col("publisher_site").is_in(publisher_top_names)
    )

    publisher_chart = (
        alt.Chart(publisher_chart_frame)
        .mark_bar()
        .encode(
            x=alt.X("corpus_share_pct:Q", title="Share of reviewed claims (%)"),
            y=alt.Y("publisher_site:N", title=None, sort=publisher_top_names),
            color=alt.Color("comparison_era:N", title="Era", sort=ERA_ORDER),
            yOffset="comparison_era:N",
            tooltip=[
                "comparison_era:N", "publisher_site:N", "records:Q",
                alt.Tooltip("corpus_share_pct:Q", format=".2f"),
                "ai_linked_records:Q", alt.Tooltip("within_publisher_ai_pct:Q", format=".3f"),
            ],
        )
        .properties(width="container", height=430, title="Publisher composition (top 12 across both eras)")
    )

    publisher_section = mo.vstack([
        publisher_explanation,
        mo.md(
            f"The largest publisher-mix shift is **{publisher_largest_shift['publisher_site']}** "
            f"at **{publisher_largest_shift['share_change_pp']:+.2f} percentage points**. "
            "This is a coverage warning: aggregate content changes should not be read "
            "as changes in the population of all Indonesian false claims."
        ),
        publisher_chart,
        mo.accordion({
            "Publisher metrics": mo.ui.table(
                publisher_summary,
                selection=None,
                pagination=True,
                page_size=15,
                show_column_summaries=False,
            ),
        }),
    ])
    return publisher_largest_shift, publisher_section, publisher_summary


@app.cell
def render_corpus_wide_changes(
    lexical_section,
    media_section,
    mo,
    publisher_section,
    theme_section,
):
    corpus_wide_tabs = mo.ui.tabs({
        "Mentioned modality": media_section,
        "Retrieval themes": theme_section,
        "Claim wording": lexical_section,
        "Publisher composition": publisher_section,
    })
    corpus_wide_tabs
    return


@app.cell(hide_code=True)
def explain_answer(mo):
    mo.md("""
    ## 5. Methods, limitations, records, and download

    Detailed grains, date eligibility, sensitivity choices, and record-level audit
    fields remain available below. The classified subset keeps heterogeneous publisher
    ratings intact rather than creating a universal true/false or “hoax” label.
    """)
    return


@app.cell
def render_answer_and_records(
    BREAKPOINT,
    EXTRACT_PATH,
    ai_linked_frame,
    frame_section,
    lexical_declining,
    lexical_rising,
    media_js_divergence,
    media_largest_shift,
    method_explanation,
    mo,
    pl,
    prevalence_change_pp,
    prevalence_post,
    prevalence_pre,
    publisher_largest_shift,
    sensitivity_agrees,
    sensitivity_change_pp,
    theme_declining,
    theme_rising,
):
    ai_export_frame = (
        ai_linked_frame
        .select(
            "record_id", "claim_text", "review_title", "claimant", "claim_date",
            "claim_dt", "review_date", "review_dt", "publisher_name", "publisher_site",
            "textual_rating", "review_url", "matched_keywords", "matched_ai_terms",
            "mentioned_modality", "matched_themes", "comparison_era", "balanced_window_era",
            "language_code",
        )
        .with_columns(
            pl.col("matched_keywords").list.join("|").alias("matched_keywords"),
            pl.col("matched_ai_terms").list.join("|").alias("matched_ai_terms"),
            pl.col("matched_themes").list.join("|").alias("matched_themes"),
        )
        .sort(["claim_dt", "review_url"], descending=[False, False], nulls_last=True)
    )
    ai_csv_bytes = ai_export_frame.write_csv().encode("utf-8")
    ai_csv_download = mo.download(
        data=ai_csv_bytes,
        filename="google_fact_check_generative_ai_linked_claims.csv",
        mimetype="text/csv",
        label=f"Download classified AI-linked CSV ({ai_export_frame.height:,} rows)",
    )
    ai_record_explorer = mo.ui.table(
        ai_linked_frame.select(
            "claim_dt", "comparison_era", "matched_ai_terms", "mentioned_modality",
            "matched_themes", "claim_text", "review_title", "publisher_site", "review_url",
        ).sort("claim_dt", nulls_last=True),
        selection=None,
        pagination=True,
        page_size=20,
        show_search=True,
        show_download=True,
        show_column_summaries=True,
        wrapped_columns=["claim_text", "review_title", "matched_themes"],
        freeze_columns_left=["claim_text"],
        label="Rule-confirmed AI-linked record explorer",
    )

    answer_section = mo.vstack([
        mo.accordion({
            "Detailed comparison design": method_explanation,
            "Analytical grains and date eligibility": frame_section,
        }, multiple=False),
        mo.callout(
            mo.md(
                "The CSV embeds the conservative AI classification, matched terms, "
                "modality, themes, dates, publisher fields, and source identifiers. "
                f"It was generated from `{EXTRACT_PATH.name}`."
            ),
            kind="info",
            title="Reproducible classified subset",
        ),
        ai_csv_download,
        ai_record_explorer,
    ])
    answer_section
    return


if __name__ == "__main__":
    app.run()
