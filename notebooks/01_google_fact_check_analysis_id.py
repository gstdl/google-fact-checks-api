import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def introduce_analysis(mo):
    mo.md("""
    # Indonesian fact-check temporal analysis

    Static-first analysis of the cached Google Fact Check Tools extract. This
    notebook never calls Google APIs and does not require cloud credentials; it
    reads `data/google_fact_check_tool_id.parquet` as its source of truth.

    Counts describe fact-check records surfaced by the extraction notebook's
    configured queries. They are not a census of misinformation in Indonesia.
    """)
    return


@app.cell(hide_code=True)
def explain_analysis_setup(mo):
    mo.md("""
    ## 1. Setup and analytical frames

    The cached Parquet is loaded directly. No network requests or environment
    variables are used. Because the extract contains keyword matches rather than
    independent observations, it is reduced into three explicit analytical grains:

    1. one keyword-to-reviewed-claim link for coverage and overlap;
    2. one `(review_url, claim_text)` record for temporal and publisher analysis;
    3. one non-empty claim text for NLP.

    Dates are parsed into separate UTC columns. Rating and claimant labels are
    only case/whitespace-normalized; raw values remain available.
    """)
    return


@app.cell
def setup_analysis_environment():
    import collections
    import itertools
    import mimetypes
    import re
    import time
    from datetime import datetime, timezone

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import Normalizer

    REPO_ROOT = mo.notebook_dir().parent
    EXTRACT_PATH = REPO_ROOT / "data" / "google_fact_check_tool_id.parquet"
    mimetypes.add_type("application/vnd.apache.parquet", ".parquet")

    if not EXTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EXTRACT_PATH}; run notebooks/00_google_fact_check_tool_id.py first."
        )

    return (
        EXTRACT_PATH,
        KMeans,
        NMF,
        Normalizer,
        StopWordRemoverFactory,
        TfidfVectorizer,
        TruncatedSVD,
        alt,
        collections,
        datetime,
        itertools,
        mo,
        np,
        pl,
        re,
        silhouette_score,
        time,
        timezone,
    )


@app.cell
def build_analytical_frames(
    EXTRACT_PATH,
    datetime,
    mo,
    pl,
    timezone,
):
    claims_frame = pl.read_parquet(EXTRACT_PATH)
    claims_source = f"parquet ({EXTRACT_PATH.name})"
    analysis_now_utc = datetime.now(timezone.utc)
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

    keyword_link_frame = (
        claims_frame
        .unique(subset=["keyword", *record_key_columns])
        .sort(["keyword", *record_key_columns])
    )

    record_conflict_frame = (
        claims_frame
        .group_by(record_key_columns)
        .agg([
            pl.col(field).drop_nulls().n_unique().alias(field)
            for field in metadata_fields
        ])
        .with_columns(
            pl.sum_horizontal([
                (pl.col(field) > 1).cast(pl.Int64)
                for field in metadata_fields
            ]).alias("conflict_count")
        )
        .filter(pl.col("conflict_count") > 0)
        .sort("conflict_count", descending=True)
    )

    metadata_conflict_summary = pl.DataFrame({
        "field": metadata_fields,
        "reviewed_claims_with_conflict": [
            record_conflict_frame.filter(pl.col(field) > 1).height
            for field in metadata_fields
        ],
    })

    review_frame = (
        claims_frame
        .sort([*record_key_columns, "keyword"])
        .group_by(record_key_columns, maintain_order=True)
        .agg(
            pl.col("keyword").unique().sort().alias("matched_keywords"),
            pl.col("keyword").n_unique().alias("keyword_count"),
            *[
                pl.col(field).drop_nulls().first().alias(field)
                for field in metadata_fields
            ],
        )
        .with_columns(
            pl.concat_str(record_key_columns, separator=" ⟂ ").alias("record_id"),
            pl.col("claim_date")
              .str.to_datetime(strict=False, time_zone="UTC")
              .alias("claim_dt"),
            pl.col("review_date")
              .str.to_datetime(strict=False, time_zone="UTC")
              .alias("review_dt"),
            pl.col("textual_rating")
              .str.to_lowercase()
              .str.strip_chars()
              .str.replace_all(r"\s+", " ")
              .alias("rating_normalized"),
            pl.col("claimant")
              .str.to_lowercase()
              .str.strip_chars()
              .str.replace_all(r"\s+", " ")
              .alias("claimant_normalized"),
            pl.col("claim_text").str.len_chars().alias("claim_chars"),
            pl.col("claim_text").str.count_matches(r"\b\w+\b").alias("claim_words"),
            pl.col("review_title").str.len_chars().alias("title_chars"),
            pl.col("review_title").str.count_matches(r"\b\w+\b").alias("title_words"),
            (
                pl.col("review_url").str.starts_with("https://") |
                pl.col("review_url").str.starts_with("http://")
            ).alias("valid_http_url"),
        )
        .with_columns(
            pl.col("claim_dt").dt.year().alias("claim_year"),
            pl.col("claim_dt").dt.month().alias("claim_month"),
            pl.col("claim_dt").dt.truncate("1mo").alias("claim_month_start"),
            pl.col("review_dt").dt.year().alias("review_year"),
            pl.col("review_dt").dt.truncate("1mo").alias("review_month_start"),
            (
                pl.col("claim_date").is_not_null() & pl.col("claim_dt").is_null()
            ).alias("claim_date_parse_failed"),
            (
                pl.col("review_date").is_not_null() & pl.col("review_dt").is_null()
            ).alias("review_date_parse_failed"),
            (pl.col("claim_dt") > analysis_now_utc).alias("future_claim_date"),
            (pl.col("review_dt") > analysis_now_utc).alias("future_review_date"),
            (pl.col("review_dt") - pl.col("claim_dt"))
              .dt.total_days()
              .alias("review_lag_days"),
        )
        .sort(record_key_columns)
    )

    claim_text_frame = (
        review_frame
        .filter(pl.col("claim_text").is_not_null() & (pl.col("claim_text").str.len_chars() > 0))
        .select("claim_text")
        .unique()
        .sort("claim_text")
    )

    metric_definitions = pl.DataFrame([
        {
            "frame": "claims_frame",
            "grain": "keyword × reviewed claim",
            "rows": claims_frame.height,
            "used_for": "source preservation and extraction audit",
        },
        {
            "frame": "keyword_link_frame",
            "grain": "distinct keyword × (review URL, claim text)",
            "rows": keyword_link_frame.height,
            "used_for": "keyword coverage, overlap, co-occurrence",
        },
        {
            "frame": "review_frame",
            "grain": "distinct (review URL, claim text)",
            "rows": review_frame.height,
            "used_for": "time, publisher, rating, claimant, text",
        },
        {
            "frame": "claim_text_frame",
            "grain": "distinct non-empty claim text",
            "rows": claim_text_frame.height,
            "used_for": "vocabulary, topics, latent clusters",
        },
    ])

    frame_section = mo.vstack([
        mo.md(
            f"The parquet yielded **{claims_frame.height:,} keyword links**, "
            f"**{review_frame.height:,} reviewed-claim records**, "
            f"**{review_frame['review_url'].n_unique():,} distinct URLs**, and "
            f"**{claim_text_frame.height:,} unique claim texts**. "
            f"There are **{record_conflict_frame.height:,}** reviewed-claim keys "
            "with at least one metadata conflict; these are audited in section 3."
        ),
        mo.ui.table(
            metric_definitions,
            selection=None,
            pagination=False,
            show_column_summaries=False,
        ),
    ])
    frame_section
    return (
        claim_text_frame,
        claims_frame,
        claims_source,
        keyword_link_frame,
        metadata_conflict_summary,
        metadata_fields,
        record_conflict_frame,
        record_key_columns,
        review_frame,
    )


@app.cell(hide_code=True)
def explain_executive_overview(mo):
    mo.md("""
    ## 2. Executive overview

    These headline metrics describe **fact-check records surfaced by the 112
    configured API queries**. They are not an estimate of how many hoaxes exist
    in Indonesia: publisher participation, Google indexing, query vocabulary,
    and missing dates all shape what appears here.
    """)
    return


@app.cell
def render_executive_overview(
    claim_text_frame,
    claims_frame,
    claims_source,
    keyword_link_frame,
    mo,
    pl,
    review_frame,
):
    executive_eligible_dates = review_frame.filter(
        pl.col("claim_dt").is_not_null() &
        ~pl.col("future_claim_date").fill_null(False)
    )
    executive_year_counts = (
        executive_eligible_dates
        .group_by("claim_year")
        .len(name="records")
        .sort("claim_year")
    )
    executive_publisher_counts = (
        review_frame
        .group_by("publisher_site")
        .len(name="records")
        .sort("records", descending=True)
    )
    executive_keyword_counts = (
        keyword_link_frame
        .group_by("keyword")
        .len(name="records")
        .sort("records", descending=True)
    )
    executive_top_publisher = executive_publisher_counts.row(0, named=True)
    executive_top_year = executive_year_counts.sort("records", descending=True).row(0, named=True)
    executive_top_keyword = executive_keyword_counts.row(0, named=True)
    executive_overlap_factor = claims_frame.height / review_frame.height
    executive_date_min = executive_eligible_dates["claim_dt"].min().date()
    executive_date_max = executive_eligible_dates["claim_dt"].max().date()

    executive_stats = mo.vstack([
        mo.hstack([
            mo.stat(f"{claims_frame.height:,}", label="Keyword-match rows", bordered=True),
            mo.stat(f"{review_frame.height:,}", label="Reviewed claims", bordered=True),
            mo.stat(f"{review_frame['review_url'].n_unique():,}", label="Distinct URLs", bordered=True),
            mo.stat(f"{claim_text_frame.height:,}", label="Distinct claim texts", bordered=True),
        ], widths="equal", wrap=True),
        mo.hstack([
            mo.stat(f"{keyword_link_frame['keyword'].n_unique():,}", label="Keywords", bordered=True),
            mo.stat(f"{review_frame['publisher_site'].n_unique():,}", label="Publisher sites", bordered=True),
            mo.stat(f"{executive_overlap_factor:.2f}×", label="Keyword overlap factor", bordered=True),
            mo.stat(f"{executive_date_min} → {executive_date_max}", label="Usable claim-date span", bordered=True),
        ], widths="equal", wrap=True),
    ])

    executive_section = mo.vstack([
        executive_stats,
        mo.callout(
            mo.md(
                f"**What stands out.** `{executive_top_publisher['publisher_site']}` "
                f"contributes the most reviewed-claim records "
                f"(**{executive_top_publisher['records']:,}**). The busiest observed "
                f"claim year is **{executive_top_year['claim_year']}** "
                f"(**{executive_top_year['records']:,}** dated records), while "
                f"`{executive_top_keyword['keyword']}` is the broadest query "
                f"(**{executive_top_keyword['records']:,}** links). Each reviewed "
                f"claim appears in **{executive_overlap_factor:.2f} keyword rows on "
                "average**, so the reviewed-claim grain—not raw matches—is the "
                "correct denominator for corpus summaries."
            ),
            kind="info",
            title=f"Current source: {claims_source}",
        ),
    ])
    executive_section
    return executive_overlap_factor, executive_publisher_counts


@app.cell(hide_code=True)
def explain_data_quality(mo):
    mo.md("""
    ## 3. Data quality and coverage

    Missingness is structural here: Google often omits claimant, claim date, or
    review date, and publishers expose different metadata. The extract also
    repeats metadata across keyword matches, which lets us test whether those
    repeated versions agree before reducing to reviewed-claim grain.

    A `review` status below is a diagnostic, not an automatic deletion. Raw
    values remain in the parquet and anomaly tables retain identifiers for
    investigation.
    """)
    return


@app.cell
def render_data_quality(
    alt,
    claims_frame,
    keyword_link_frame,
    metadata_conflict_summary,
    metadata_fields,
    mo,
    pl,
    record_conflict_frame,
    record_key_columns,
    review_frame,
):
    quality_dictionary_rows = []
    for quality_column in claims_frame.columns:
        quality_series = claims_frame[quality_column]
        quality_non_null = quality_series.len() - quality_series.null_count()
        quality_values = quality_series.drop_nulls()
        quality_example = "" if quality_values.is_empty() else str(quality_values[0])[:100]
        quality_dictionary_rows.append({
            "column": quality_column,
            "source_type": str(claims_frame.schema[quality_column]),
            "non_null": quality_non_null,
            "missing_pct": round(100 * (claims_frame.height - quality_non_null) / claims_frame.height, 2),
            "cardinality": quality_series.n_unique(),
            "example": quality_example,
        })
    quality_data_dictionary = pl.DataFrame(quality_dictionary_rows)

    quality_missingness = pl.DataFrame([
        {
            "field": quality_field,
            "missing": review_frame[quality_field].null_count(),
            "missing_pct": round(100 * review_frame[quality_field].null_count() / review_frame.height, 2),
        }
        for quality_field in metadata_fields
    ])

    quality_publisher_wide = (
        review_frame
        .group_by("publisher_site")
        .agg(
            pl.len().alias("records"),
            (100 * pl.col("claim_dt").is_not_null().sum() / pl.len()).round(2).alias("claim date"),
            (100 * pl.col("review_dt").is_not_null().sum() / pl.len()).round(2).alias("review date"),
            (100 * pl.col("claimant").is_not_null().sum() / pl.len()).round(2).alias("claimant"),
        )
        .sort("records", descending=True)
    )
    quality_publisher_order = quality_publisher_wide["publisher_site"].to_list()
    quality_publisher_completeness = pl.concat([
        quality_publisher_wide.select(
            "publisher_site", pl.lit(quality_metric).alias("field"),
            pl.col(quality_metric).alias("complete_pct")
        )
        for quality_metric in ["claim date", "review date", "claimant"]
    ])

    quality_overlap_distribution = (
        review_frame
        .group_by("keyword_count")
        .len(name="reviewed_claims")
        .sort("keyword_count")
    )
    quality_nonnegative_lags = review_frame.filter(pl.col("review_lag_days") >= 0)
    quality_lag_p99 = (
        float(quality_nonnegative_lags["review_lag_days"].quantile(0.99))
        if quality_nonnegative_lags.height else 0.0
    )
    quality_integrity_checks = pl.DataFrame([
        {"check": "non-Indonesian language rows", "count": review_frame.filter(pl.col("language_code") != "id").height},
        {"check": "invalid HTTP(S) review URLs", "count": review_frame.filter(~pl.col("valid_http_url")).height},
        {"check": "duplicate keyword links", "count": claims_frame.height - keyword_link_frame.height},
        {"check": "duplicate reviewed-claim keys after reduction", "count": review_frame.height - review_frame.unique(record_key_columns).height},
        {"check": "reviewed claims with metadata conflicts", "count": record_conflict_frame.height},
        {"check": "claim-date parse failures", "count": review_frame.filter(pl.col("claim_date_parse_failed")).height},
        {"check": "review-date parse failures", "count": review_frame.filter(pl.col("review_date_parse_failed")).height},
        {"check": "future claim dates", "count": review_frame.filter(pl.col("future_claim_date").fill_null(False)).height},
        {"check": "future review dates", "count": review_frame.filter(pl.col("future_review_date").fill_null(False)).height},
        {"check": "negative claim-to-review lags", "count": review_frame.filter(pl.col("review_lag_days") < 0).height},
        {"check": f"non-negative lags above p99 ({quality_lag_p99:.0f} days)", "count": review_frame.filter(pl.col("review_lag_days") > quality_lag_p99).height},
    ]).with_columns(
        pl.when(pl.col("count") == 0).then(pl.lit("pass")).otherwise(pl.lit("review")).alias("status")
    )

    quality_conflict_details = (
        record_conflict_frame
        .rename({field: f"{field}_variants" for field in metadata_fields})
        .join(
            review_frame.select(
                *record_key_columns, "publisher_site", "claim_date", "review_title"
            ),
            on=record_key_columns,
            how="left",
        )
        .select(
            "review_url", "claim_text", "publisher_site", "claim_date",
            "conflict_count", *[f"{field}_variants" for field in metadata_fields],
            "review_title",
        )
        .sort("conflict_count", descending=True)
    )
    quality_date_anomalies = (
        review_frame
        .filter(
            pl.col("claim_date_parse_failed") |
            pl.col("review_date_parse_failed") |
            pl.col("future_claim_date").fill_null(False) |
            pl.col("future_review_date").fill_null(False) |
            (pl.col("review_lag_days") < 0) |
            (pl.col("review_lag_days") > quality_lag_p99)
        )
        .select(
            "review_url", "claim_text", "publisher_site", "claim_date", "review_date",
            "review_lag_days", "claim_date_parse_failed", "review_date_parse_failed",
            "future_claim_date", "future_review_date",
        )
        .sort("review_lag_days", descending=True, nulls_last=True)
    )
    quality_identity_anomalies = (
        review_frame
        .filter((pl.col("language_code") != "id") | ~pl.col("valid_http_url"))
        .select("review_url", "claim_text", "publisher_site", "language_code", "valid_http_url")
    )

    quality_missing_chart = (
        alt.Chart(quality_missingness)
        .mark_bar(color="#E45756")
        .encode(
            x=alt.X("missing_pct:Q", title="Missing reviewed claims (%)"),
            y=alt.Y("field:N", title=None, sort="-x"),
            tooltip=["field:N", "missing:Q", alt.Tooltip("missing_pct:Q", format=".2f")],
        )
        .properties(width=430, height=230, title="Missingness at reviewed-claim grain")
    )
    quality_completeness_chart = (
        alt.Chart(quality_publisher_completeness)
        .mark_rect()
        .encode(
            x=alt.X("field:N", title=None),
            y=alt.Y("publisher_site:N", title=None, sort=quality_publisher_order),
            color=alt.Color("complete_pct:Q", title="Complete (%)", scale=alt.Scale(scheme="blues", domain=[0, 100])),
            tooltip=["publisher_site:N", "field:N", alt.Tooltip("complete_pct:Q", format=".2f")],
        )
        .properties(width=360, height=230, title="Metadata completeness by publisher")
    )
    quality_overlap_chart = (
        alt.Chart(quality_overlap_distribution)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("keyword_count:O", title="Matched keywords per reviewed claim"),
            y=alt.Y("reviewed_claims:Q", title="Reviewed claims"),
            tooltip=["keyword_count:O", "reviewed_claims:Q"],
        )
        .properties(width=330, height=230, title="Keyword overlap")
    )
    quality_top_missing = quality_missingness.sort("missing_pct", descending=True).row(0, named=True)
    quality_top_conflict = metadata_conflict_summary.sort("reviewed_claims_with_conflict", descending=True).row(0, named=True)

    quality_section = mo.vstack([
        mo.md(
            f"**{quality_top_missing['field']}** is the least complete reviewed-claim "
            f"field (**{quality_top_missing['missing_pct']:.2f}% missing**). "
            f"The most common repeated-metadata disagreement is "
            f"**{quality_top_conflict['field']}** "
            f"(**{quality_top_conflict['reviewed_claims_with_conflict']:,} keys**). "
            f"Of {review_frame.height:,} reviewed claims, "
            f"{review_frame.filter(pl.col('keyword_count') > 1).height:,} match more than one keyword."
        ),
        mo.hstack([
            quality_missing_chart,
            quality_completeness_chart,
            quality_overlap_chart,
        ], align="start", wrap=True),
        mo.md("### Integrity checks"),
        mo.ui.table(quality_integrity_checks, selection=None, pagination=False, show_column_summaries=False),
        mo.md("### Source data dictionary"),
        mo.ui.table(quality_data_dictionary, selection=None, pagination=True, page_size=15),
        mo.md("### Investigate anomalies"),
        mo.accordion({
            f"Metadata conflicts ({quality_conflict_details.height:,})": mo.ui.table(
                quality_conflict_details, selection=None, pagination=True, page_size=20,
                wrapped_columns=["claim_text", "review_title"],
            ),
            f"Date anomalies and extreme lags ({quality_date_anomalies.height:,})": mo.ui.table(
                quality_date_anomalies, selection=None, pagination=True, page_size=20,
                wrapped_columns=["claim_text"],
            ),
            f"Language or URL anomalies ({quality_identity_anomalies.height:,})": mo.ui.table(
                quality_identity_anomalies, selection=None, pagination=True, page_size=20,
                wrapped_columns=["claim_text"],
            ),
        }, multiple=False),
    ])
    quality_section
    return (
        quality_lag_p99,
        quality_nonnegative_lags,
        quality_overlap_chart,
        quality_top_conflict,
    )


@app.cell(hide_code=True)
def explain_temporal_patterns(mo):
    mo.md("""
    ## 4. Temporal patterns

    `claim_date` answers when the underlying claim was dated; `review_date`
    answers when the fact-check was published. They are never mixed. Primary
    trends use parseable, non-future claim dates. Publication coverage and lag
    use only the smaller subset with review dates.

    Counts describe records surfaced by this retrieval—not the incidence of
    misinformation. Changes can reflect indexing and publisher coverage as
    well as events in the world.
    """)
    return


@app.cell
def render_temporal_patterns(
    alt,
    executive_publisher_counts,
    mo,
    pl,
    quality_lag_p99,
    quality_nonnegative_lags,
    review_frame,
):
    temporal_claim_records = review_frame.filter(
        pl.col("claim_dt").is_not_null() &
        ~pl.col("future_claim_date").fill_null(False)
    )
    temporal_review_records = review_frame.filter(
        pl.col("review_dt").is_not_null() &
        ~pl.col("future_review_date").fill_null(False)
    )
    temporal_annual = (
        temporal_claim_records
        .group_by("claim_year")
        .len(name="records")
        .sort("claim_year")
    )
    temporal_monthly_observed = (
        temporal_claim_records
        .group_by("claim_month_start")
        .len(name="records")
        .sort("claim_month_start")
    )
    temporal_month_index = pl.DataFrame({
        "claim_month_start": pl.datetime_range(
            temporal_monthly_observed["claim_month_start"].min(),
            temporal_monthly_observed["claim_month_start"].max(),
            interval="1mo",
            eager=True,
            time_zone="UTC",
        )
    })
    temporal_monthly = (
        temporal_month_index
        .join(temporal_monthly_observed, on="claim_month_start", how="left")
        .with_columns(pl.col("records").fill_null(0))
        .with_columns(
            pl.col("records")
              .rolling_mean(window_size=6, min_samples=1)
              .round(2)
              .alias("six_month_mean")
        )
    )
    temporal_calendar = (
        temporal_claim_records
        .group_by("claim_year", "claim_month")
        .len(name="records")
        .sort("claim_year", "claim_month")
    )
    temporal_review_annual = (
        temporal_review_records
        .group_by("review_year")
        .len(name="records")
        .sort("review_year")
    )
    temporal_lag_complete = review_frame.filter(
        pl.col("claim_dt").is_not_null() & pl.col("review_dt").is_not_null()
    )
    temporal_lag_plotted = temporal_lag_complete.filter(
        (pl.col("review_lag_days") >= 0) &
        (pl.col("review_lag_days") <= quality_lag_p99)
    )
    temporal_lag_quantiles = pl.DataFrame([
        {
            "quantile": temporal_quantile_label,
            "lag_days": float(quality_nonnegative_lags["review_lag_days"].quantile(temporal_quantile)),
        }
        for temporal_quantile_label, temporal_quantile in [
            ("minimum", 0.0), ("p25", 0.25), ("median", 0.5),
            ("p75", 0.75), ("p90", 0.9), ("p95", 0.95),
            ("p99", 0.99), ("maximum", 1.0),
        ]
    ])
    temporal_top_publishers = executive_publisher_counts.head(6)["publisher_site"].to_list()
    temporal_publisher_year = (
        temporal_claim_records
        .with_columns(
            pl.when(pl.col("publisher_site").is_in(temporal_top_publishers))
              .then(pl.col("publisher_site"))
              .otherwise(pl.lit("Other"))
              .alias("publisher_group")
        )
        .group_by("claim_year", "publisher_group")
        .len(name="records")
        .sort("claim_year", "publisher_group")
    )

    _temporal_year_chart = (
        alt.Chart(temporal_annual)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("claim_year:O", title="Claim year"),
            y=alt.Y("records:Q", title="Reviewed claims"),
            tooltip=["claim_year:O", "records:Q"],
        )
        .properties(width=430, height=260, title="Reviewed claims by claim year")
    )
    _temporal_month_bars = (
        alt.Chart(temporal_monthly)
        .mark_bar(color="#9ECAE9", opacity=0.65)
        .encode(
            x=alt.X("claim_month_start:T", title="Claim month"),
            y=alt.Y("records:Q", title="Reviewed claims"),
            tooltip=[alt.Tooltip("claim_month_start:T", title="Month", format="%Y-%m"), "records:Q"],
        )
    )
    _temporal_month_line = (
        alt.Chart(temporal_monthly)
        .mark_line(color="#E45756", strokeWidth=2)
        .encode(
            x=alt.X("claim_month_start:T"),
            y=alt.Y("six_month_mean:Q", title="Reviewed claims"),
            tooltip=[alt.Tooltip("claim_month_start:T", title="Month", format="%Y-%m"), alt.Tooltip("six_month_mean:Q", title="6-month mean", format=".2f")],
        )
    )
    temporal_month_chart = (
        (_temporal_month_bars + _temporal_month_line)
        .properties(width=690, height=260, title="Monthly records and six-month rolling mean")
    )
    temporal_calendar_chart = (
        alt.Chart(temporal_calendar)
        .mark_rect()
        .encode(
            x=alt.X("claim_month:O", title="Month"),
            y=alt.Y("claim_year:O", title="Year"),
            color=alt.Color("records:Q", title="Records", scale=alt.Scale(scheme="blues")),
            tooltip=["claim_year:O", "claim_month:O", "records:Q"],
        )
        .properties(width=520, height=290, title="Year × month coverage")
    )
    temporal_review_chart = (
        alt.Chart(temporal_review_annual)
        .mark_bar(color="#72B7B2")
        .encode(
            x=alt.X("review_year:O", title="Review year"),
            y=alt.Y("records:Q", title="Reviewed claims with review date"),
            tooltip=["review_year:O", "records:Q"],
        )
        .properties(width=430, height=260, title="Publication dates (separate, sparse field)")
    )
    temporal_lag_chart = (
        alt.Chart(temporal_lag_plotted)
        .mark_bar(color="#F58518")
        .encode(
            x=alt.X("review_lag_days:Q", bin=alt.Bin(maxbins=40), title="Days from claim to review (0–p99)"),
            y=alt.Y("count():Q", title="Reviewed claims"),
            tooltip=[alt.Tooltip("count():Q", title="Records")],
        )
        .properties(width=520, height=260, title="Claim-to-review lag")
    )
    temporal_publisher_chart = (
        alt.Chart(temporal_publisher_year)
        .mark_area()
        .encode(
            x=alt.X("claim_year:O", title="Claim year"),
            y=alt.Y("records:Q", title="Reviewed claims", stack="zero"),
            color=alt.Color("publisher_group:N", title="Publisher"),
            tooltip=["claim_year:O", "publisher_group:N", "records:Q"],
        )
        .properties(width=690, height=300, title="Publisher composition by claim year (top six + Other)")
    )
    temporal_peak_month = temporal_monthly.sort("records", descending=True).row(0, named=True)
    temporal_review_completeness = 100 * temporal_review_records.height / review_frame.height
    temporal_negative_lags = temporal_lag_complete.filter(pl.col("review_lag_days") < 0).height
    temporal_above_p99 = temporal_lag_complete.filter(pl.col("review_lag_days") > quality_lag_p99).height

    _temporal_year_chart_named = _temporal_year_chart
    _temporal_quantile_table = mo.ui.table(
        temporal_lag_quantiles, selection=None, pagination=False, show_column_summaries=False
    )
    temporal_section = mo.vstack([
        mo.md(
            f"The peak observed month is **{temporal_peak_month['claim_month_start'].strftime('%B %Y')}** "
            f"with **{temporal_peak_month['records']:,}** reviewed claims. Only "
            f"**{temporal_review_completeness:.1f}%** of reviewed claims carry a review date. "
            f"Lag summaries use {temporal_lag_complete.height:,} complete date pairs; "
            f"{temporal_negative_lags:,} negative lags and {temporal_above_p99:,} values above "
            f"p99 ({quality_lag_p99:.0f} days) are reported but omitted from the histogram."
        ),
        mo.hstack([_temporal_year_chart_named, temporal_month_chart], align="start", wrap=True),
        mo.hstack([temporal_calendar_chart, temporal_review_chart], align="start", wrap=True),
        mo.hstack([temporal_lag_chart, _temporal_quantile_table], align="start", wrap=True),
        temporal_publisher_chart,
    ])
    temporal_section
    return temporal_claim_records, temporal_publisher_year


@app.cell(hide_code=True)
def explain_publisher_landscape(mo):
    mo.md("""
    ## 5. Publisher landscape

    Publisher totals use one row per reviewed claim. Site hostnames are kept
    separate—even when they appear related—because merging `suara.com` with
    `amp.suara.com`, or production and beta domains, would add an unverified
    identity rule.

    Coverage reflects the Google index and this notebook's query vocabulary,
    not each publisher's complete fact-check archive.
    """)
    return


@app.cell
def render_publisher_landscape(
    alt,
    mo,
    pl,
    review_frame,
    temporal_claim_records,
    temporal_publisher_year,
):
    publisher_summary = (
        review_frame
        .group_by("publisher_site")
        .agg(
            pl.len().alias("records"),
            pl.col("review_url").n_unique().alias("distinct_urls"),
            pl.col("claim_dt").min().alias("first_claim_date"),
            pl.col("claim_dt").max().alias("last_claim_date"),
            pl.col("review_dt").min().alias("first_review_date"),
            pl.col("review_dt").max().alias("last_review_date"),
            (100 * pl.col("claim_dt").is_not_null().sum() / pl.len()).round(2).alias("claim_date_complete_pct"),
            (100 * pl.col("review_dt").is_not_null().sum() / pl.len()).round(2).alias("review_date_complete_pct"),
            pl.col("rating_normalized").n_unique().alias("rating_labels"),
        )
        .with_columns((100 * pl.col("records") / review_frame.height).round(2).alias("share_pct"))
        .sort("records", descending=True)
    )
    publisher_year = (
        temporal_claim_records
        .group_by("claim_year", "publisher_site")
        .len(name="records")
        .sort("claim_year", "publisher_site")
    )
    publisher_year_share = (
        temporal_publisher_year
        .with_columns(
            (100 * pl.col("records") / pl.col("records").sum().over("claim_year"))
            .round(2)
            .alias("share_pct")
        )
    )
    publisher_order = publisher_summary["publisher_site"].to_list()

    publisher_rank_chart = (
        alt.Chart(publisher_summary)
        .mark_bar()
        .encode(
            x=alt.X("records:Q", title="Reviewed claims"),
            y=alt.Y("publisher_site:N", title=None, sort=publisher_order),
            color=alt.Color("share_pct:Q", title="Share (%)", scale=alt.Scale(scheme="teals")),
            tooltip=["publisher_site:N", "records:Q", alt.Tooltip("share_pct:Q", format=".2f"), "distinct_urls:Q"],
        )
        .properties(width=520, height=300, title="Reviewed claims by publisher site")
    )
    publisher_year_heatmap = (
        alt.Chart(publisher_year)
        .mark_rect()
        .encode(
            x=alt.X("claim_year:O", title="Claim year"),
            y=alt.Y("publisher_site:N", title=None, sort=publisher_order),
            color=alt.Color("records:Q", title="Records", scale=alt.Scale(scheme="blues")),
            tooltip=["publisher_site:N", "claim_year:O", "records:Q"],
        )
        .properties(width=660, height=300, title="Publisher × claim year")
    )
    publisher_share_chart = (
        alt.Chart(publisher_year_share)
        .mark_line(point=True)
        .encode(
            x=alt.X("claim_year:O", title="Claim year"),
            y=alt.Y("share_pct:Q", title="Share of dated records (%)"),
            color=alt.Color("publisher_group:N", title="Publisher"),
            tooltip=["claim_year:O", "publisher_group:N", alt.Tooltip("share_pct:Q", format=".2f")],
        )
        .properties(width=780, height=320, title="Publisher share over time (top six + Other)")
    )
    publisher_top = publisher_summary.row(0, named=True)
    publisher_top_share = publisher_top["share_pct"]
    publisher_review_dates_best = publisher_summary.sort("review_date_complete_pct", descending=True).row(0, named=True)

    publisher_section = mo.vstack([
        mo.md(
            f"`{publisher_top['publisher_site']}` supplies **{publisher_top['records']:,} "
            f"records ({publisher_top_share:.2f}% of the reviewed-claim corpus)**. "
            f"`{publisher_review_dates_best['publisher_site']}` has the strongest review-date "
            f"coverage ({publisher_review_dates_best['review_date_complete_pct']:.2f}%). "
            "Large share shifts should be read alongside each site's coverage window."
        ),
        mo.hstack([publisher_rank_chart, publisher_year_heatmap], align="start", wrap=True),
        publisher_share_chart,
        mo.ui.table(
            publisher_summary,
            selection=None,
            pagination=True,
            page_size=15,
            show_column_summaries=False,
        ),
    ])
    publisher_section
    return publisher_order, publisher_summary


@app.cell(hide_code=True)
def explain_keyword_overlap(mo):
    mo.md("""
    ## 6. Keyword coverage and overlap

    Keywords are retrieval lenses, not mutually exclusive topics. A single
    reviewed claim can match up to several terms, so counts below use distinct
    keyword-to-reviewed-claim links and overlap is shown explicitly.

    Jaccard similarity divides shared records by the union of two keywords'
    records. This prevents the broadest query from looking related to everything
    purely because it is frequent.
    """)
    return


@app.cell
def render_keyword_overlap(
    alt,
    collections,
    itertools,
    keyword_link_frame,
    mo,
    pl,
    quality_overlap_chart,
    record_key_columns,
    review_frame,
):
    keyword_enriched_links = keyword_link_frame.join(
        review_frame.select(
            *record_key_columns, "claim_dt", "claim_year", "future_claim_date", "keyword_count"
        ),
        on=record_key_columns,
        how="left",
    )
    keyword_metrics = (
        keyword_enriched_links
        .group_by("keyword")
        .agg(
            pl.len().alias("records"),
            pl.col("claim_dt").filter(~pl.col("future_claim_date").fill_null(False)).min().alias("first_claim_date"),
            pl.col("claim_dt").filter(~pl.col("future_claim_date").fill_null(False)).max().alias("last_claim_date"),
            pl.col("keyword_count").median().alias("median_keywords_per_record"),
        )
        .with_columns((100 * pl.col("records") / review_frame.height).round(2).alias("corpus_coverage_pct"))
        .sort("records", descending=True)
    )
    keyword_order = keyword_metrics["keyword"].to_list()
    keyword_size_map = dict(zip(keyword_metrics["keyword"].to_list(), keyword_metrics["records"].to_list()))
    keyword_top25 = keyword_order[:25]
    keyword_top25_set = set(keyword_top25)
    keyword_pair_counts = collections.Counter()
    for keyword_values in review_frame["matched_keywords"].to_list():
        keyword_present = sorted(keyword_top25_set.intersection(keyword_values))
        for keyword_a_value, keyword_b_value in itertools.combinations(keyword_present, 2):
            keyword_pair_counts[(keyword_a_value, keyword_b_value)] += 1

    keyword_jaccard_rows = []
    for keyword_a_value in keyword_top25:
        for keyword_b_value in keyword_top25:
            if keyword_a_value == keyword_b_value:
                keyword_shared = keyword_size_map[keyword_a_value]
            else:
                keyword_shared = keyword_pair_counts[tuple(sorted((keyword_a_value, keyword_b_value)))]
            keyword_union = keyword_size_map[keyword_a_value] + keyword_size_map[keyword_b_value] - keyword_shared
            keyword_jaccard_rows.append({
                "keyword_a": keyword_a_value,
                "keyword_b": keyword_b_value,
                "shared_records": keyword_shared,
                "union_records": keyword_union,
                "jaccard": keyword_shared / keyword_union if keyword_union else 0.0,
            })
    keyword_jaccard = pl.DataFrame(keyword_jaccard_rows)
    keyword_top_pair = (
        keyword_jaccard
        .filter(pl.col("keyword_a") < pl.col("keyword_b"))
        .sort(["jaccard", "shared_records"], descending=True)
        .row(0, named=True)
    )

    keyword_trend_top = keyword_order[:20]
    keyword_year_top = (
        keyword_enriched_links
        .filter(
            pl.col("keyword").is_in(keyword_trend_top) &
            pl.col("claim_dt").is_not_null() &
            ~pl.col("future_claim_date").fill_null(False)
        )
        .group_by("keyword", "claim_year")
        .len(name="records")
        .with_columns(
            (100 * pl.col("records") / pl.col("records").sum().over("keyword"))
            .round(2)
            .alias("within_keyword_pct")
        )
        .sort("keyword", "claim_year")
    )

    keyword_bar_chart = (
        alt.Chart(keyword_metrics)
        .mark_bar()
        .encode(
            x=alt.X("records:Q", title="Distinct reviewed-claim links"),
            y=alt.Y("keyword:N", title="Keyword", sort=keyword_order),
            color=alt.Color("records:Q", title="Records", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["keyword:N", "records:Q", alt.Tooltip("corpus_coverage_pct:Q", format=".2f"), "first_claim_date:T", "last_claim_date:T"],
        )
        .properties(width=650, height=alt.Step(14), title="Coverage of all configured keywords")
    )
    keyword_jaccard_chart = (
        alt.Chart(keyword_jaccard)
        .mark_rect()
        .encode(
            x=alt.X("keyword_a:N", title=None, sort=keyword_top25),
            y=alt.Y("keyword_b:N", title=None, sort=keyword_top25),
            color=alt.Color("jaccard:Q", title="Jaccard", scale=alt.Scale(scheme="viridis", domain=[0, 1])),
            tooltip=["keyword_a:N", "keyword_b:N", "shared_records:Q", "union_records:Q", alt.Tooltip("jaccard:Q", format=".3f")],
        )
        .properties(width=620, height=620, title="Top-25 keyword co-occurrence (Jaccard)")
    )
    keyword_year_chart = (
        alt.Chart(keyword_year_top)
        .mark_rect()
        .encode(
            x=alt.X("claim_year:O", title="Claim year"),
            y=alt.Y("keyword:N", title=None, sort=keyword_trend_top),
            color=alt.Color("within_keyword_pct:Q", title="Within keyword (%)", scale=alt.Scale(scheme="blues")),
            tooltip=["keyword:N", "claim_year:O", "records:Q", alt.Tooltip("within_keyword_pct:Q", format=".2f")],
        )
        .properties(width=720, height=440, title="When each top keyword's dated records occur")
    )
    keyword_multi_record_count = review_frame.filter(pl.col("keyword_count") > 1).height
    keyword_multi_record_pct = 100 * keyword_multi_record_count / review_frame.height
    keyword_top = keyword_metrics.row(0, named=True)

    keyword_section = mo.vstack([
        mo.md(
            f"`{keyword_top['keyword']}` covers the most reviewed claims "
            f"(**{keyword_top['records']:,}**, {keyword_top['corpus_coverage_pct']:.2f}% of the corpus). "
            f"**{keyword_multi_record_count:,} records ({keyword_multi_record_pct:.1f}%)** match "
            f"multiple configured terms. The strongest top-25 Jaccard pair is "
            f"`{keyword_top_pair['keyword_a']}` ↔ `{keyword_top_pair['keyword_b']}` "
            f"(**{keyword_top_pair['jaccard']:.3f}**, {keyword_top_pair['shared_records']:,} shared records)."
        ),
        mo.hstack([quality_overlap_chart, keyword_year_chart], align="start", wrap=True),
        mo.hstack([keyword_bar_chart, keyword_jaccard_chart], align="start", wrap=True),
        mo.md("### Full keyword metrics"),
        mo.ui.table(keyword_metrics, selection=None, pagination=True, page_size=25, show_column_summaries=False),
    ])
    keyword_section
    return (keyword_multi_record_pct,)


@app.cell(hide_code=True)
def explain_ratings_and_claimants(mo):
    mo.md("""
    ## 7. Ratings and claimants

    Rating labels are publisher-authored taxonomies. Only case and repeated
    whitespace are normalized; labels are not mapped into invented universal
    categories or treated as sentiment. The publisher heatmap therefore shows
    each label's share of that publisher's own records.

    Claimants receive the same conservative normalization. Missing claimant
    values remain missing rather than becoming an “unknown actor” category.
    """)
    return


@app.cell
def render_ratings_and_claimants(
    alt,
    mo,
    pl,
    publisher_order,
    review_frame,
    temporal_claim_records,
):
    rating_summary = (
        review_frame
        .group_by("rating_normalized")
        .agg(
            pl.len().alias("records"),
            pl.col("textual_rating").n_unique().alias("raw_variants"),
            pl.col("textual_rating").unique().sort().alias("raw_labels"),
        )
        .with_columns((100 * pl.col("records") / review_frame.height).round(2).alias("share_pct"))
        .sort("records", descending=True)
    )
    rating_top_labels = rating_summary.head(12)["rating_normalized"].to_list()
    rating_publisher_totals = review_frame.group_by("publisher_site").len(name="publisher_records")
    rating_publisher_matrix = (
        review_frame
        .filter(pl.col("rating_normalized").is_in(rating_top_labels))
        .group_by("publisher_site", "rating_normalized")
        .len(name="records")
        .join(rating_publisher_totals, on="publisher_site", how="left")
        .with_columns((100 * pl.col("records") / pl.col("publisher_records")).round(2).alias("publisher_share_pct"))
    )

    claimant_missing_count = review_frame["claimant"].null_count()
    claimant_summary = (
        review_frame
        .filter(pl.col("claimant_normalized").is_not_null() & (pl.col("claimant_normalized").str.len_chars() > 0))
        .group_by("claimant_normalized")
        .agg(
            pl.len().alias("records"),
            pl.col("claimant").n_unique().alias("raw_variants"),
            pl.col("claimant").unique().sort().alias("raw_labels"),
            pl.col("claim_dt").min().alias("first_claim_date"),
            pl.col("claim_dt").max().alias("last_claim_date"),
        )
        .sort("records", descending=True)
    )
    claimant_top25 = claimant_summary.head(25)
    claimant_top10_labels = claimant_summary.head(10)["claimant_normalized"].to_list()
    claimant_year = (
        temporal_claim_records
        .filter(pl.col("claimant_normalized").is_in(claimant_top10_labels))
        .group_by("claim_year", "claimant_normalized")
        .len(name="records")
        .sort("claim_year", "claimant_normalized")
    )

    rating_bar_chart = (
        alt.Chart(rating_summary.head(20))
        .mark_bar(color="#E45756")
        .encode(
            x=alt.X("records:Q", title="Reviewed claims"),
            y=alt.Y("rating_normalized:N", title=None, sort="-x"),
            tooltip=["rating_normalized:N", "records:Q", alt.Tooltip("share_pct:Q", format=".2f"), "raw_variants:Q"],
        )
        .properties(width=520, height=420, title="Most frequent normalized rating labels")
    )
    rating_publisher_chart = (
        alt.Chart(rating_publisher_matrix)
        .mark_rect()
        .encode(
            x=alt.X("rating_normalized:N", title="Normalized rating", sort=rating_top_labels),
            y=alt.Y("publisher_site:N", title=None, sort=publisher_order),
            color=alt.Color("publisher_share_pct:Q", title="Publisher share (%)", scale=alt.Scale(scheme="oranges")),
            tooltip=["publisher_site:N", "rating_normalized:N", "records:Q", alt.Tooltip("publisher_share_pct:Q", format=".2f")],
        )
        .properties(width=700, height=300, title="Publisher-specific use of top rating labels")
    )
    claimant_bar_chart = (
        alt.Chart(claimant_top25)
        .mark_bar(color="#54A24B")
        .encode(
            x=alt.X("records:Q", title="Reviewed claims"),
            y=alt.Y("claimant_normalized:N", title=None, sort="-x"),
            tooltip=["claimant_normalized:N", "records:Q", "raw_variants:Q", "first_claim_date:T", "last_claim_date:T"],
        )
        .properties(width=520, height=500, title="Top normalized claimants")
    )
    claimant_year_chart = (
        alt.Chart(claimant_year)
        .mark_line(point=True)
        .encode(
            x=alt.X("claim_year:O", title="Claim year"),
            y=alt.Y("records:Q", title="Reviewed claims"),
            color=alt.Color("claimant_normalized:N", title="Claimant"),
            tooltip=["claim_year:O", "claimant_normalized:N", "records:Q"],
        )
        .properties(width=720, height=330, title="Top claimant records over time")
    )
    rating_top = rating_summary.row(0, named=True)
    claimant_top = claimant_summary.row(0, named=True)
    claimant_missing_pct = 100 * claimant_missing_count / review_frame.height

    rating_claimant_section = mo.vstack([
        mo.md(
            f"The most frequent normalized rating is **{rating_top['rating_normalized']}** "
            f"({rating_top['records']:,} records; {rating_top['raw_variants']} raw spelling/case variants). "
            f"Claimant is missing for **{claimant_missing_count:,} records ({claimant_missing_pct:.1f}%)**; "
            f"among present values, **{claimant_top['claimant_normalized']}** appears most often "
            f"({claimant_top['records']:,} records)."
        ),
        mo.hstack([rating_bar_chart, rating_publisher_chart], align="start", wrap=True),
        mo.md("### Rating variant audit"),
        mo.ui.table(rating_summary, selection=None, pagination=True, page_size=20, wrapped_columns=["raw_labels"]),
        mo.hstack([claimant_bar_chart, claimant_year_chart], align="start", wrap=True),
        mo.md("### Claimant variant audit"),
        mo.ui.table(claimant_summary, selection=None, pagination=True, page_size=25, wrapped_columns=["raw_labels"]),
    ])
    rating_claimant_section
    return


@app.cell(hide_code=True)
def explain_text_vocabulary(mo):
    mo.md("""
    ## 8. Text shape and vocabulary

    Linguistic analysis uses one copy of each non-empty `claim_text`, so a claim
    reviewed by several publishers does not dominate the model. Original text
    is never altered in the source frame.

    Tokenization is offline and Indonesian-aware: Sastrawi supplies function
    words, while a short documented list removes fact-check boilerplate such as
    “klaim”, “cek”, and “fakta”. Bars show **document frequency**—the number of
    unique claim texts containing a term—not raw repetition. Word clouds are
    deliberately avoided because area is a poor quantitative encoding.
    """)
    return


@app.cell
def analyze_text_vocabulary(
    StopWordRemoverFactory,
    TfidfVectorizer,
    alt,
    claim_text_frame,
    mo,
    np,
    pl,
    review_frame,
    time,
):
    NLP_BOILERPLATE_STOPWORDS = {
        "akun", "artikel", "berdasarkan", "benar", "benarkah", "beredar",
        "berisi", "cek", "diklaim", "disebut", "disebutkan", "fakta",
        "faktanya", "hoaks", "hoax", "informasi", "kabar", "keliru",
        "klaim", "mengklaim", "menyesatkan", "narasi", "palsu",
        "postingan", "salah", "sebagian", "unggahan",
    }
    nlp_stopwords = sorted(
        set(StopWordRemoverFactory().get_stop_words()) |
        NLP_BOILERPLATE_STOPWORDS
    )
    nlp_texts = claim_text_frame["claim_text"].to_list()
    nlp_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=nlp_stopwords,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.90,
        max_features=8_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][\w-]{2,}\b",
        dtype=np.float64,
    )
    nlp_tfidf_started = time.perf_counter()
    nlp_tfidf = nlp_vectorizer.fit_transform(nlp_texts)
    nlp_tfidf_seconds = time.perf_counter() - nlp_tfidf_started
    nlp_feature_names = np.asarray(nlp_vectorizer.get_feature_names_out())
    nlp_document_frequency_values = np.asarray((nlp_tfidf > 0).sum(axis=0)).ravel()
    nlp_vocabulary = (
        pl.DataFrame({
            "term": nlp_feature_names.tolist(),
            "documents": nlp_document_frequency_values.tolist(),
        })
        .with_columns(
            pl.when(pl.col("term").str.contains(" "))
              .then(pl.lit("bigram"))
              .otherwise(pl.lit("unigram"))
              .alias("term_type")
        )
        .sort("documents", descending=True)
    )
    nlp_top_unigrams = nlp_vocabulary.filter(pl.col("term_type") == "unigram").head(25)
    nlp_top_bigrams = nlp_vocabulary.filter(pl.col("term_type") == "bigram").head(25)

    text_claim_profile = (
        claim_text_frame
        .with_columns(
            pl.col("claim_text").str.len_chars().alias("characters"),
            pl.col("claim_text").str.count_matches(r"\b\w+\b").alias("words"),
        )
    )
    text_title_profile = (
        review_frame
        .select("review_url", "review_title")
        .with_columns(
            pl.col("review_title").str.len_chars().alias("characters"),
            pl.col("review_title").str.count_matches(r"\b\w+\b").alias("words"),
        )
    )
    text_quantile_rows = []
    for text_measure_name, text_measure_series in [
        ("claim characters", text_claim_profile["characters"]),
        ("claim words", text_claim_profile["words"]),
        ("title characters", text_title_profile["characters"]),
        ("title words", text_title_profile["words"]),
    ]:
        for text_quantile_label, text_quantile_value in [
            ("minimum", 0.0), ("p25", 0.25), ("median", 0.5),
            ("p75", 0.75), ("p90", 0.9), ("p95", 0.95),
            ("p99", 0.99), ("maximum", 1.0),
        ]:
            text_quantile_rows.append({
                "measure": text_measure_name,
                "quantile": text_quantile_label,
                "value": float(text_measure_series.quantile(text_quantile_value)),
            })
    text_quantiles = pl.DataFrame(text_quantile_rows)

    def _histogram_frame(values, measure, bins=40):
        counts, edges = np.histogram(np.asarray(values), bins=bins)
        return pl.DataFrame({
            "measure": [measure] * len(counts),
            "bin_start": edges[:-1].tolist(),
            "bin_end": edges[1:].tolist(),
            "records": counts.tolist(),
        })

    text_histograms = pl.concat([
        _histogram_frame(text_claim_profile["characters"].to_numpy(), "claim characters"),
        _histogram_frame(text_claim_profile["words"].to_numpy(), "claim words"),
        _histogram_frame(text_title_profile["characters"].to_numpy(), "title characters"),
        _histogram_frame(text_title_profile["words"].to_numpy(), "title words"),
    ])
    text_claim_outliers = text_claim_profile.sort("characters", descending=True).head(15)
    text_title_outliers = text_title_profile.sort("characters", descending=True).head(15)

    text_histogram_chart = (
        alt.Chart(text_histograms)
        .mark_bar(color="#4C78A8")
        .encode(
            x=alt.X("bin_start:Q", title="Value"),
            x2="bin_end:Q",
            y=alt.Y("records:Q", title="Texts"),
            facet=alt.Facet("measure:N", columns=2, title=None),
            tooltip=["measure:N", alt.Tooltip("bin_start:Q", format=".1f"), alt.Tooltip("bin_end:Q", format=".1f"), "records:Q"],
        )
        .properties(width=430, height=180, title="Text-length distributions")
    )
    text_unigram_chart = (
        alt.Chart(nlp_top_unigrams)
        .mark_bar(color="#59A14F")
        .encode(
            x=alt.X("documents:Q", title="Unique claim texts"),
            y=alt.Y("term:N", title=None, sort="-x"),
            tooltip=["term:N", "documents:Q"],
        )
        .properties(width=470, height=480, title="Top unigrams by document frequency")
    )
    text_bigram_chart = (
        alt.Chart(nlp_top_bigrams)
        .mark_bar(color="#B279A2")
        .encode(
            x=alt.X("documents:Q", title="Unique claim texts"),
            y=alt.Y("term:N", title=None, sort="-x"),
            tooltip=["term:N", "documents:Q"],
        )
        .properties(width=470, height=480, title="Top bigrams by document frequency")
    )
    text_top_unigram = nlp_top_unigrams.row(0, named=True)
    text_top_bigram = nlp_top_bigrams.row(0, named=True)

    text_section = mo.vstack([
        mo.md(
            f"The TF-IDF matrix contains **{nlp_tfidf.shape[0]:,} texts × "
            f"{nlp_tfidf.shape[1]:,} features** ({nlp_tfidf.nnz:,} non-zero weights) "
            f"and builds in **{nlp_tfidf_seconds:.2f}s**. The most widespread retained "
            f"unigram is **{text_top_unigram['term']}** "
            f"({text_top_unigram['documents']:,} texts); the leading bigram is "
            f"**{text_top_bigram['term']}** ({text_top_bigram['documents']:,} texts)."
        ),
        text_histogram_chart,
        mo.ui.table(text_quantiles, selection=None, pagination=False, show_column_summaries=False),
        mo.hstack([text_unigram_chart, text_bigram_chart], align="start", wrap=True),
        mo.accordion({
            "Longest claim texts": mo.ui.table(
                text_claim_outliers, selection=None, pagination=True, page_size=15,
                wrapped_columns=["claim_text"],
            ),
            "Longest review titles": mo.ui.table(
                text_title_outliers, selection=None, pagination=True, page_size=15,
                wrapped_columns=["review_title"],
            ),
            "Stopword audit": mo.ui.table(
                pl.DataFrame({"stopword": nlp_stopwords, "source": [
                    "fact-check boilerplate" if word in NLP_BOILERPLATE_STOPWORDS else "Sastrawi"
                    for word in nlp_stopwords
                ]}),
                selection=None, pagination=True, page_size=25,
            ),
        }),
    ])
    text_section
    return nlp_feature_names, nlp_stopwords, nlp_texts, nlp_tfidf


@app.cell(hide_code=True)
def explain_topic_model(mo):
    mo.md("""
    ## 9. Topic modeling

    Non-negative matrix factorization decomposes TF-IDF into 12 recurring term
    patterns. Labels are mechanical—the top three weighted features—not human
    interpretations. Each unique claim text receives one dominant topic, then
    that assignment is joined back to reviewed claims for time and publisher
    summaries.

    Topic IDs are canonically reordered by prevalence and label after fitting,
    so deterministic runs do not expose arbitrary component numbering.
    """)
    return


@app.cell
def fit_topic_model(
    NMF,
    alt,
    claim_text_frame,
    mo,
    nlp_feature_names,
    nlp_texts,
    nlp_tfidf,
    np,
    pl,
    publisher_order,
    review_frame,
    time,
):
    topic_started = time.perf_counter()
    topic_model = NMF(
        n_components=12,
        init="nndsvda",
        random_state=42,
        max_iter=300,
    )
    topic_raw_weights = topic_model.fit_transform(nlp_tfidf)
    topic_raw_components = topic_model.components_
    topic_raw_assignments = topic_raw_weights.argmax(axis=1)
    topic_raw_counts = np.bincount(topic_raw_assignments, minlength=12)
    topic_raw_labels = [
        " · ".join(nlp_feature_names[component.argsort()[-3:][::-1]])
        for component in topic_raw_components
    ]
    topic_canonical_order = sorted(
        range(12),
        key=lambda topic_raw_id: (-int(topic_raw_counts[topic_raw_id]), topic_raw_labels[topic_raw_id]),
    )
    topic_weights = topic_raw_weights[:, topic_canonical_order]
    topic_components = topic_raw_components[topic_canonical_order]
    topic_labels = [
        " · ".join(nlp_feature_names[component.argsort()[-3:][::-1]])
        for component in topic_components
    ]
    topic_assignments = topic_weights.argmax(axis=1) + 1
    topic_confidence = topic_weights.max(axis=1) / np.maximum(topic_weights.sum(axis=1), 1e-12)
    topic_seconds = time.perf_counter() - topic_started

    topic_text_assignments = pl.DataFrame({
        "claim_text": nlp_texts,
        "topic_id": topic_assignments.tolist(),
        "topic_label": [topic_labels[topic_id - 1] for topic_id in topic_assignments],
        "topic_weight": topic_weights.max(axis=1).tolist(),
        "topic_confidence": topic_confidence.tolist(),
    })
    topic_summary = (
        topic_text_assignments
        .group_by("topic_id", "topic_label")
        .agg(
            pl.len().alias("claim_texts"),
            pl.col("topic_confidence").median().round(4).alias("median_confidence"),
        )
        .with_columns((100 * pl.col("claim_texts") / claim_text_frame.height).round(2).alias("share_pct"))
        .sort("topic_id")
    )
    topic_term_rows = []
    for topic_index, topic_component in enumerate(topic_components, start=1):
        for topic_rank, topic_feature_index in enumerate(topic_component.argsort()[-10:][::-1], start=1):
            topic_term_rows.append({
                "topic_id": topic_index,
                "topic_label": topic_labels[topic_index - 1],
                "rank": topic_rank,
                "term": nlp_feature_names[topic_feature_index],
                "weight": float(topic_component[topic_feature_index]),
            })
    topic_terms = pl.DataFrame(topic_term_rows)
    topic_representatives = (
        topic_text_assignments
        .sort(["topic_id", "topic_weight", "claim_text"], descending=[False, True, False])
        .group_by("topic_id", maintain_order=True)
        .head(3)
        .select("topic_id", "topic_label", "topic_weight", "topic_confidence", "claim_text")
    )
    topic_review_frame = review_frame.join(topic_text_assignments, on="claim_text", how="left")
    topic_year = (
        topic_review_frame
        .filter(
            pl.col("claim_dt").is_not_null() &
            ~pl.col("future_claim_date").fill_null(False)
        )
        .group_by("topic_id", "topic_label", "claim_year")
        .len(name="records")
        .with_columns(
            (100 * pl.col("records") / pl.col("records").sum().over("claim_year"))
            .round(2)
            .alias("year_share_pct")
        )
    )
    topic_publisher = (
        topic_review_frame
        .group_by("topic_id", "topic_label", "publisher_site")
        .len(name="records")
        .with_columns(
            (100 * pl.col("records") / pl.col("records").sum().over("publisher_site"))
            .round(2)
            .alias("publisher_share_pct")
        )
    )

    topic_prevalence_chart = (
        alt.Chart(topic_summary)
        .mark_bar()
        .encode(
            x=alt.X("claim_texts:Q", title="Unique claim texts"),
            y=alt.Y("topic_label:N", title=None, sort=topic_labels),
            color=alt.Color("topic_id:O", title="Topic", scale=alt.Scale(scheme="tableau20"), legend=None),
            tooltip=["topic_id:O", "topic_label:N", "claim_texts:Q", alt.Tooltip("share_pct:Q", format=".2f"), alt.Tooltip("median_confidence:Q", format=".4f")],
        )
        .properties(width=620, height=360, title="Dominant NMF topic prevalence")
    )
    topic_year_chart = (
        alt.Chart(topic_year)
        .mark_rect()
        .encode(
            x=alt.X("claim_year:O", title="Claim year"),
            y=alt.Y("topic_label:N", title=None, sort=topic_labels),
            color=alt.Color("year_share_pct:Q", title="Year share (%)", scale=alt.Scale(scheme="blues")),
            tooltip=["topic_id:O", "topic_label:N", "claim_year:O", "records:Q", alt.Tooltip("year_share_pct:Q", format=".2f")],
        )
        .properties(width=700, height=360, title="Topic composition within each claim year")
    )
    topic_publisher_chart = (
        alt.Chart(topic_publisher)
        .mark_rect()
        .encode(
            x=alt.X("publisher_site:N", title="Publisher", sort=publisher_order),
            y=alt.Y("topic_label:N", title=None, sort=topic_labels),
            color=alt.Color("publisher_share_pct:Q", title="Publisher share (%)", scale=alt.Scale(scheme="purples")),
            tooltip=["publisher_site:N", "topic_id:O", "topic_label:N", "records:Q", alt.Tooltip("publisher_share_pct:Q", format=".2f")],
        )
        .properties(width=760, height=360, title="Topic composition within each publisher")
    )
    topic_largest = topic_summary.sort("claim_texts", descending=True).row(0, named=True)

    topic_section = mo.vstack([
        mo.md(
            f"NMF converged in **{topic_model.n_iter_} iterations / {topic_seconds:.2f}s**. "
            f"The largest mechanical topic is **{topic_largest['topic_label']}** "
            f"({topic_largest['claim_texts']:,} unique texts; {topic_largest['share_pct']:.2f}%). "
            "A label names high-weight terms only; representative claims are required for interpretation."
        ),
        mo.hstack([topic_prevalence_chart, topic_year_chart], align="start", wrap=True),
        topic_publisher_chart,
        mo.md("### Top terms and representative claims"),
        mo.hstack([
            mo.ui.table(topic_terms, selection=None, pagination=True, page_size=20),
            mo.ui.table(topic_representatives, selection=None, pagination=True, page_size=18, wrapped_columns=["claim_text"]),
        ], align="start", widths="equal", wrap=True),
    ])
    topic_section
    return topic_review_frame, topic_summary


@app.cell(hide_code=True)
def explain_semantic_clusters(mo):
    mo.md("""
    ## 10. Latent semantic clusters

    Clustering asks a different question from NMF topics: which claim texts are
    close in a 50-dimensional latent semantic representation? The notebook
    evaluates four bounded K-means candidates and chooses the highest silhouette
    score on a fixed sample, then refits that `k` on all texts.

    The scatter plot is only the first two latent dimensions and displays a
    deterministic sample. Distance on screen is neither geographic nor causal,
    and nearby points are not proof that claims share an origin.
    """)
    return


@app.cell
def fit_semantic_clusters(
    KMeans,
    Normalizer,
    TruncatedSVD,
    alt,
    claim_text_frame,
    mo,
    nlp_feature_names,
    nlp_texts,
    nlp_tfidf,
    np,
    pl,
    review_frame,
    silhouette_score,
    time,
):
    cluster_started = time.perf_counter()
    cluster_svd = TruncatedSVD(n_components=50, random_state=42)
    cluster_lsa_raw = cluster_svd.fit_transform(nlp_tfidf)
    cluster_lsa = Normalizer(copy=False).fit_transform(cluster_lsa_raw)
    cluster_rng = np.random.default_rng(42)
    cluster_silhouette_indices = np.sort(
        cluster_rng.choice(
            cluster_lsa.shape[0],
            size=min(1_500, cluster_lsa.shape[0]),
            replace=False,
        )
    )
    cluster_candidate_rows = []
    for cluster_candidate_k in [6, 8, 10, 12]:
        cluster_candidate_model = KMeans(
            n_clusters=cluster_candidate_k,
            random_state=42,
            n_init=20,
        ).fit(cluster_lsa)
        cluster_candidate_score = silhouette_score(
            cluster_lsa[cluster_silhouette_indices],
            cluster_candidate_model.labels_[cluster_silhouette_indices],
            metric="euclidean",
        )
        cluster_candidate_rows.append({
            "k": cluster_candidate_k,
            "silhouette": float(cluster_candidate_score),
        })
    cluster_model_selection = pl.DataFrame(cluster_candidate_rows).sort("k")
    cluster_selected_k = int(
        cluster_model_selection.sort(["silhouette", "k"], descending=[True, False])["k"][0]
    )
    cluster_model = KMeans(
        n_clusters=cluster_selected_k,
        random_state=42,
        n_init=20,
    ).fit(cluster_lsa)
    cluster_raw_assignments = cluster_model.labels_
    cluster_raw_counts = np.bincount(cluster_raw_assignments, minlength=cluster_selected_k)
    cluster_raw_term_weights = []
    cluster_raw_labels = []
    for cluster_raw_id in range(cluster_selected_k):
        cluster_raw_mean = np.asarray(
            nlp_tfidf[cluster_raw_assignments == cluster_raw_id].mean(axis=0)
        ).ravel()
        cluster_raw_term_weights.append(cluster_raw_mean)
        cluster_raw_labels.append(
            " · ".join(nlp_feature_names[cluster_raw_mean.argsort()[-3:][::-1]])
        )
    cluster_canonical_order = sorted(
        range(cluster_selected_k),
        key=lambda cluster_raw_id: (-int(cluster_raw_counts[cluster_raw_id]), cluster_raw_labels[cluster_raw_id]),
    )
    cluster_raw_to_canonical = {
        cluster_raw_id: cluster_canonical_id + 1
        for cluster_canonical_id, cluster_raw_id in enumerate(cluster_canonical_order)
    }
    cluster_assignments = np.asarray([
        cluster_raw_to_canonical[cluster_raw_id]
        for cluster_raw_id in cluster_raw_assignments
    ])
    cluster_labels = [cluster_raw_labels[cluster_raw_id] for cluster_raw_id in cluster_canonical_order]
    cluster_term_weights = [cluster_raw_term_weights[cluster_raw_id] for cluster_raw_id in cluster_canonical_order]
    cluster_centroids = cluster_model.cluster_centers_[cluster_canonical_order]
    cluster_distances = np.asarray([
        np.linalg.norm(cluster_lsa[text_index] - cluster_centroids[cluster_assignments[text_index] - 1])
        for text_index in range(cluster_lsa.shape[0])
    ])
    cluster_seconds = time.perf_counter() - cluster_started

    cluster_text_assignments = pl.DataFrame({
        "claim_text": nlp_texts,
        "cluster_id": cluster_assignments.tolist(),
        "cluster_label": [cluster_labels[cluster_id - 1] for cluster_id in cluster_assignments],
        "distance_to_centroid": cluster_distances.tolist(),
        "lsa_x": cluster_lsa[:, 0].tolist(),
        "lsa_y": cluster_lsa[:, 1].tolist(),
    })
    cluster_summary = (
        cluster_text_assignments
        .group_by("cluster_id", "cluster_label")
        .agg(
            pl.len().alias("claim_texts"),
            pl.col("distance_to_centroid").median().round(4).alias("median_distance"),
        )
        .with_columns((100 * pl.col("claim_texts") / claim_text_frame.height).round(2).alias("share_pct"))
        .sort("cluster_id")
    )
    cluster_term_rows = []
    for cluster_index, cluster_component in enumerate(cluster_term_weights, start=1):
        for cluster_rank, cluster_feature_index in enumerate(cluster_component.argsort()[-10:][::-1], start=1):
            cluster_term_rows.append({
                "cluster_id": cluster_index,
                "cluster_label": cluster_labels[cluster_index - 1],
                "rank": cluster_rank,
                "term": nlp_feature_names[cluster_feature_index],
                "mean_tfidf": float(cluster_component[cluster_feature_index]),
            })
    cluster_terms = pl.DataFrame(cluster_term_rows)
    cluster_representatives = (
        cluster_text_assignments
        .sort(["cluster_id", "distance_to_centroid", "claim_text"])
        .group_by("cluster_id", maintain_order=True)
        .head(3)
        .select("cluster_id", "cluster_label", "distance_to_centroid", "claim_text")
    )
    cluster_review_frame = review_frame.join(cluster_text_assignments, on="claim_text", how="left")
    cluster_plot_indices = np.sort(
        np.random.default_rng(42).choice(
            cluster_lsa.shape[0],
            size=min(2_500, cluster_lsa.shape[0]),
            replace=False,
        )
    )
    cluster_plot_frame = cluster_text_assignments[cluster_plot_indices.tolist()]

    cluster_selection_chart = (
        alt.Chart(cluster_model_selection)
        .mark_line(point=True, color="#E45756")
        .encode(
            x=alt.X("k:O", title="Number of clusters (k)"),
            y=alt.Y("silhouette:Q", title="Silhouette score", scale=alt.Scale(zero=False)),
            tooltip=["k:O", alt.Tooltip("silhouette:Q", format=".4f")],
        )
        .properties(width=360, height=260, title="Bounded K-means model selection")
    )
    cluster_size_chart = (
        alt.Chart(cluster_summary)
        .mark_bar()
        .encode(
            x=alt.X("claim_texts:Q", title="Unique claim texts"),
            y=alt.Y("cluster_label:N", title=None, sort=cluster_labels),
            color=alt.Color("cluster_id:O", title="Cluster", scale=alt.Scale(scheme="tableau20"), legend=None),
            tooltip=["cluster_id:O", "cluster_label:N", "claim_texts:Q", alt.Tooltip("share_pct:Q", format=".2f"), alt.Tooltip("median_distance:Q", format=".4f")],
        )
        .properties(width=590, height=330, title="Latent semantic cluster sizes")
    )
    cluster_scatter_chart = (
        alt.Chart(cluster_plot_frame)
        .mark_circle(size=28, opacity=0.45)
        .encode(
            x=alt.X("lsa_x:Q", title="Latent dimension 1"),
            y=alt.Y("lsa_y:Q", title="Latent dimension 2"),
            color=alt.Color("cluster_id:O", title="Cluster", scale=alt.Scale(scheme="tableau20")),
            tooltip=["cluster_id:O", "cluster_label:N", "claim_text:N", alt.Tooltip("distance_to_centroid:Q", format=".4f")],
        )
        .properties(width=820, height=520, title="2D projection of a deterministic 2,500-text sample")
    )
    cluster_selected_score = cluster_model_selection.filter(pl.col("k") == cluster_selected_k)["silhouette"][0]
    cluster_largest = cluster_summary.sort("claim_texts", descending=True).row(0, named=True)

    cluster_section = mo.vstack([
        mo.md(
            f"The bounded search selects **k={cluster_selected_k}** with silhouette "
            f"**{cluster_selected_score:.4f}**. Fifty latent dimensions explain "
            f"**{100 * cluster_svd.explained_variance_ratio_.sum():.1f}%** of TF-IDF variance, "
            f"and the complete cluster pipeline runs in **{cluster_seconds:.2f}s**. "
            f"The largest cluster is **{cluster_largest['cluster_label']}** "
            f"({cluster_largest['claim_texts']:,} texts)."
        ),
        mo.hstack([cluster_selection_chart, cluster_size_chart], align="start", wrap=True),
        cluster_scatter_chart,
        mo.md("### Cluster terms and nearest-centroid claims"),
        mo.hstack([
            mo.ui.table(cluster_terms, selection=None, pagination=True, page_size=20),
            mo.ui.table(cluster_representatives, selection=None, pagination=True, page_size=18, wrapped_columns=["claim_text"]),
        ], align="start", widths="equal", wrap=True),
    ])
    cluster_section
    return cluster_summary, cluster_text_assignments


@app.cell(hide_code=True)
def explain_entity_candidates(mo):
    mo.md("""
    ## 11. Entity candidates

    This section is deliberately labeled **entity candidates**, not named-entity
    recognition. With no downloaded language model, it uses transparent rules:
    repeated title-cased multiword phrases, repeated all-caps acronyms, and the
    structured claimant field.

    Candidates must occur in at least two unique claim texts. Sentence-start
    fact-check boilerplate and stopword-only phrases are removed. Capitalization
    rules can still miss lowercase names or retain headline fragments, so the
    representative text should always be inspected.
    """)
    return


@app.cell
def extract_entity_candidates(
    alt,
    collections,
    mo,
    nlp_stopwords,
    nlp_texts,
    pl,
    re,
    review_frame,
    temporal_claim_records,
):
    entity_title_pattern = re.compile(
        r"\b(?:[A-Z][a-zà-öø-ÿ]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-zà-öø-ÿ]+|[A-Z]{2,})){1,3}\b"
    )
    entity_acronym_pattern = re.compile(r"\b[A-Z]{2,8}\b")
    ENTITY_BLOCKED_STARTS = {
        "akun", "artikel", "benar", "benarkah", "beredar", "cek", "dalam",
        "fakta", "foto", "hoaks", "informasi", "kabar", "keliru", "klaim",
        "narasi", "postingan", "salah", "sebuah", "video",
    }
    ENTITY_BLOCKED_ACRONYMS = {
        "BENAR", "CEK", "FAKTA", "FOTO", "HOAKS", "HOAX", "KELIRU",
        "SALAH", "VIDEO",
    }
    entity_documents = collections.defaultdict(set)
    entity_display_forms = collections.defaultdict(collections.Counter)
    for entity_claim_text in nlp_texts:
        entity_candidates_in_document = set()
        for entity_match in entity_title_pattern.findall(entity_claim_text):
            entity_normalized = re.sub(r"\s+", " ", entity_match).strip().lower()
            entity_tokens = entity_normalized.split()
            if (
                4 <= len(entity_match) <= 80 and
                2 <= len(entity_tokens) <= 4 and
                entity_tokens[0] not in ENTITY_BLOCKED_STARTS and
                not all(token in nlp_stopwords for token in entity_tokens)
            ):
                entity_candidates_in_document.add(("title-cased phrase", entity_normalized, entity_match))
        for entity_match in entity_acronym_pattern.findall(entity_claim_text):
            entity_normalized = entity_match.lower()
            if entity_match not in ENTITY_BLOCKED_ACRONYMS and entity_normalized not in nlp_stopwords:
                entity_candidates_in_document.add(("acronym", entity_normalized, entity_match))
        for entity_type, entity_normalized, entity_display in entity_candidates_in_document:
            entity_key = (entity_type, entity_normalized)
            entity_documents[entity_key].add(entity_claim_text)
            entity_display_forms[entity_key][entity_display] += 1

    entity_claimant_groups = (
        review_frame
        .filter(pl.col("claimant_normalized").is_not_null() & (pl.col("claimant_normalized").str.len_chars() > 0))
        .group_by("claimant_normalized")
        .agg(
            pl.col("claim_text").unique().alias("claim_texts"),
            pl.col("claimant").drop_nulls().first().alias("display"),
        )
    )
    for entity_claimant_row in entity_claimant_groups.iter_rows(named=True):
        entity_key = ("structured claimant", entity_claimant_row["claimant_normalized"])
        entity_documents[entity_key].update(entity_claimant_row["claim_texts"])
        entity_display_forms[entity_key][entity_claimant_row["display"]] += len(entity_claimant_row["claim_texts"])

    entity_year_ranges = (
        temporal_claim_records
        .group_by("claim_text")
        .agg(
            pl.col("claim_year").min().alias("first_year"),
            pl.col("claim_year").max().alias("last_year"),
        )
    )
    entity_year_map = {
        entity_year_row["claim_text"]: (entity_year_row["first_year"], entity_year_row["last_year"])
        for entity_year_row in entity_year_ranges.iter_rows(named=True)
    }
    entity_rows = []
    for entity_key, entity_claim_texts in entity_documents.items():
        if len(entity_claim_texts) < 2:
            continue
        entity_type, entity_normalized = entity_key
        entity_years = [
            year
            for entity_text in entity_claim_texts
            if entity_text in entity_year_map
            for year in entity_year_map[entity_text]
            if year is not None
        ]
        entity_rows.append({
            "candidate": entity_display_forms[entity_key].most_common(1)[0][0],
            "normalized_candidate": entity_normalized,
            "candidate_type": entity_type,
            "document_frequency": len(entity_claim_texts),
            "first_year": min(entity_years) if entity_years else None,
            "last_year": max(entity_years) if entity_years else None,
            "representative_claim": min(entity_claim_texts, key=lambda text_value: (len(text_value), text_value)),
        })
    entity_candidates = (
        pl.DataFrame(entity_rows)
        .sort(["document_frequency", "candidate"], descending=[True, False])
    )
    entity_top30 = entity_candidates.head(30)
    entity_top25_dated = entity_candidates.filter(pl.col("first_year").is_not_null()).head(25)

    entity_frequency_chart = (
        alt.Chart(entity_top30)
        .mark_bar()
        .encode(
            x=alt.X("document_frequency:Q", title="Unique claim texts"),
            y=alt.Y("candidate:N", title=None, sort="-x"),
            color=alt.Color("candidate_type:N", title="Candidate type"),
            tooltip=["candidate:N", "candidate_type:N", "document_frequency:Q", "first_year:O", "last_year:O"],
        )
        .properties(width=570, height=560, title="Top heuristic entity candidates")
    )
    entity_coverage_chart = (
        alt.Chart(entity_top25_dated)
        .mark_rule(strokeWidth=4)
        .encode(
            x=alt.X("first_year:Q", title="Claim year", scale=alt.Scale(zero=False)),
            x2="last_year:Q",
            y=alt.Y("candidate:N", title=None, sort=entity_top25_dated["candidate"].to_list()),
            color=alt.Color("candidate_type:N", title="Candidate type"),
            tooltip=["candidate:N", "candidate_type:N", "document_frequency:Q", "first_year:O", "last_year:O"],
        )
        .properties(width=570, height=500, title="Observed temporal span of top candidates")
    )
    entity_top_by_type = (
        entity_candidates
        .sort(["candidate_type", "document_frequency", "candidate"], descending=[False, True, False])
        .group_by("candidate_type", maintain_order=True)
        .head(1)
    )
    entity_type_finding = "; ".join(
        f"{row['candidate_type']}: {row['candidate']} ({row['document_frequency']:,})"
        for row in entity_top_by_type.iter_rows(named=True)
    )

    entity_section = mo.vstack([
        mo.md(
            f"The transparent rules retain **{entity_candidates.height:,} repeated candidates**. "
            f"Leaders by source are: **{entity_type_finding}**. These counts are useful "
            "for discovery, but capitalization artifacts and generic claimant labels require review."
        ),
        mo.hstack([entity_frequency_chart, entity_coverage_chart], align="start", wrap=True),
        mo.ui.table(
            entity_candidates,
            selection=None,
            pagination=True,
            page_size=25,
            wrapped_columns=["representative_claim"],
        ),
    ])
    entity_section
    return


@app.cell(hide_code=True)
def explain_record_explorer(mo):
    mo.md("""
    ## 12. Record explorer and conclusions

    Three **eager** downloads make the exported static HTML self-contained:

    - analyzed reviewed-claim CSV with parsed dates, normalized labels, keywords,
      topics, and clusters;
    - untouched raw keyword-match CSV from `claims_frame`;
    - the typed source-of-truth Parquet.

    Each payload is below marimo's 10 MiB per-file inline limit, so static HTML can
    download it without Python or a server. The explorer exposes every reviewed-
    claim record and supports client-side search, sorting, pagination, and download.
    The concluding statements summarize coverage and limitations, not causes or
    population-level hoax prevalence.
    """)
    return


@app.cell
def render_record_explorer(
    EXTRACT_PATH,
    claims_frame,
    claims_source,
    cluster_summary,
    cluster_text_assignments,
    executive_overlap_factor,
    keyword_multi_record_pct,
    mo,
    pl,
    publisher_summary,
    quality_top_conflict,
    record_conflict_frame,
    review_frame,
    topic_review_frame,
    topic_summary,
):
    explorer_frame = (
        topic_review_frame
        .join(
            cluster_text_assignments.select(
                "claim_text", "cluster_id", "cluster_label", "distance_to_centroid"
            ),
            on="claim_text",
            how="left",
        )
        .select(
            "record_id",
            "claim_text",
            "claimant",
            "claimant_normalized",
            "claim_date",
            "claim_dt",
            "review_date",
            "review_dt",
            "review_lag_days",
            "publisher_name",
            "publisher_site",
            "textual_rating",
            "rating_normalized",
            "review_title",
            "review_url",
            "matched_keywords",
            "keyword_count",
            "topic_id",
            "topic_label",
            "topic_confidence",
            "cluster_id",
            "cluster_label",
            "distance_to_centroid",
            "language_code",
        )
        .sort(["claim_dt", "review_url", "claim_text"], descending=[True, False, False], nulls_last=True)
    )
    conclusion_top3_publisher_share = float(publisher_summary.head(3)["share_pct"].sum())
    conclusion_claim_date_complete = 100 * review_frame["claim_dt"].is_not_null().sum() / review_frame.height
    conclusion_review_date_complete = 100 * review_frame["review_dt"].is_not_null().sum() / review_frame.height
    conclusion_conflict_pct = 100 * record_conflict_frame.height / review_frame.height
    conclusion_largest_cluster = cluster_summary.sort("claim_texts", descending=True).row(0, named=True)
    conclusion_largest_topic = topic_summary.sort("claim_texts", descending=True).row(0, named=True)

    conclusion_findings = mo.md(f"""
    ### Computed takeaways

    1. **Coverage is concentrated.** The three largest publisher sites contribute
       **{conclusion_top3_publisher_share:.1f}%** of reviewed-claim records; publisher
       shifts can therefore move the aggregate trend.
    2. **Temporal fields are uneven.** Claim dates are available for
       **{conclusion_claim_date_complete:.1f}%** of records, while review dates cover
       only **{conclusion_review_date_complete:.1f}%**. Publication-lag findings apply
       only to complete date pairs.
    3. **Keywords overlap materially.** **{keyword_multi_record_pct:.1f}%** of
       reviewed claims match more than one configured query, producing a
       **{executive_overlap_factor:.2f}×** match-to-record ratio.
    4. **Mechanical NLP summaries need examples.** The largest NMF topic is
       **{conclusion_largest_topic['topic_label']}**
       ({conclusion_largest_topic['share_pct']:.1f}% of unique texts); the largest
       latent cluster is **{conclusion_largest_cluster['cluster_label']}**
       ({conclusion_largest_cluster['share_pct']:.1f}%). These labels are top terms,
       not human-coded themes.
    5. **Metadata is not perfectly stable.** **{record_conflict_frame.height:,}**
       reviewed-claim keys ({conclusion_conflict_pct:.1f}%) have a repeated-field
       conflict, led by `{quality_top_conflict['field']}`. The source parquet is
       preserved and diagnostics expose the affected records.

    **Main limitation:** this is a query-conditioned view of Google's indexed
    fact-check corpus, not a census of Indonesian misinformation. Missing metadata,
    publisher coverage, retrieval vocabulary, and heuristic NLP all constrain what
    can be concluded.
    """)

    analyzed_export_frame = explorer_frame.with_columns(
        pl.col("matched_keywords").list.join("|").alias("matched_keywords")
    )
    analyzed_csv_bytes = analyzed_export_frame.write_csv().encode("utf-8")
    raw_csv_bytes = claims_frame.write_csv().encode("utf-8")
    source_parquet_bytes = EXTRACT_PATH.read_bytes()

    analyzed_csv_download = mo.download(
        data=analyzed_csv_bytes,
        filename="google_fact_check_tool_id_analyzed.csv",
        mimetype="text/csv",
        label=f"Download analyzed CSV ({analyzed_export_frame.height:,} rows)",
    )
    raw_csv_download = mo.download(
        data=raw_csv_bytes,
        filename="google_fact_check_tool_id_raw.csv",
        mimetype="text/csv",
        label=f"Download raw CSV ({claims_frame.height:,} rows)",
    )
    source_parquet_download = mo.download(
        data=source_parquet_bytes,
        filename=EXTRACT_PATH.name,
        mimetype="application/vnd.apache.parquet",
        label=f"Download source Parquet ({len(source_parquet_bytes) / 1024:.0f} KiB)",
    )

    record_explorer = mo.ui.table(
        explorer_frame,
        selection=None,
        pagination=True,
        page_size=25,
        show_search=True,
        show_download=True,
        show_column_summaries=True,
        wrapped_columns=["claim_text", "review_title", "matched_keywords"],
        freeze_columns_left=["claim_text"],
        max_columns=30,
        label="Complete reviewed-claim explorer",
    )

    conclusion_section = mo.vstack([
        conclusion_findings,
        mo.md("### Static-compatible dataset downloads"),
        mo.hstack(
            [analyzed_csv_download, raw_csv_download, source_parquet_download],
            justify="start",
            wrap=True,
        ),
        mo.callout(
            mo.md(
                f"All three files are embedded in a static export. Their raw payloads "
                f"total **{(len(analyzed_csv_bytes) + len(raw_csv_bytes) + len(source_parquet_bytes)) / 1024 / 1024:.2f} MiB** "
                "before base64 encoding."
            ),
            kind="info",
            title="No Python backend required for downloads",
        ),
        mo.callout(
            mo.md(
                f"Loaded **{explorer_frame.height:,} complete reviewed-claim rows** "
                f"from **{claims_source}**. The explorer is not truncated."
            ),
            kind="success",
            title="Record-level access",
        ),
        record_explorer,
    ])
    conclusion_section
    return


if __name__ == "__main__":
    app.run()
