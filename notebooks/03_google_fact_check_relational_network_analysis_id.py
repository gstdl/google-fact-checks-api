import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def introduce_report(mo):
    mo.md("""
    # Descriptive, Temporal, and Relational Network Analysis of Indonesian Fact-Check Records

    This research-oriented report turns the cached Google Fact Check Tools extract
    into four explicitly defined association networks: keyword co-matches,
    publisher–keyword coverage, actor co-mentions, and actor–publisher coverage.

    An edge means only what its definition says. It is **not** evidence of
    misinformation diffusion, coordination, endorsement, influence, affiliation,
    audience exposure, or causality. Google indexing, the configured retrieval
    vocabulary, publisher participation, missing dates, NER errors, and support
    thresholds condition every result.
    """)
    return


@app.cell(hide_code=True)
def render_network_executive_summary(
    actor_node_metrics,
    actor_publisher_edges,
    actor_summary,
    keyword_node_metrics,
    keyword_summary,
    mo,
    publisher_keyword_edges,
    window_edge_changes,
    window_turnover,
):
    summary_keyword_graph = keyword_summary.row(0, named=True)
    summary_actor_graph = actor_summary.row(0, named=True)
    summary_top_keyword = keyword_node_metrics.sort(
        ["betweenness", "strength"], descending=True
    ).row(0, named=True)
    summary_top_actor = actor_node_metrics.sort("strength", descending=True).row(0, named=True)
    summary_persistent = window_turnover.sort("edge_jaccard", descending=True).row(0, named=True)
    summary_largest_change = (
        window_edge_changes
        .with_columns(window_edge_changes["support_change"].abs().alias("absolute_support_change"))
        .sort(["absolute_support_change", "post_support"], descending=True)
        .row(0, named=True)
    )

    network_summary = mo.vstack([
        mo.md("""
        ## 1. Executive summary

        [How to read the networks](#2-how-to-read-the-four-networks) ·
        [Retrieval and publishers](#3-retrieval-and-publisher-structure) ·
        [Actors and publishers](#4-actor-and-publisher-associations) ·
        [Change over time](#5-how-network-structure-changes-over-time) ·
        [Methods and data](#6-methods-sensitivity-audit-and-data)
        """),
        mo.hstack([
            mo.stat(f"{summary_keyword_graph['nodes']:,}", label="Keyword nodes", bordered=True),
            mo.stat(f"{summary_keyword_graph['edges']:,}", label="Keyword edges", bordered=True),
            mo.stat(f"{summary_actor_graph['nodes']:,}", label="Eligible actors", bordered=True),
            mo.stat(f"{summary_actor_graph['edges']:,}", label="Actor co-mentions", bordered=True),
            mo.stat(f"{actor_publisher_edges.height:,}", label="Actor–publisher edges", bordered=True),
        ], widths="equal", wrap=True),
        mo.md(f"""
        - **Retrieval bridge:** `{summary_top_keyword['label']}` has the highest exact
          inverse-Jaccard keyword betweenness (**{summary_top_keyword['betweenness']:.4f}**).
        - **Actor co-mention strength:** `{summary_top_actor['label']}` has the largest retained
          strength (**{summary_top_actor['strength']:.0f}**).
        - **Window persistence:** the `{summary_persistent['network_family']}` network has the
          higher matched-window edge-set Jaccard (**{summary_persistent['edge_jaccard']:.3f}**).
        - **Largest support movement:** `{summary_largest_change['source']}` ↔
          `{summary_largest_change['target']}` changes by
          **{summary_largest_change['support_change']:+,}** records in the
          `{summary_largest_change['network_family']}` network.
        """),
        mo.callout(
            mo.md(
                f"The four networks contain **{publisher_keyword_edges.height:,} publisher–keyword "
                "edges** plus keyword, actor co-mention, and actor–publisher relations. Every edge "
                "means only retrieval overlap, textual co-mention, or indexed publisher coverage."
            ),
            kind="info",
            title="What an edge means",
        ),
        mo.callout(
            mo.md(
                "Edges are not evidence of misinformation diffusion, coordination, endorsement, "
                "influence, affiliation, audience exposure, or causality. Centrality is a position "
                "inside this query-conditioned extract—not social importance."
            ),
            kind="warn",
            title="Interpretation boundary",
        ),
    ])
    network_summary
    return


@app.cell(hide_code=True)
def explain_scope(
    ACTOR_MIN_COMENTIONS,
    ACTOR_MIN_TEXTS,
    KEYWORD_MIN_SHARED,
    mo,
):
    network_reading_guide = mo.vstack([
        mo.md(f"""
        ## 2. How to read the four networks

        - **Keyword co-match:** two query keywords retrieved the same reviewed claim.
        - **Publisher–keyword:** a publisher reviewed a claim retrieved by a keyword.
        - **Actor co-mention:** two model-generated actor candidates appear in the same
          unique claim text.
        - **Actor–publisher:** a publisher reviewed a claim text that mentions an actor candidate.

        The primary keyword graph requires **{KEYWORD_MIN_SHARED} shared reviewed claims**.
        Actor nodes require **{ACTOR_MIN_TEXTS} claim texts**, and actor co-mention edges
        require **{ACTOR_MIN_COMENTIONS} texts**. These thresholds reduce visual clutter;
        they do not define natural boundaries.
        """),
        mo.accordion({
            "Centrality, communities, and analytical grains": mo.md("""
            Keyword betweenness uses `1 / Jaccard` as path distance. Communities use
            deterministic greedy modularity; bridges and articulation points describe
            graph topology. Keyword/publisher relations use distinct reviewed claims,
            while actor relations use distinct claim texts so repeated publisher reviews
            do not inflate co-mentions.
            """),
        }),
    ])
    network_reading_guide
    return


@app.cell
def setup_environment():
    import collections
    import hashlib
    import itertools
    import json
    import math
    import re
    import unicodedata
    from datetime import datetime, timezone

    import altair as alt
    import marimo as mo
    import networkx as nx
    import numpy as np
    import polars as pl
    from sklearn.metrics import normalized_mutual_info_score

    REPO_ROOT = mo.notebook_dir().parent
    SOURCE_PATH = REPO_ROOT / "data" / "google_fact_check_tool_id.parquet"
    NER_CACHE_PATH = REPO_ROOT / "data" / "google_fact_check_ner_entities_id.parquet"
    REFRESH_COMMAND = "uv run --extra ner python scripts/refresh-ner-entities.py"

    MODEL_ID = "cahya/bert-base-indonesian-NER"
    MODEL_REVISION = "a3a3fa494cf7555ef87f446af5e826de3ed181c0"
    SEED = 42
    KEYWORD_MIN_SHARED = 5
    ACTOR_MIN_TEXTS = 5
    ACTOR_MIN_COMENTIONS = 2
    YEARLY_LAST_YEAR = 2024
    YEARLY_MIN_RECORDS = 100
    BREAKPOINT = datetime(2022, 11, 30, tzinfo=timezone.utc)
    PRE_START = datetime(2019, 11, 30, tzinfo=timezone.utc)
    POST_END = datetime(2025, 11, 30, tzinfo=timezone.utc)

    def normalize_identity(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = unicodedata.normalize("NFKC", value)
        normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
        return normalized or None

    def corpus_hash(texts: list[str]) -> str:
        digest = hashlib.sha256()
        for text in texts:
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    def edge_rows_from_memberships(
        memberships: list[tuple[str, list[str]]],
        node_support: dict[str, int],
        similarity: bool,
    ) -> list[dict[str, object]]:
        pair_counts: collections.Counter[tuple[str, str]] = collections.Counter()
        representative: dict[tuple[str, str], str] = {}
        for record_key, members in memberships:
            unique_members = sorted(set(members))
            for source, target in itertools.combinations(unique_members, 2):
                pair_counts[(source, target)] += 1
                representative.setdefault((source, target), record_key)
        rows = []
        for (source, target), support in sorted(pair_counts.items()):
            union = node_support[source] + node_support[target] - support
            jaccard = support / union if similarity and union else None
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "support": support,
                    "jaccard": jaccard,
                    "distance": (1 / jaccard) if jaccard else (1 / support),
                    "representative_record": representative[(source, target)],
                }
            )
        return rows

    def analyze_undirected(
        node_frame: pl.DataFrame,
        edge_frame: pl.DataFrame,
        exact_betweenness: bool,
        distance_column: str = "distance",
    ) -> tuple[pl.DataFrame, pl.DataFrame, nx.Graph]:
        graph = nx.Graph()
        for row in node_frame.iter_rows(named=True):
            graph.add_node(row["node_id"], **row)
        for row in edge_frame.iter_rows(named=True):
            graph.add_edge(
                row["source"],
                row["target"],
                support=float(row["support"]),
                distance=float(row[distance_column]),
            )

        components = sorted(
            nx.connected_components(graph),
            key=lambda members: (-len(members), min(members)),
        )
        component_map = {
            node_id: component_id
            for component_id, members in enumerate(components, start=1)
            for node_id in members
        }
        communities: list[set[str]] = []
        if graph.number_of_edges():
            communities = [set(members) for members in nx.community.greedy_modularity_communities(graph, weight="support")]
            covered = set().union(*communities) if communities else set()
            communities.extend([{node_id} for node_id in sorted(set(graph) - covered)])
            communities.sort(key=lambda members: (-len(members), min(members)))
        else:
            communities = [{node_id} for node_id in sorted(graph)]
        community_map = {
            node_id: community_id
            for community_id, members in enumerate(communities, start=1)
            for node_id in members
        }
        if graph.number_of_nodes() <= 2 or graph.number_of_edges() == 0:
            betweenness = {node_id: 0.0 for node_id in graph}
        elif exact_betweenness or graph.number_of_nodes() <= 200:
            betweenness = nx.betweenness_centrality(graph, weight=distance_column, normalized=True)
        else:
            betweenness = nx.betweenness_centrality(
                graph,
                k=min(200, graph.number_of_nodes()),
                weight=distance_column,
                normalized=True,
                seed=SEED,
            )
        articulation = set(nx.articulation_points(graph)) if graph.number_of_edges() else set()
        bridge_edges = {
            tuple(sorted(edge)) for edge in nx.bridges(graph)
        } if graph.number_of_edges() else set()
        strength = dict(graph.degree(weight="support"))
        degree = dict(graph.degree())
        metric_rows = []
        for row in node_frame.iter_rows(named=True):
            node_id = row["node_id"]
            metric_rows.append(
                {
                    **row,
                    "degree": int(degree[node_id]),
                    "strength": float(strength[node_id]),
                    "betweenness": float(betweenness[node_id]),
                    "component_id": component_map[node_id],
                    "community_id": community_map[node_id],
                    "is_articulation": node_id in articulation,
                }
            )
        node_metrics = pl.DataFrame(metric_rows).sort(["strength", "node_id"], descending=[True, False])

        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        degrees = list(degree.values())
        centralization_denominator = (node_count - 1) * (node_count - 2)
        degree_centralization = (
            sum(max(degrees, default=0) - value for value in degrees) / centralization_denominator
            if centralization_denominator > 0 else 0.0
        )
        modularity = (
            nx.community.modularity(graph, communities, weight="support")
            if edge_count and communities else 0.0
        )
        summary = pl.DataFrame(
            [
                {
                    "nodes": node_count,
                    "edges": edge_count,
                    "isolates": len(list(nx.isolates(graph))),
                    "components": len(components),
                    "largest_component": max((len(members) for members in components), default=0),
                    "density": nx.density(graph) if node_count > 1 else 0.0,
                    "degree_centralization": degree_centralization,
                    "communities": len(communities),
                    "modularity": modularity,
                    "bridges": len(bridge_edges),
                    "articulation_nodes": len(articulation),
                    "total_edge_support": int(edge_frame["support"].sum()) if edge_frame.height else 0,
                }
            ]
        )

        assert node_metrics["node_id"].n_unique() == node_metrics.height
        assert node_metrics["component_id"].null_count() == 0
        assert node_metrics["community_id"].null_count() == 0
        assert math.isclose(
            node_metrics["strength"].sum(),
            2 * (edge_frame["support"].sum() if edge_frame.height else 0),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        return node_metrics, summary, graph

    def backbone_frames(
        node_metrics: pl.DataFrame,
        edge_frame: pl.DataFrame,
        max_nodes: int,
        neighbors_per_node: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        selected_nodes = node_metrics.head(max_nodes)["node_id"].to_list()
        selected_set = set(selected_nodes)
        candidate_edges = edge_frame.filter(
            pl.col("source").is_in(selected_nodes) & pl.col("target").is_in(selected_nodes)
        )
        retained_pairs: set[tuple[str, str]] = set()
        for node_id in selected_nodes:
            incident = (
                candidate_edges
                .filter((pl.col("source") == node_id) | (pl.col("target") == node_id))
                .sort(["support", "source", "target"], descending=[True, False, False])
                .head(neighbors_per_node)
            )
            retained_pairs.update((row["source"], row["target"]) for row in incident.iter_rows(named=True))
        if retained_pairs:
            backbone_edges = pl.DataFrame(
                [row for row in candidate_edges.iter_rows(named=True) if (row["source"], row["target"]) in retained_pairs],
                schema=candidate_edges.schema,
            ).sort(["support", "source", "target"], descending=[True, False, False])
        else:
            backbone_edges = candidate_edges.head(0)
        visual_ids = sorted(selected_set)
        visual_graph = nx.Graph()
        visual_graph.add_nodes_from(visual_ids)
        visual_graph.add_weighted_edges_from(
            [(row["source"], row["target"], float(row["support"])) for row in backbone_edges.iter_rows(named=True)],
            weight="support",
        )
        positions = nx.spring_layout(visual_graph, seed=SEED, weight="support", iterations=200)
        repeated_positions = nx.spring_layout(visual_graph, seed=SEED, weight="support", iterations=200)
        assert all(np.allclose(positions[node], repeated_positions[node]) for node in visual_graph)
        visual_nodes = (
            node_metrics
            .filter(pl.col("node_id").is_in(visual_ids))
            .with_columns(
                pl.col("node_id").replace_strict({node: float(position[0]) for node, position in positions.items()}).alias("x"),
                pl.col("node_id").replace_strict({node: float(position[1]) for node, position in positions.items()}).alias("y"),
            )
        )
        if backbone_edges.height:
            visual_edges = (
                backbone_edges
                .with_columns(
                    pl.col("source").replace_strict({node: float(position[0]) for node, position in positions.items()}).alias("x"),
                    pl.col("source").replace_strict({node: float(position[1]) for node, position in positions.items()}).alias("y"),
                    pl.col("target").replace_strict({node: float(position[0]) for node, position in positions.items()}).alias("x2"),
                    pl.col("target").replace_strict({node: float(position[1]) for node, position in positions.items()}).alias("y2"),
                )
            )
        else:
            visual_edges = backbone_edges.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("x"),
                pl.lit(None, dtype=pl.Float64).alias("y"),
                pl.lit(None, dtype=pl.Float64).alias("x2"),
                pl.lit(None, dtype=pl.Float64).alias("y2"),
            )
        assert set(visual_edges["source"].to_list() + visual_edges["target"].to_list()).issubset(selected_set)
        return visual_nodes, visual_edges

    def network_chart(
        visual_nodes: pl.DataFrame,
        visual_edges: pl.DataFrame,
        title: str,
        color_field: str,
        color_title: str,
    ) -> alt.LayerChart:
        edges = (
            alt.Chart(visual_edges)
            .mark_rule(color="#9CA3AF", opacity=0.35)
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                x2="x2:Q",
                y2="y2:Q",
                strokeWidth=alt.StrokeWidth("support:Q", title="Edge support", scale=alt.Scale(range=[0.4, 4])),
                tooltip=["source:N", "target:N", "support:Q"],
            )
        )
        nodes = (
            alt.Chart(visual_nodes)
            .mark_circle(opacity=0.9, stroke="white", strokeWidth=0.7)
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                size=alt.Size(
                    "strength:Q",
                    title="Strength",
                    scale=alt.Scale(range=[60, 1300]),
                    legend=alt.Legend(orient="bottom"),
                ),
                color=alt.Color(
                    f"{color_field}:N",
                    title=color_title,
                    scale=alt.Scale(scheme="tableau20"),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    "node_id:N", "label:N", "node_type:N", "support:Q",
                    "degree:Q", alt.Tooltip("strength:Q", format=".1f"),
                    alt.Tooltip("betweenness:Q", format=".4f"),
                    "component_id:O", "community_id:O",
                ],
            )
        )
        labels = (
            alt.Chart(visual_nodes.sort("strength", descending=True).head(14))
            .mark_text(dx=8, dy=-8, align="left", fontSize=10)
            .encode(x="x:Q", y="y:Q", text="label:N")
        )
        return (edges + nodes + labels).properties(width="container", height=610, title=title)

    def graph_summary_only(
        node_ids: list[str], edge_frame: pl.DataFrame
    ) -> tuple[dict[str, object], dict[str, int], set[tuple[str, str]]]:
        graph = nx.Graph()
        graph.add_nodes_from(node_ids)
        graph.add_weighted_edges_from(
            [(row["source"], row["target"], float(row["support"])) for row in edge_frame.iter_rows(named=True)],
            weight="support",
        )
        components = list(nx.connected_components(graph))
        if graph.number_of_edges():
            communities = list(nx.community.greedy_modularity_communities(graph, weight="support"))
            community_map = {
                node: community_id
                for community_id, members in enumerate(communities, start=1)
                for node in members
            }
            for isolate in nx.isolates(graph):
                community_map.setdefault(isolate, max(community_map.values(), default=0) + 1)
            modularity = nx.community.modularity(graph, communities, weight="support")
        else:
            community_map = {node: index for index, node in enumerate(sorted(node_ids), start=1)}
            modularity = 0.0
        strengths = dict(graph.degree(weight="support"))
        degrees = dict(graph.degree())
        n = graph.number_of_nodes()
        denominator = (n - 1) * (n - 2)
        centralization = (
            sum(max(degrees.values(), default=0) - value for value in degrees.values()) / denominator
            if denominator > 0 else 0.0
        )
        top_node = min(
            strengths,
            key=lambda node: (-strengths[node], node),
        ) if strengths else None
        summary = {
            "nodes": n,
            "active_nodes": sum(value > 0 for value in degrees.values()),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph) if n > 1 else 0.0,
            "components": len(components),
            "largest_component": max((len(component) for component in components), default=0),
            "modularity": modularity,
            "degree_centralization": centralization,
            "top_node": top_node,
            "top_strength": float(strengths.get(top_node, 0.0)) if top_node else 0.0,
        }
        edge_set = {tuple(sorted(edge)) for edge in graph.edges()}
        return summary, community_map, edge_set

    return (
        ACTOR_MIN_COMENTIONS,
        ACTOR_MIN_TEXTS,
        BREAKPOINT,
        KEYWORD_MIN_SHARED,
        MODEL_ID,
        MODEL_REVISION,
        NER_CACHE_PATH,
        POST_END,
        PRE_START,
        REFRESH_COMMAND,
        REPO_ROOT,
        SEED,
        SOURCE_PATH,
        YEARLY_LAST_YEAR,
        YEARLY_MIN_RECORDS,
        alt,
        analyze_undirected,
        backbone_frames,
        collections,
        corpus_hash,
        edge_rows_from_memberships,
        graph_summary_only,
        itertools,
        json,
        mo,
        network_chart,
        normalize_identity,
        normalized_mutual_info_score,
        np,
        nx,
        pl,
        re,
    )


@app.cell
def load_validate_and_prepare_frames(
    BREAKPOINT,
    MODEL_ID,
    MODEL_REVISION,
    NER_CACHE_PATH,
    POST_END,
    PRE_START,
    REFRESH_COMMAND,
    SOURCE_PATH,
    corpus_hash,
    json,
    normalize_identity,
    pl,
    re,
):
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_PATH}; run notebooks/00_google_fact_check_tool_id.py first."
        )
    if not NER_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {NER_CACHE_PATH}. Refresh it with `{REFRESH_COMMAND}`."
        )

    raw_frame = pl.read_parquet(SOURCE_PATH)
    source_texts = (
        raw_frame
        .filter(pl.col("claim_text").is_not_null() & (pl.col("claim_text").str.len_chars() > 0))
        .select("claim_text")
        .unique()
        .sort("claim_text")["claim_text"]
        .to_list()
    )
    source_text_hash = corpus_hash(source_texts)
    ner_cache_frame = pl.read_parquet(NER_CACHE_PATH).sort("claim_text")
    cache_metadata = ner_cache_frame.select(
        "model_id", "model_revision", "model_license", "model_file_sha256_json", "model_label_mapping_json",
        "tokenizer_class", "aggregation_strategy", "max_length", "stride", "batch_size",
        "seed", "inference_schema_version", "source_text_hash",
    ).unique()
    if (
        ner_cache_frame.height != len(source_texts)
        or ner_cache_frame["claim_text"].n_unique() != len(source_texts)
        or ner_cache_frame["claim_text"].to_list() != source_texts
        or cache_metadata.height != 1
        or cache_metadata["source_text_hash"][0] != source_text_hash
        or cache_metadata["model_id"][0] != MODEL_ID
        or cache_metadata["model_revision"][0] != MODEL_REVISION
        or cache_metadata["tokenizer_class"][0] != "BertTokenizerFast"
        or cache_metadata["aggregation_strategy"][0] != "simple"
        or cache_metadata["max_length"][0] != 512
        or cache_metadata["stride"][0] != 64
        or cache_metadata["batch_size"][0] != 16
        or cache_metadata["seed"][0] != 42
        or cache_metadata["inference_schema_version"][0] != 3
    ):
        raise RuntimeError(f"NER cache is absent, stale, or inconsistent. Run `{REFRESH_COMMAND}`.")

    record_keys = ["review_url", "claim_text"]
    metadata_fields = [
        "claimant", "claim_date", "publisher_name", "publisher_site",
        "review_title", "textual_rating", "review_date", "language_code",
    ]
    keyword_link_frame = (
        raw_frame
        .unique(subset=["keyword", *record_keys])
        .sort(["keyword", *record_keys])
    )
    review_frame = (
        raw_frame
        .sort([*record_keys, "keyword"])
        .group_by(record_keys, maintain_order=True)
        .agg(
            pl.col("keyword").unique().sort().alias("matched_keywords"),
            *[pl.col(field).drop_nulls().first().alias(field) for field in metadata_fields],
        )
        .with_columns(
            pl.concat_str(record_keys, separator=" ⟂ ").alias("record_id"),
            pl.col("claim_date").str.to_datetime(strict=False, time_zone="UTC").alias("claim_dt"),
        )
        .with_columns(
            pl.col("claim_dt").dt.year().alias("claim_year"),
            (pl.col("claim_date").is_not_null() & pl.col("claim_dt").is_null()).alias("claim_date_parse_failed"),
            pl.when((pl.col("claim_dt") >= PRE_START) & (pl.col("claim_dt") < BREAKPOINT))
            .then(pl.lit("Pre-launch"))
            .when((pl.col("claim_dt") >= BREAKPOINT) & (pl.col("claim_dt") < POST_END))
            .then(pl.lit("Post-launch"))
            .otherwise(None)
            .alias("launch_window"),
        )
        .sort(record_keys)
    )

    ner_nonempty = ner_cache_frame.filter(pl.col("entities").list.len() > 0)
    ner_span_frame = (
        ner_nonempty
        .explode("entities", empty_as_null=True)
        .unnest("entities")
        .sort(["claim_text", "start_char", "end_char", "raw_label"])
    )
    label_mapping = json.loads(cache_metadata["model_label_mapping_json"][0])
    allowed_raw_groups = {label.split("-", 1)[-1] for label in label_mapping.values() if label != "O"}
    invalid_spans = [
        row
        for row in ner_span_frame.select(
            "claim_text", "mention_text", "start_char", "end_char", "raw_label", "normalized_text"
        ).iter_rows(named=True)
        if not (
            0 <= row["start_char"] < row["end_char"] <= len(row["claim_text"])
            and row["claim_text"][row["start_char"]:row["end_char"]] == row["mention_text"]
            and bool(row["normalized_text"])
            and row["raw_label"] in allowed_raw_groups
        )
    ]
    assert not invalid_spans

    publisher_aliases = {
        normalized
        for value in review_frame.select("publisher_site", "publisher_name").to_series(0).to_list()
        for normalized in [normalize_identity(value)]
        if normalized
    }
    publisher_aliases.update(
        normalized
        for value in review_frame["publisher_name"].to_list()
        for normalized in [normalize_identity(value)]
        if normalized
    )
    generic_claimant_pattern = re.compile(
        r"(?i)\b(?:akun|beredar|berbagai|beberapa|pelbagai|sumber|media\s+sosial|"
        r"facebook|instagram|whatsapp|twitter|youtube|tiktok|grup|kanal|situs|"
        r"multiple|sejumlah|viral|video|narasi|postingan|person)\b"
    )
    domain_pattern = re.compile(r"(?i)(?:https?://|www\.|\b[a-z0-9-]+\.(?:com|co|id|org|net)\b)")

    def claimant_exclusion_reason(value: str | None, normalized: str | None) -> str | None:
        if normalized is None:
            return "missing"
        if normalized in publisher_aliases:
            return "publisher name/site"
        if normalized == "[site:name]" or normalized.startswith("["):
            return "placeholder"
        if domain_pattern.search(value or ""):
            return "URL/domain-like"
        if generic_claimant_pattern.search(value or ""):
            return "generic source/platform label"
        return None

    claimant_audit_frame = (
        review_frame
        .select("record_id", "claim_text", "claimant", "publisher_site")
        .with_columns(
            pl.col("claimant").map_elements(normalize_identity, return_dtype=pl.Utf8).alias("normalized_claimant")
        )
        .with_columns(
            pl.struct("claimant", "normalized_claimant")
            .map_elements(
                lambda row: claimant_exclusion_reason(row["claimant"], row["normalized_claimant"]),
                return_dtype=pl.Utf8,
            )
            .alias("exclusion_reason")
        )
    )
    rejected_claimant_frame = claimant_audit_frame.filter(pl.col("exclusion_reason").is_not_null())
    retained_claimant_frame = claimant_audit_frame.filter(pl.col("exclusion_reason").is_null())

    ner_actor_occurrences = (
        ner_span_frame
        .filter(pl.col("entity_type").is_in(["PERSON", "ORGANIZATION"]))
        .sort(["claim_text", "normalized_text", "score"], descending=[False, False, True])
        .group_by("claim_text", "normalized_text", maintain_order=True)
        .agg(
            pl.col("mention_text").first().alias("display_name"),
            pl.col("entity_type").unique().sort().alias("model_entity_types"),
            pl.col("score").max().alias("max_ner_score"),
            pl.len().alias("span_count"),
        )
        .rename({"normalized_text": "normalized_actor"})
        .with_columns(pl.lit("ner").alias("provenance"))
    )
    claimant_actor_occurrences = (
        retained_claimant_frame
        .select("claim_text", "normalized_claimant", "claimant")
        .unique()
        .rename({"normalized_claimant": "normalized_actor", "claimant": "display_name"})
        .with_columns(
            pl.lit([], dtype=pl.List(pl.Utf8)).alias("model_entity_types"),
            pl.lit(None, dtype=pl.Float64).alias("max_ner_score"),
            pl.lit(0, dtype=pl.UInt32).alias("span_count"),
            pl.lit("structured_claimant").alias("provenance"),
        )
    )
    actor_occurrence_frame = (
        pl.concat(
            [
                ner_actor_occurrences.select(
                    "claim_text", "normalized_actor", "display_name", "model_entity_types",
                    "max_ner_score", "span_count", "provenance",
                ),
                claimant_actor_occurrences.select(
                    "claim_text", "normalized_actor", "display_name", "model_entity_types",
                    "max_ner_score", "span_count", "provenance",
                ),
            ],
            how="vertical_relaxed",
        )
        .sort(["claim_text", "normalized_actor", "provenance"])
        .group_by("claim_text", "normalized_actor", maintain_order=True)
        .agg(
            pl.col("display_name").sort().first().alias("display_name"),
            pl.col("model_entity_types").explode(empty_as_null=True).drop_nulls().unique().sort().alias("model_entity_types"),
            pl.col("provenance").unique().sort().alias("provenance_flags"),
            pl.col("max_ner_score").max().alias("max_ner_score"),
            pl.col("span_count").sum().alias("span_count"),
        )
        .with_columns((pl.lit("actor::") + pl.col("normalized_actor")).alias("actor_id"))
        .sort(["claim_text", "actor_id"])
    )

    assert raw_frame.height >= keyword_link_frame.height >= review_frame.height
    assert review_frame.select(record_keys).is_duplicated().sum() == 0
    assert ner_cache_frame["claim_text"].n_unique() == len(source_texts)
    assert actor_occurrence_frame.select("claim_text", "actor_id").is_duplicated().sum() == 0

    return (
        actor_occurrence_frame,
        cache_metadata,
        claimant_audit_frame,
        keyword_link_frame,
        ner_cache_frame,
        ner_span_frame,
        raw_frame,
        rejected_claimant_frame,
        retained_claimant_frame,
        review_frame,
        source_text_hash,
    )


@app.cell
def render_provenance_and_grains(
    cache_metadata,
    claimant_audit_frame,
    json,
    keyword_link_frame,
    mo,
    ner_cache_frame,
    ner_span_frame,
    pl,
    raw_frame,
    review_frame,
    source_text_hash,
):
    grain_table = pl.DataFrame(
        [
            {"frame": "raw", "grain": "keyword-match row", "rows": raw_frame.height},
            {"frame": "keyword links", "grain": "keyword × reviewed claim", "rows": keyword_link_frame.height},
            {"frame": "reviewed claims", "grain": "distinct (review URL, claim text)", "rows": review_frame.height},
            {"frame": "NER cache", "grain": "unique non-empty claim text", "rows": ner_cache_frame.height},
            {"frame": "NER spans", "grain": "model span", "rows": ner_span_frame.height},
        ]
    )
    date_quality = pl.DataFrame(
        [
            {"status": "parseable claim date", "records": review_frame.filter(pl.col("claim_dt").is_not_null()).height},
            {"status": "missing claim date", "records": review_frame["claim_date"].null_count()},
            {"status": "unparseable non-null date", "records": review_frame.filter(pl.col("claim_date_parse_failed")).height},
            {"status": "inside matched windows", "records": review_frame.filter(pl.col("launch_window").is_not_null()).height},
            {"status": "parseable but outside matched windows", "records": review_frame.filter(pl.col("claim_dt").is_not_null() & pl.col("launch_window").is_null()).height},
        ]
    )
    provenance_table = cache_metadata.select(
        "model_id", "model_revision", "model_license", "tokenizer_class",
        "aggregation_strategy", "max_length", "stride", "batch_size", "seed",
        "inference_schema_version",
    )
    model_label_mapping = json.loads(cache_metadata["model_label_mapping_json"][0])
    model_label_table = pl.DataFrame(
        [
            {
                "label_id": int(label_id),
                "raw_model_label": raw_label,
                "actor_graph_use": raw_label.endswith("-PER") or raw_label.endswith("-ORG"),
            }
            for label_id, raw_label in sorted(model_label_mapping.items(), key=lambda item: int(item[0]))
        ]
    )
    model_file_hashes = json.loads(cache_metadata["model_file_sha256_json"][0])
    model_file_hash_table = pl.DataFrame(
        [{"model_file": filename, "sha256": digest} for filename, digest in sorted(model_file_hashes.items())]
    )
    missing_claimants = claimant_audit_frame.filter(pl.col("exclusion_reason") == "missing").height
    provenance_section = mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{review_frame.height:,}", label="Reviewed claims", bordered=True),
                    mo.stat(f"{ner_cache_frame.height:,}", label="NER documents", bordered=True),
                    mo.stat(f"{ner_span_frame.height:,}", label="NER spans", bordered=True),
                    mo.stat(f"{missing_claimants / review_frame.height:.1%}", label="Missing claimants", bordered=True),
                ],
                widths="equal",
                wrap=True,
            ),
            mo.md(f"NER source-text hash: `{source_text_hash}`"),
            mo.hstack(
                [
                    mo.ui.table(grain_table, selection=None, pagination=False, show_column_summaries=False),
                    mo.ui.table(date_quality, selection=None, pagination=False, show_column_summaries=False),
                ],
                widths="equal",
                align="start",
                wrap=True,
            ),
            mo.md("### Pinned NER inference provenance"),
            mo.ui.table(provenance_table, selection=None, pagination=False, show_column_summaries=False),
            mo.ui.tabs(
                {
                    "Model BIO label mapping": mo.ui.table(
                        model_label_table,
                        selection=None,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                    "Pinned model file checksums": mo.ui.table(
                        model_file_hash_table,
                        selection=None,
                        pagination=False,
                        show_column_summaries=False,
                    ),
                }
            ),
        ]
    )
    return (provenance_section,)


@app.cell(hide_code=True)
def explain_keyword_network(mo):
    mo.md("""
    ## 3. Retrieval and publisher structure

    ### Keyword co-match network

    Nodes are configured API query keywords. Two nodes connect when both queries
    retrieved the same reviewed-claim record. Edge support counts shared records;
    Jaccard divides that support by the union of both keywords' records. The
    primary graph requires at least **5 shared records**.

    Betweenness uses `1 / Jaccard` as path distance because NetworkX interprets a
    weight passed to shortest-path centrality as distance. Communities use
    deterministic greedy modularity. These are properties of retrieval overlap,
    not latent truth or information flow.
    """)
    return


@app.cell
def build_keyword_network(
    KEYWORD_MIN_SHARED,
    analyze_undirected,
    backbone_frames,
    edge_rows_from_memberships,
    keyword_link_frame,
    network_chart,
    nx,
    pl,
    review_frame,
):
    keyword_support = {
        f"keyword::{row['keyword']}": row["records"]
        for row in keyword_link_frame.group_by("keyword").len(name="records").iter_rows(named=True)
    }
    keyword_memberships = [
        (row["record_id"], [f"keyword::{keyword}" for keyword in row["matched_keywords"]])
        for row in review_frame.select("record_id", "matched_keywords").iter_rows(named=True)
    ]
    keyword_edge_all = pl.DataFrame(
        edge_rows_from_memberships(keyword_memberships, keyword_support, similarity=True)
    ).sort(["support", "source", "target"], descending=[True, False, False])
    keyword_nodes = pl.DataFrame(
        [
            {
                "node_id": node_id,
                "label": node_id.removeprefix("keyword::"),
                "node_type": "keyword",
                "support": support,
            }
            for node_id, support in sorted(keyword_support.items())
        ]
    )
    keyword_edges = keyword_edge_all.filter(pl.col("support") >= KEYWORD_MIN_SHARED)
    keyword_node_metrics, keyword_summary, keyword_graph = analyze_undirected(
        keyword_nodes, keyword_edges, exact_betweenness=True
    )
    keyword_visual_nodes, keyword_visual_edges = backbone_frames(
        keyword_node_metrics, keyword_edges, max_nodes=35, neighbors_per_node=3
    )
    keyword_chart = network_chart(
        keyword_visual_nodes,
        keyword_visual_edges,
        "Keyword co-match backbone",
        "community_id",
        "Community",
    )
    keyword_bridge_pairs = {tuple(sorted(edge)) for edge in nx.bridges(keyword_graph)}
    keyword_edges = keyword_edges.with_columns(
        pl.struct("source", "target")
        .map_elements(
            lambda row: tuple(sorted((row["source"], row["target"]))) in keyword_bridge_pairs,
            return_dtype=pl.Boolean,
        )
        .alias("is_bridge")
    )
    assert keyword_edges.select("source", "target").is_duplicated().sum() == 0
    assert all(
        row["support"] <= min(keyword_support[row["source"]], keyword_support[row["target"]])
        for row in keyword_edge_all.iter_rows(named=True)
    )
    assert keyword_edges.filter(pl.col("source") >= pl.col("target")).height == 0
    assert keyword_edges.filter((pl.col("support") <= 0) | (pl.col("source") == pl.col("target"))).height == 0
    assert keyword_visual_edges.height <= keyword_edges.height
    return (
        keyword_chart,
        keyword_edge_all,
        keyword_edges,
        keyword_node_metrics,
        keyword_nodes,
        keyword_summary,
    )


@app.cell
def render_keyword_network(
    KEYWORD_MIN_SHARED,
    keyword_chart,
    keyword_edges,
    keyword_node_metrics,
    keyword_summary,
    mo,
    pl,
):
    keyword_summary_row = keyword_summary.row(0, named=True)
    top_bridge = keyword_node_metrics.sort(["betweenness", "strength"], descending=True).row(0, named=True)
    keyword_community_summary = (
        keyword_node_metrics
        .sort(["community_id", "strength", "label"], descending=[False, True, False])
        .group_by("community_id", maintain_order=True)
        .agg(
            pl.len().alias("nodes"),
            pl.col("strength").sum().alias("total_strength"),
            pl.col("label").head(10).alias("leading_keywords"),
        )
        .sort(["total_strength", "community_id"], descending=[True, False])
    )
    keyword_section = mo.vstack(
        [
            mo.md(
                f"At support ≥ **{KEYWORD_MIN_SHARED}**, the full graph has **{keyword_summary_row['nodes']} nodes**, "
                f"**{keyword_summary_row['edges']} edges**, **{keyword_summary_row['components']} components**, density "
                f"**{keyword_summary_row['density']:.3f}**, and **{keyword_summary_row['communities']} communities** "
                f"(modularity **{keyword_summary_row['modularity']:.3f}**). The highest inverse-Jaccard "
                f"betweenness belongs to **{top_bridge['label']}** ({top_bridge['betweenness']:.4f})."
            ),
            mo.callout(
                mo.md(
                    "Metrics use the complete support-thresholded graph. The chart is a deterministic "
                    "top-neighbor backbone for legibility and does not define the metrics."
                ),
                kind="warn",
                title="Full graph versus visual backbone",
            ),
            keyword_chart,
            mo.ui.tabs({
                "Central and bridging keywords": mo.ui.table(
                    keyword_node_metrics.select(
                        "label", "support", "degree", "strength", "betweenness",
                        "community_id", "component_id", "is_articulation",
                    ).head(30),
                    selection=None,
                    pagination=True,
                    page_size=15,
                    show_column_summaries=False,
                ),
                "Strongest edges": mo.ui.table(
                    keyword_edges.select(
                        "source", "target", "support", "jaccard", "is_bridge", "representative_record"
                    ).head(40),
                    selection=None,
                    pagination=True,
                    page_size=15,
                    show_search=True,
                    show_column_summaries=False,
                ),
                "Communities": mo.ui.table(
                    keyword_community_summary,
                    selection=None,
                    pagination=True,
                    page_size=20,
                    wrapped_columns=["leading_keywords"],
                    show_column_summaries=False,
                ),
            }),
        ]
    )
    keyword_section
    return


@app.cell(hide_code=True)
def explain_publisher_keyword(mo):
    mo.md("""
    ### Publisher–keyword coverage network

    This bipartite network links a publisher site to a query keyword when at least
    one indexed reviewed claim from that publisher matched the keyword. The
    within-publisher rate uses all reviewed claims from that publisher as its
    denominator; because keywords overlap, rates do not sum to 100%. Concentration
    summarizes indexed retrieval links—not editorial intent or a publisher's full
    archive.
    """)
    return


@app.cell
def build_publisher_keyword_network(keyword_link_frame, nx, pl, review_frame):
    publisher_keyword_edges = (
        keyword_link_frame
        .join(review_frame.select("review_url", "claim_text", "publisher_site"), on=["review_url", "claim_text"], how="left")
        .filter(pl.col("publisher_site").is_not_null())
        .group_by("publisher_site", "keyword")
        .len(name="support")
        .join(review_frame.group_by("publisher_site").len(name="publisher_records"), on="publisher_site", how="left")
        .with_columns(
            (100 * pl.col("support") / pl.col("publisher_records")).alias("within_publisher_pct"),
            (pl.lit("publisher::") + pl.col("publisher_site")).alias("source"),
            (pl.lit("keyword::") + pl.col("keyword")).alias("target"),
            pl.lit("publisher-keyword").alias("network"),
        )
        .sort(["support", "publisher_site", "keyword"], descending=[True, False, False])
    )
    publisher_link_totals = publisher_keyword_edges.group_by("publisher_site").agg(pl.col("support").sum().alias("keyword_links"))
    publisher_concentration = (
        publisher_keyword_edges
        .join(publisher_link_totals, on="publisher_site", how="left")
        .with_columns((pl.col("support") / pl.col("keyword_links")).alias("link_share"))
        .group_by("publisher_site")
        .agg(
            pl.col("support").sum().alias("keyword_links"),
            pl.col("keyword").n_unique().alias("keyword_breadth"),
            (pl.col("link_share") ** 2).sum().alias("keyword_hhi"),
        )
        .sort("keyword_links", descending=True)
    )
    keyword_concentration = (
        publisher_keyword_edges
        .with_columns((pl.col("support") / pl.col("support").sum().over("keyword")).alias("publisher_share"))
        .group_by("keyword")
        .agg(
            pl.col("support").sum().alias("records"),
            pl.col("publisher_site").n_unique().alias("publisher_breadth"),
            (pl.col("publisher_share") ** 2).sum().alias("publisher_hhi"),
        )
        .sort("records", descending=True)
    )
    bipartite_graph = nx.Graph()
    publishers = sorted(publisher_keyword_edges["source"].unique().to_list())
    keywords = sorted(publisher_keyword_edges["target"].unique().to_list())
    bipartite_graph.add_nodes_from(publishers, bipartite=0)
    bipartite_graph.add_nodes_from(keywords, bipartite=1)
    bipartite_graph.add_weighted_edges_from(
        [(row["source"], row["target"], row["support"]) for row in publisher_keyword_edges.iter_rows(named=True)],
        weight="support",
    )
    assert all(
        bipartite_graph.nodes[source]["bipartite"] != bipartite_graph.nodes[target]["bipartite"]
        for source, target in bipartite_graph.edges()
    )
    assert publisher_keyword_edges.select("source", "target").is_duplicated().sum() == 0
    assert publisher_keyword_edges.filter(pl.col("support") <= 0).height == 0

    top_keywords = keyword_concentration.head(20)["keyword"].to_list()
    visual_edges = (
        publisher_keyword_edges
        .filter(pl.col("keyword").is_in(top_keywords) & (pl.col("support") >= 5))
        .sort(["source", "support", "target"], descending=[False, True, False])
        .group_by("source", maintain_order=True)
        .head(5)
    )
    publisher_order = publisher_concentration["publisher_site"].to_list()
    keyword_order = top_keywords
    publisher_positions = {
        f"publisher::{publisher}": (0.0, float(index))
        for index, publisher in enumerate(publisher_order)
    }
    keyword_positions = {
        f"keyword::{keyword}": (1.0, float(index) * max(1, len(publisher_order) - 1) / max(1, len(keyword_order) - 1))
        for index, keyword in enumerate(keyword_order)
    }
    positions = {**publisher_positions, **keyword_positions}
    visual_edges = visual_edges.with_columns(
        pl.col("source").replace_strict({node: point[0] for node, point in positions.items()}).alias("x"),
        pl.col("source").replace_strict({node: point[1] for node, point in positions.items()}).alias("y"),
        pl.col("target").replace_strict({node: point[0] for node, point in positions.items()}).alias("x2"),
        pl.col("target").replace_strict({node: point[1] for node, point in positions.items()}).alias("y2"),
    )
    assert visual_edges.height <= publisher_keyword_edges.height
    visual_nodes = pl.DataFrame(
        [
            {
                "node_id": node,
                "label": node.split("::", 1)[1],
                "node_type": node.split("::", 1)[0],
                "x": point[0],
                "y": point[1],
                "strength": float(bipartite_graph.degree(node, weight="support")),
                "degree": int(bipartite_graph.degree(node)),
            }
            for node, point in positions.items()
        ]
    )
    return (
        keyword_concentration,
        publisher_concentration,
        publisher_keyword_edges,
        visual_edges,
        visual_nodes,
    )


@app.cell
def render_publisher_keyword_network(
    alt,
    keyword_concentration,
    mo,
    publisher_concentration,
    publisher_keyword_edges,
    visual_edges,
    visual_nodes,
):
    edge_layer = alt.Chart(visual_edges).mark_rule(opacity=0.25, color="#6B7280").encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), x2="x2:Q", y2="y2:Q",
        strokeWidth=alt.StrokeWidth("support:Q", scale=alt.Scale(range=[0.3, 5]), title="Reviewed claims"),
        tooltip=["publisher_site:N", "keyword:N", "support:Q", alt.Tooltip("within_publisher_pct:Q", format=".2f")],
    )
    node_layer = alt.Chart(visual_nodes).mark_circle(stroke="white", strokeWidth=0.7).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
        size=alt.Size("strength:Q", scale=alt.Scale(range=[70, 1000]), title="Link strength", legend=alt.Legend(orient="bottom")),
        color=alt.Color("node_type:N", title="Partition", legend=alt.Legend(orient="bottom")),
        tooltip=["node_id:N", "label:N", "node_type:N", "degree:Q", "strength:Q"],
    )
    labels = alt.Chart(visual_nodes).mark_text(
        align=alt.expr("datum.x < 0.5 ? 'right' : 'left'"), dx=alt.expr("datum.x < 0.5 ? -8 : 8"), fontSize=10
    ).encode(x="x:Q", y="y:Q", text="label:N")
    bipartite_chart = (edge_layer + node_layer + labels).properties(
        width="container", height=620, title="Publisher–keyword coverage backbone"
    )
    publisher_keyword_section = mo.vstack(
        [
            mo.md(
                f"The full bipartite graph contains **{publisher_keyword_edges['publisher_site'].n_unique()} publisher sites**, "
                f"**{publisher_keyword_edges['keyword'].n_unique()} keywords**, and "
                f"**{publisher_keyword_edges.height} observed publisher–keyword edges**."
            ),
            bipartite_chart,
            mo.ui.tabs({
                "Publisher concentration": mo.ui.table(
                    publisher_concentration,
                    selection=None,
                    pagination=True,
                    page_size=12,
                    show_column_summaries=False,
                ),
                "Keyword concentration": mo.ui.table(
                    keyword_concentration,
                    selection=None,
                    pagination=True,
                    page_size=20,
                    show_column_summaries=False,
                ),
            }),
        ]
    )
    publisher_keyword_section
    return


@app.cell(hide_code=True)
def explain_actor_quality(mo):
    mo.md("""
    ## 4. Actor and publisher associations

    The actor layer combines model-generated PERSON/ORGANIZATION spans with the
    structured claimant field. The model is not manually adjudicated and exposes
    no calibrated document-level confidence. Exact Unicode-normalized identity is
    the only alias merge; `Anies` and `Anies Baswedan`, for example, remain
    separate unless their normalized strings are equal.

    Structured claimants that are missing, publisher-like, domain-like,
    placeholders, or generic source/platform descriptions are excluded. Rejected
    values remain visible below. This prevents labels such as `Suara.com`,
    `Berbagai sumber`, or `Akun Facebook` from becoming central “actors.”
    """)
    return


@app.cell
def build_actor_networks(
    ACTOR_MIN_COMENTIONS,
    ACTOR_MIN_TEXTS,
    actor_occurrence_frame,
    analyze_undirected,
    backbone_frames,
    edge_rows_from_memberships,
    network_chart,
    nx,
    pl,
    review_frame,
):
    actor_support_frame = (
        actor_occurrence_frame
        .group_by("actor_id", "normalized_actor")
        .agg(
            pl.col("claim_text").n_unique().alias("support"),
            pl.col("display_name").sort().first().alias("label"),
            pl.col("provenance_flags").explode(empty_as_null=True).unique().sort().alias("provenance_flags"),
            pl.col("model_entity_types").explode(empty_as_null=True).drop_nulls().unique().sort().alias("model_entity_types"),
            pl.col("max_ner_score").max().alias("max_ner_score"),
        )
        .with_columns(pl.lit("actor").alias("node_type"))
        .sort(["support", "actor_id"], descending=[True, False])
    )
    eligible_actor_nodes = actor_support_frame.filter(pl.col("support") >= ACTOR_MIN_TEXTS)
    eligible_actor_ids = eligible_actor_nodes["actor_id"].to_list()
    primary_actor_occurrences = actor_occurrence_frame.filter(pl.col("actor_id").is_in(eligible_actor_ids))
    actor_memberships = [
        (row["claim_text"], row["actor_ids"])
        for row in primary_actor_occurrences.group_by("claim_text").agg(pl.col("actor_id").unique().sort().alias("actor_ids")).iter_rows(named=True)
    ]
    actor_support = dict(zip(eligible_actor_nodes["actor_id"].to_list(), eligible_actor_nodes["support"].to_list()))
    actor_edge_all = pl.DataFrame(
        edge_rows_from_memberships(actor_memberships, actor_support, similarity=False)
    ).sort(["support", "source", "target"], descending=[True, False, False])
    actor_edges = actor_edge_all.filter(pl.col("support") >= ACTOR_MIN_COMENTIONS)
    actor_nodes = eligible_actor_nodes.select(
        "actor_id", "label", "node_type", "support", "normalized_actor",
        "provenance_flags", "model_entity_types", "max_ner_score",
    ).rename({"actor_id": "node_id"})
    actor_node_metrics, actor_summary, actor_graph = analyze_undirected(
        actor_nodes, actor_edges, exact_betweenness=False
    )
    actor_visual_nodes, actor_visual_edges = backbone_frames(
        actor_node_metrics, actor_edges, max_nodes=45, neighbors_per_node=3
    )
    actor_visual_nodes = actor_visual_nodes.with_columns(
        pl.col("provenance_flags").list.join(" + ").alias("provenance_group")
    )
    actor_chart = network_chart(
        actor_visual_nodes,
        actor_visual_edges,
        "Actor co-mention backbone",
        "provenance_group",
        "Provenance",
    )

    actor_publisher_links = (
        review_frame
        .select("record_id", "review_url", "claim_text", "publisher_site")
        .join(primary_actor_occurrences.select("claim_text", "actor_id").unique(), on="claim_text", how="inner")
        .filter(pl.col("publisher_site").is_not_null())
        .unique(subset=["record_id", "actor_id", "publisher_site"])
    )
    actor_publisher_edges = (
        actor_publisher_links
        .group_by("actor_id", "publisher_site")
        .len(name="support")
        .with_columns(
            pl.col("actor_id").alias("source"),
            (pl.lit("publisher::") + pl.col("publisher_site")).alias("target"),
            pl.lit("actor-publisher").alias("network"),
        )
        .sort(["support", "actor_id", "publisher_site"], descending=[True, False, False])
    )
    actor_publisher_graph = nx.Graph()
    actor_publisher_graph.add_nodes_from(eligible_actor_ids, bipartite=0)
    publisher_ids = [f"publisher::{site}" for site in sorted(actor_publisher_edges["publisher_site"].unique().to_list())]
    actor_publisher_graph.add_nodes_from(publisher_ids, bipartite=1)
    actor_publisher_graph.add_weighted_edges_from(
        [(row["source"], row["target"], row["support"]) for row in actor_publisher_edges.iter_rows(named=True)],
        weight="support",
    )
    assert all(
        actor_publisher_graph.nodes[source]["bipartite"] != actor_publisher_graph.nodes[target]["bipartite"]
        for source, target in actor_publisher_graph.edges()
    )
    actor_breadth = (
        actor_publisher_edges
        .group_by("actor_id")
        .agg(
            pl.col("support").sum().alias("reviewed_claims"),
            pl.col("publisher_site").n_unique().alias("publisher_breadth"),
        )
    )
    actor_node_metrics = actor_node_metrics.join(actor_breadth, left_on="node_id", right_on="actor_id", how="left").with_columns(
        pl.col("reviewed_claims").fill_null(0), pl.col("publisher_breadth").fill_null(0)
    )
    actor_bridge_pairs = {tuple(sorted(edge)) for edge in nx.bridges(actor_graph)}
    actor_edges = actor_edges.with_columns(
        pl.struct("source", "target")
        .map_elements(
            lambda row: tuple(sorted((row["source"], row["target"]))) in actor_bridge_pairs,
            return_dtype=pl.Boolean,
        )
        .alias("is_bridge")
    )
    actor_publisher_actor_ids = actor_node_metrics.sort(
        ["reviewed_claims", "strength", "node_id"], descending=[True, True, False]
    ).head(20)["node_id"].to_list()
    actor_publisher_visual_edges = (
        actor_publisher_edges
        .filter(pl.col("actor_id").is_in(actor_publisher_actor_ids) & (pl.col("support") >= 2))
        .sort(["source", "support", "target"], descending=[False, True, False])
        .group_by("source", maintain_order=True)
        .head(2)
    )
    actor_publisher_actor_order = (
        actor_node_metrics
        .filter(pl.col("node_id").is_in(actor_publisher_actor_ids))
        .sort(["reviewed_claims", "node_id"], descending=[True, False])["node_id"]
        .to_list()
    )
    actor_publisher_site_order = sorted(actor_publisher_edges["publisher_site"].unique().to_list())
    actor_positions = {
        actor_id: (0.0, float(index) * max(1, len(actor_publisher_site_order) - 1) / max(1, len(actor_publisher_actor_order) - 1))
        for index, actor_id in enumerate(actor_publisher_actor_order)
    }
    actor_publisher_positions = {
        f"publisher::{site}": (1.0, float(index))
        for index, site in enumerate(actor_publisher_site_order)
    }
    actor_publisher_positions = {**actor_positions, **actor_publisher_positions}
    actor_publisher_visual_edges = actor_publisher_visual_edges.with_columns(
        pl.col("source").replace_strict({node: point[0] for node, point in actor_publisher_positions.items()}).alias("x"),
        pl.col("source").replace_strict({node: point[1] for node, point in actor_publisher_positions.items()}).alias("y"),
        pl.col("target").replace_strict({node: point[0] for node, point in actor_publisher_positions.items()}).alias("x2"),
        pl.col("target").replace_strict({node: point[1] for node, point in actor_publisher_positions.items()}).alias("y2"),
    )
    actor_label_map = dict(zip(actor_node_metrics["node_id"].to_list(), actor_node_metrics["label"].to_list()))
    actor_publisher_visual_nodes = pl.DataFrame(
        [
            {
                "node_id": node_id,
                "label": actor_label_map.get(node_id, node_id.removeprefix("publisher::")),
                "node_type": node_id.split("::", 1)[0],
                "x": point[0],
                "y": point[1],
                "strength": float(actor_publisher_graph.degree(node_id, weight="support")),
                "degree": int(actor_publisher_graph.degree(node_id)),
            }
            for node_id, point in actor_publisher_positions.items()
        ]
    )
    assert actor_publisher_visual_edges.height <= actor_publisher_edges.height
    assert actor_edges.select("source", "target").is_duplicated().sum() == 0
    assert all(
        row["support"] <= min(actor_support[row["source"]], actor_support[row["target"]])
        for row in actor_edge_all.iter_rows(named=True)
    )
    assert actor_edges.filter(pl.col("source") >= pl.col("target")).height == 0
    assert actor_edges.filter((pl.col("support") <= 0) | (pl.col("source") == pl.col("target"))).height == 0
    assert actor_publisher_edges.select("source", "target").is_duplicated().sum() == 0
    assert actor_publisher_edges.filter(pl.col("support") <= 0).height == 0
    return (
        actor_chart,
        actor_edge_all,
        actor_edges,
        actor_node_metrics,
        actor_publisher_edges,
        actor_publisher_links,
        actor_publisher_visual_edges,
        actor_publisher_visual_nodes,
        actor_summary,
        actor_support_frame,
        eligible_actor_ids,
        primary_actor_occurrences,
    )


@app.cell
def render_actor_quality_and_networks(
    ACTOR_MIN_COMENTIONS,
    ACTOR_MIN_TEXTS,
    actor_chart,
    actor_edges,
    actor_node_metrics,
    actor_occurrence_frame,
    actor_publisher_edges,
    actor_publisher_visual_edges,
    actor_publisher_visual_nodes,
    actor_summary,
    alt,
    mo,
    ner_span_frame,
    pl,
    rejected_claimant_frame,
    review_frame,
):
    exclusion_summary = (
        rejected_claimant_frame
        .group_by("exclusion_reason", "claimant", "normalized_claimant")
        .len(name="reviewed_claims")
        .sort(["reviewed_claims", "claimant"], descending=[True, False])
    )
    ner_label_summary = (
        ner_span_frame
        .group_by("entity_type", "raw_label")
        .agg(
            pl.len().alias("spans"),
            pl.col("claim_text").n_unique().alias("claim_texts"),
            pl.col("score").median().alias("median_model_score"),
        )
        .sort("spans", descending=True)
    )
    ner_actor_audit = (
        ner_span_frame
        .filter(pl.col("entity_type").is_in(["PERSON", "ORGANIZATION"]))
        .sort(["score", "claim_text"], descending=[False, False])
        .select(
            "entity_type", "mention_text", "normalized_text", "score", "start_char", "end_char", "claim_text"
        )
        .head(100)
    )
    actor_publisher_edge_layer = alt.Chart(actor_publisher_visual_edges).mark_rule(
        opacity=0.25, color="#6B7280"
    ).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), x2="x2:Q", y2="y2:Q",
        strokeWidth=alt.StrokeWidth("support:Q", scale=alt.Scale(range=[0.3, 5]), title="Reviewed claims"),
        tooltip=["actor_id:N", "publisher_site:N", "support:Q"],
    )
    actor_publisher_node_layer = alt.Chart(actor_publisher_visual_nodes).mark_circle(
        stroke="white", strokeWidth=0.7
    ).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
        size=alt.Size("strength:Q", scale=alt.Scale(range=[60, 900]), title="Coverage strength", legend=alt.Legend(orient="bottom")),
        color=alt.Color("node_type:N", title="Partition", legend=alt.Legend(orient="bottom")),
        tooltip=["node_id:N", "label:N", "node_type:N", "degree:Q", "strength:Q"],
    )
    actor_publisher_labels = alt.Chart(actor_publisher_visual_nodes).mark_text(
        align=alt.expr("datum.x < 0.5 ? 'right' : 'left'"),
        dx=alt.expr("datum.x < 0.5 ? -8 : 8"),
        fontSize=10,
    ).encode(x="x:Q", y="y:Q", text="label:N")
    actor_publisher_chart = (
        actor_publisher_edge_layer + actor_publisher_node_layer + actor_publisher_labels
    ).properties(
        width="container",
        height=620,
        title="Actor–publisher coverage backbone",
    )

    actor_summary_row = actor_summary.row(0, named=True)
    actor_section_top_actor = actor_node_metrics.row(0, named=True)
    actor_community_summary = (
        actor_node_metrics
        .sort(["community_id", "strength", "label"], descending=[False, True, False])
        .group_by("community_id", maintain_order=True)
        .agg(
            pl.len().alias("nodes"),
            pl.col("strength").sum().alias("total_strength"),
            pl.col("label").head(10).alias("leading_actors"),
        )
        .sort(["total_strength", "community_id"], descending=[True, False])
    )
    actor_details = mo.ui.tabs({
        "Communities": mo.ui.table(
            actor_community_summary,
            selection=None,
            pagination=True,
            page_size=20,
            wrapped_columns=["leading_actors"],
            show_column_summaries=False,
        ),
        "Actor metrics": mo.ui.table(
            actor_node_metrics.select(
                "label", "normalized_actor", "support", "degree", "strength",
                "betweenness", "publisher_breadth", "provenance_flags",
                "model_entity_types", "community_id", "component_id", "is_articulation",
            ).head(50),
            selection=None, pagination=True, page_size=20, show_search=True,
            wrapped_columns=["provenance_flags", "model_entity_types"],
            show_column_summaries=False,
        ),
        "Strongest co-mentions": mo.ui.table(
            actor_edges.select(
                "source", "target", "support", "is_bridge", "representative_record"
            ).head(60),
            selection=None, pagination=True, page_size=20, show_search=True,
            show_column_summaries=False,
        ),
    })
    actor_audit_section = mo.vstack([
        mo.md(
            f"Claimant is missing for **{review_frame['claimant'].null_count():,} of "
            f"{review_frame.height:,} reviewed claims**. Exclusions are retained rather "
            "than silently dropped."
        ),
        mo.ui.tabs({
            "Structured claimant exclusions": mo.ui.table(
                exclusion_summary,
                selection=None,
                pagination=True,
                page_size=25,
                show_search=True,
                show_column_summaries=False,
            ),
            "NER label summary": mo.ui.table(
                ner_label_summary,
                selection=None,
                pagination=True,
                page_size=20,
                show_column_summaries=False,
            ),
            "Low-score span audit": mo.ui.table(
                ner_actor_audit,
                selection=None,
                pagination=True,
                page_size=20,
                show_search=True,
                wrapped_columns=["claim_text"],
                show_column_summaries=False,
            ),
        }),
    ])
    actor_section = mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{actor_occurrence_frame['actor_id'].n_unique():,}", label="All actor candidates", bordered=True),
                    mo.stat(f"{actor_summary_row['nodes']:,}", label=f"Actors with ≥{ACTOR_MIN_TEXTS} texts", bordered=True),
                    mo.stat(f"{actor_summary_row['edges']:,}", label=f"Co-mention edges ≥{ACTOR_MIN_COMENTIONS}", bordered=True),
                    mo.stat(f"{actor_publisher_edges.height:,}", label="Actor–publisher edges", bordered=True),
                ], widths="equal", wrap=True,
            ),
            mo.callout(
                mo.md(
                    "Actor labels combine model-generated PERSON/ORGANIZATION mentions with "
                    "conservatively retained structured claimants. They are mention candidates, "
                    "not verified identities."
                ),
                kind="warn",
                title="Model-generated actor layer",
            ),
            mo.md(
                f"The strongest retained co-mention node is **{actor_section_top_actor['label']}** "
                f"(support {actor_section_top_actor['support']:,}; edge strength "
                f"{actor_section_top_actor['strength']:.0f}; publisher breadth "
                f"{actor_section_top_actor['publisher_breadth']}). Betweenness is exact for graphs "
                "up to 200 nodes and otherwise uses a deterministic 200-node approximation."
            ),
            mo.md("### Actor co-mention network"),
            actor_chart,
            actor_details,
            mo.md("### Actor–publisher coverage network"),
            actor_publisher_chart,
            mo.ui.table(
                actor_publisher_edges.select("actor_id", "publisher_site", "support").head(60),
                selection=None,
                pagination=True,
                page_size=20,
                show_search=True,
                show_column_summaries=False,
            ),
        ]
    )
    actor_section
    return actor_audit_section, exclusion_summary, ner_actor_audit, ner_label_summary


@app.cell(hide_code=True)
def explain_sensitivity(mo):
    sensitivity_explanation = mo.md("""
    Network conclusions can be artifacts of pruning. Keyword graphs are recomputed
    at shared-record support `1/2/5/10`; actor graphs cross actor text-frequency
    thresholds `3/5/10` with co-mention support `1/2/5`. Primary settings are
    highlighted, but no threshold is presented as a natural law.
    """)
    return (sensitivity_explanation,)


@app.cell
def compute_sensitivity(
    ACTOR_MIN_COMENTIONS,
    ACTOR_MIN_TEXTS,
    actor_occurrence_frame,
    actor_support_frame,
    edge_rows_from_memberships,
    graph_summary_only,
    keyword_edge_all,
    keyword_nodes,
    pl,
):
    keyword_sensitivity_rows = []
    keyword_node_ids = keyword_nodes["node_id"].to_list()
    for sensitivity_threshold in [1, 2, 5, 10]:
        sensitivity_edges = keyword_edge_all.filter(pl.col("support") >= sensitivity_threshold)
        sensitivity_summary, _, _ = graph_summary_only(keyword_node_ids, sensitivity_edges)
        keyword_sensitivity_rows.append(
            {"shared_claim_threshold": sensitivity_threshold, "is_primary": sensitivity_threshold == 5, **sensitivity_summary}
        )
    keyword_sensitivity = pl.DataFrame(keyword_sensitivity_rows)

    actor_sensitivity_rows = []
    for node_threshold in [3, 5, 10]:
        actor_ids = actor_support_frame.filter(pl.col("support") >= node_threshold)["actor_id"].to_list()
        occurrences = actor_occurrence_frame.filter(pl.col("actor_id").is_in(actor_ids))
        memberships = [
            (row["claim_text"], row["actor_ids"])
            for row in occurrences.group_by("claim_text").agg(pl.col("actor_id").unique().sort().alias("actor_ids")).iter_rows(named=True)
        ]
        support = {
            row["actor_id"]: row["support"]
            for row in actor_support_frame.filter(pl.col("actor_id").is_in(actor_ids)).iter_rows(named=True)
        }
        sensitivity_all_edges = pl.DataFrame(edge_rows_from_memberships(memberships, support, similarity=False))
        for edge_threshold in [1, 2, 5]:
            sensitivity_actor_edges = sensitivity_all_edges.filter(pl.col("support") >= edge_threshold)
            sensitivity_actor_summary, _, _ = graph_summary_only(actor_ids, sensitivity_actor_edges)
            actor_sensitivity_rows.append(
                {
                    "actor_text_threshold": node_threshold,
                    "co_mention_threshold": edge_threshold,
                    "is_primary": node_threshold == ACTOR_MIN_TEXTS and edge_threshold == ACTOR_MIN_COMENTIONS,
                    **sensitivity_actor_summary,
                }
            )
    actor_sensitivity = pl.DataFrame(actor_sensitivity_rows)
    return actor_sensitivity, keyword_sensitivity


@app.cell
def render_sensitivity(
    actor_sensitivity,
    alt,
    keyword_sensitivity,
    mo,
    sensitivity_explanation,
):
    sensitivity_keyword_chart = alt.Chart(keyword_sensitivity).mark_line(point=True, color="#4C78A8").encode(
        x=alt.X("shared_claim_threshold:O", title="Minimum shared reviewed claims"),
        y=alt.Y("edges:Q", title="Retained edges"),
        tooltip=["shared_claim_threshold:O", "nodes:Q", "active_nodes:Q", "edges:Q", alt.Tooltip("density:Q", format=".4f"), "components:Q"],
    ).properties(width="container", height=280, title="Keyword threshold sensitivity")
    sensitivity_actor_chart = alt.Chart(actor_sensitivity).mark_line(point=True).encode(
        x=alt.X("co_mention_threshold:O", title="Minimum co-mentioning texts"),
        y=alt.Y("edges:Q", title="Retained edges"),
        color=alt.Color("actor_text_threshold:O", title="Actor text threshold"),
        tooltip=["actor_text_threshold:O", "co_mention_threshold:O", "nodes:Q", "active_nodes:Q", "edges:Q", alt.Tooltip("density:Q", format=".4f"), "components:Q"],
    ).properties(width="container", height=280, title="Actor threshold sensitivity")
    sensitivity_section = mo.vstack(
        [
            sensitivity_explanation,
            mo.vstack([
                sensitivity_keyword_chart,
                sensitivity_actor_chart,
            ]),
            mo.hstack(
                [
                    mo.ui.table(keyword_sensitivity, selection=None, pagination=False, show_column_summaries=False),
                    mo.ui.table(actor_sensitivity, selection=None, pagination=False, show_column_summaries=False),
                ], widths="equal", align="start", wrap=True,
            ),
        ]
    )
    return (sensitivity_section,)


@app.cell(hide_code=True)
def explain_temporal_networks(mo):
    mo.md("""
    ## 5. How network structure changes over time

    Two designs complement each other:

    1. **Calendar years 2019–2024**, each with at least 100 dated reviewed claims.
       Sparse earlier years and 2025 are listed as excluded. A fixed node universe
       within each network family makes density and turnover comparable.
    2. **Matched three-year windows** around 30 November 2022: pre-launch
       `[2019-11-30, 2022-11-30)` and post-launch
       `[2022-11-30, 2025-11-30)`.

    Temporal coexistence is not a causal intervention. Edge formation means an
    association crossed the fixed support threshold in the observed extract; it
    does not mean a new real-world relationship began.
    """)
    return


@app.cell
def compute_temporal_networks(
    ACTOR_MIN_COMENTIONS,
    BREAKPOINT,
    KEYWORD_MIN_SHARED,
    POST_END,
    PRE_START,
    YEARLY_LAST_YEAR,
    YEARLY_MIN_RECORDS,
    actor_publisher_links,
    edge_rows_from_memberships,
    eligible_actor_ids,
    graph_summary_only,
    keyword_nodes,
    normalized_mutual_info_score,
    pl,
    primary_actor_occurrences,
    review_frame,
):
    dated_review = review_frame.filter(pl.col("claim_dt").is_not_null())
    yearly_counts = (
        dated_review
        .filter(pl.col("claim_year") <= YEARLY_LAST_YEAR)
        .group_by("claim_year")
        .len(name="reviewed_claims")
        .sort("claim_year")
    )
    eligible_years = yearly_counts.filter(pl.col("reviewed_claims") >= YEARLY_MIN_RECORDS)["claim_year"].to_list()
    excluded_years = yearly_counts.filter(pl.col("reviewed_claims") < YEARLY_MIN_RECORDS).with_columns(
        pl.lit(f"fewer than {YEARLY_MIN_RECORDS} dated reviewed claims").alias("reason")
    )

    keyword_universe = keyword_nodes["node_id"].to_list()
    actor_universe = sorted(eligible_actor_ids)

    def keyword_period_edges(period_review: pl.DataFrame) -> pl.DataFrame:
        support = collections.Counter()
        memberships = []
        for row in period_review.select("record_id", "matched_keywords").iter_rows(named=True):
            members = [f"keyword::{keyword}" for keyword in row["matched_keywords"]]
            support.update(set(members))
            memberships.append((row["record_id"], members))
        if not support:
            return pl.DataFrame(schema={"source": pl.Utf8, "target": pl.Utf8, "support": pl.Int64, "jaccard": pl.Float64, "distance": pl.Float64, "representative_record": pl.Utf8})
        return pl.DataFrame(edge_rows_from_memberships(memberships, dict(support), similarity=True))

    def actor_period_edges(period_review: pl.DataFrame) -> pl.DataFrame:
        content_texts = period_review.select("claim_text").unique()["claim_text"].to_list()
        occurrences = primary_actor_occurrences.filter(pl.col("claim_text").is_in(content_texts))
        support = {
            row["actor_id"]: row["support"]
            for row in occurrences.group_by("actor_id").agg(pl.col("claim_text").n_unique().alias("support")).iter_rows(named=True)
        }
        memberships = [
            (row["claim_text"], row["actor_ids"])
            for row in occurrences.group_by("claim_text").agg(pl.col("actor_id").unique().sort().alias("actor_ids")).iter_rows(named=True)
        ]
        if not memberships:
            return pl.DataFrame(schema={"source": pl.Utf8, "target": pl.Utf8, "support": pl.Int64, "jaccard": pl.Null, "distance": pl.Float64, "representative_record": pl.Utf8})
        return pl.DataFrame(edge_rows_from_memberships(memberships, support, similarity=False))

    temporal_rows = []
    temporal_edge_sets: dict[tuple[str, str], set[tuple[str, str]]] = {}
    yearly_strengths: dict[tuple[str, str], dict[str, float]] = {}
    for year in eligible_years:
        period_review = dated_review.filter(pl.col("claim_year") == year)
        for family, universe, period_all_edges, period_threshold in [
            ("keyword", keyword_universe, keyword_period_edges(period_review), KEYWORD_MIN_SHARED),
            ("actor co-mention", actor_universe, actor_period_edges(period_review), ACTOR_MIN_COMENTIONS),
        ]:
            period_edges = period_all_edges.filter(pl.col("support") >= period_threshold)
            period_summary, _, edge_set = graph_summary_only(universe, period_edges)
            period_strength = {node: 0.0 for node in universe}
            for edge in period_edges.iter_rows(named=True):
                period_strength[edge["source"]] += edge["support"]
                period_strength[edge["target"]] += edge["support"]
            temporal_edge_sets[(family, str(year))] = edge_set
            yearly_strengths[(family, str(year))] = period_strength
            temporal_rows.append(
                {
                    "design": "calendar year",
                    "period": str(year),
                    "network_family": family,
                    "reviewed_claims": period_review.height,
                    **period_summary,
                }
            )
    temporal_metrics = pl.DataFrame(temporal_rows).sort("network_family", "period")

    turnover_rows = []
    for family in ["keyword", "actor co-mention"]:
        previous_edges: set[tuple[str, str]] | None = None
        previous_period = None
        for year in eligible_years:
            current_edges = temporal_edge_sets[(family, str(year))]
            if previous_edges is not None:
                persistent = previous_edges & current_edges
                formed = current_edges - previous_edges
                disappeared = previous_edges - current_edges
                turnover_rows.append(
                    {
                        "design": "consecutive years",
                        "network_family": family,
                        "from_period": previous_period,
                        "to_period": str(year),
                        "persistent_edges": len(persistent),
                        "formed_edges": len(formed),
                        "disappeared_edges": len(disappeared),
                        "edge_jaccard": len(persistent) / len(previous_edges | current_edges) if previous_edges | current_edges else 1.0,
                    }
                )
            previous_edges = current_edges
            previous_period = str(year)
    yearly_turnover = pl.DataFrame(turnover_rows)
    yearly_rank_change_rows = []
    for family, universe in [("keyword", keyword_universe), ("actor co-mention", actor_universe)]:
        previous_rank = None
        previous_year = None
        for year in eligible_years:
            strengths = yearly_strengths[(family, str(year))]
            current_rank = {
                node: rank
                for rank, node in enumerate(
                    sorted(universe, key=lambda node: (-strengths[node], node)), start=1
                )
            }
            if previous_rank is not None:
                for node in universe:
                    if strengths[node] > 0 or yearly_strengths[(family, str(previous_year))][node] > 0:
                        yearly_rank_change_rows.append(
                            {
                                "network_family": family,
                                "from_year": previous_year,
                                "to_year": year,
                                "node_id": node,
                                "from_rank": previous_rank[node],
                                "to_rank": current_rank[node],
                                "rank_change": previous_rank[node] - current_rank[node],
                                "from_strength": yearly_strengths[(family, str(previous_year))][node],
                                "to_strength": strengths[node],
                            }
                        )
            previous_rank = current_rank
            previous_year = year
    yearly_rank_change = pl.DataFrame(yearly_rank_change_rows).sort(
        ["network_family", "to_year", "rank_change", "to_strength"],
        descending=[False, False, True, True],
    )

    window_reviews = {
        "Pre-launch": dated_review.filter((pl.col("claim_dt") >= PRE_START) & (pl.col("claim_dt") < BREAKPOINT)),
        "Post-launch": dated_review.filter((pl.col("claim_dt") >= BREAKPOINT) & (pl.col("claim_dt") < POST_END)),
    }
    assert window_reviews["Pre-launch"].height > 0 and window_reviews["Post-launch"].height > 0
    assert window_reviews["Pre-launch"].filter(pl.col("claim_dt") >= BREAKPOINT).height == 0
    assert window_reviews["Post-launch"].filter(pl.col("claim_dt") < BREAKPOINT).height == 0

    window_rows = []
    window_edge_sets: dict[tuple[str, str], set[tuple[str, str]]] = {}
    window_edge_frames: dict[tuple[str, str], pl.DataFrame] = {}
    window_communities: dict[tuple[str, str], dict[str, int]] = {}
    window_strengths: dict[tuple[str, str], dict[str, float]] = {}
    for period, period_review in window_reviews.items():
        for family, universe, window_all_edges, window_threshold in [
            ("keyword", keyword_universe, keyword_period_edges(period_review), KEYWORD_MIN_SHARED),
            ("actor co-mention", actor_universe, actor_period_edges(period_review), ACTOR_MIN_COMENTIONS),
        ]:
            window_edges = window_all_edges.filter(pl.col("support") >= window_threshold)
            window_summary, community_map, edge_set = graph_summary_only(universe, window_edges)
            graph_strength = {node: 0.0 for node in universe}
            for edge in window_edges.iter_rows(named=True):
                graph_strength[edge["source"]] += edge["support"]
                graph_strength[edge["target"]] += edge["support"]
            window_edge_sets[(family, period)] = edge_set
            window_edge_frames[(family, period)] = window_edges
            window_communities[(family, period)] = community_map
            window_strengths[(family, period)] = graph_strength
            window_rows.append(
                {
                    "design": "matched launch window",
                    "period": period,
                    "network_family": family,
                    "reviewed_claims": period_review.height,
                    **window_summary,
                }
            )
    window_metrics = pl.DataFrame(window_rows)

    window_turnover_rows = []
    rank_change_rows = []
    for family, universe in [("keyword", keyword_universe), ("actor co-mention", actor_universe)]:
        pre_edges = window_edge_sets[(family, "Pre-launch")]
        post_edges = window_edge_sets[(family, "Post-launch")]
        persistent = pre_edges & post_edges
        shared_active_nodes = [
            node for node in universe
            if window_strengths[(family, "Pre-launch")][node] > 0
            and window_strengths[(family, "Post-launch")][node] > 0
        ]
        pre_labels = [window_communities[(family, "Pre-launch")][node] for node in shared_active_nodes]
        post_labels = [window_communities[(family, "Post-launch")][node] for node in shared_active_nodes]
        window_turnover_rows.append(
            {
                "design": "matched launch windows",
                "network_family": family,
                "from_period": "Pre-launch",
                "to_period": "Post-launch",
                "persistent_edges": len(persistent),
                "formed_edges": len(post_edges - pre_edges),
                "disappeared_edges": len(pre_edges - post_edges),
                "edge_jaccard": len(persistent) / len(pre_edges | post_edges) if pre_edges | post_edges else 1.0,
                "community_nmi": normalized_mutual_info_score(pre_labels, post_labels) if len(shared_active_nodes) >= 2 else None,
                "community_nmi_nodes": len(shared_active_nodes),
            }
        )
        pre_strength = window_strengths[(family, "Pre-launch")]
        post_strength = window_strengths[(family, "Post-launch")]
        pre_rank = {node: rank for rank, node in enumerate(sorted(universe, key=lambda node: (-pre_strength[node], node)), start=1)}
        post_rank = {node: rank for rank, node in enumerate(sorted(universe, key=lambda node: (-post_strength[node], node)), start=1)}
        for node in universe:
            rank_change_rows.append(
                {
                    "network_family": family,
                    "node_id": node,
                    "pre_strength": pre_strength[node],
                    "post_strength": post_strength[node],
                    "pre_rank": pre_rank[node],
                    "post_rank": post_rank[node],
                    "rank_change": pre_rank[node] - post_rank[node],
                }
            )
    window_turnover = pl.DataFrame(window_turnover_rows)
    window_rank_change = pl.DataFrame(rank_change_rows).sort(
        ["network_family", "rank_change", "post_strength"], descending=[False, True, True]
    )
    window_edge_change_rows = []
    for family in ["keyword", "actor co-mention"]:
        pre_support = {
            (row["source"], row["target"]): row["support"]
            for row in window_edge_frames[(family, "Pre-launch")].iter_rows(named=True)
        }
        post_support = {
            (row["source"], row["target"]): row["support"]
            for row in window_edge_frames[(family, "Post-launch")].iter_rows(named=True)
        }
        for source, target in sorted(set(pre_support) | set(post_support)):
            pre_value = pre_support.get((source, target), 0)
            post_value = post_support.get((source, target), 0)
            window_edge_change_rows.append(
                {
                    "network_family": family,
                    "source": source,
                    "target": target,
                    "pre_support": pre_value,
                    "post_support": post_value,
                    "support_change": post_value - pre_value,
                    "status": "persistent" if pre_value and post_value else ("emerged" if post_value else "disappeared"),
                }
            )

    publisher_temporal_rows = []
    publisher_edge_sets: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    publisher_edge_frames: dict[tuple[str, str, str], tuple[pl.DataFrame, str, str]] = {}
    for design, periods in [
        ("calendar year", {str(year): dated_review.filter(pl.col("claim_year") == year) for year in eligible_years}),
        ("matched launch window", window_reviews),
    ]:
        for period, period_review in periods.items():
            publisher_keyword = (
                period_review
                .select("record_id", "publisher_site", "matched_keywords")
                .explode("matched_keywords", empty_as_null=True)
                .filter(pl.col("publisher_site").is_not_null())
                .group_by("publisher_site", "matched_keywords")
                .len(name="support")
            )
            actor_publisher = (
                actor_publisher_links
                .join(period_review.select("record_id").unique(), on="record_id", how="semi")
                .group_by("actor_id", "publisher_site")
                .len(name="support")
            )
            for family, edge_data, left, right in [
                ("publisher-keyword", publisher_keyword, "publisher_site", "matched_keywords"),
                ("actor-publisher", actor_publisher, "actor_id", "publisher_site"),
            ]:
                edge_set = {(row[left], row[right]) for row in edge_data.iter_rows(named=True)}
                publisher_edge_sets[(design, period, family)] = edge_set
                publisher_edge_frames[(design, period, family)] = (edge_data, left, right)
                publisher_temporal_rows.append(
                    {
                        "design": design,
                        "period": period,
                        "network_family": family,
                        "edges": len(edge_set),
                        "total_support": int(edge_data["support"].sum()) if edge_data.height else 0,
                    }
                )
    publisher_temporal = pl.DataFrame(publisher_temporal_rows)
    publisher_window_changes = []
    for family in ["publisher-keyword", "actor-publisher"]:
        pre_edges = publisher_edge_sets[("matched launch window", "Pre-launch", family)]
        post_edges = publisher_edge_sets[("matched launch window", "Post-launch", family)]
        publisher_window_changes.append(
            {
                "network_family": family,
                "pre_edges": len(pre_edges),
                "post_edges": len(post_edges),
                "persistent_edges": len(pre_edges & post_edges),
                "formed_edges": len(post_edges - pre_edges),
                "disappeared_edges": len(pre_edges - post_edges),
            }
        )
    publisher_window_change = pl.DataFrame(publisher_window_changes)
    for family in ["publisher-keyword", "actor-publisher"]:
        pre_data, left, right = publisher_edge_frames[("matched launch window", "Pre-launch", family)]
        post_data, _, _ = publisher_edge_frames[("matched launch window", "Post-launch", family)]
        pre_support = {(row[left], row[right]): row["support"] for row in pre_data.iter_rows(named=True)}
        post_support = {(row[left], row[right]): row["support"] for row in post_data.iter_rows(named=True)}
        for source_value, target_value in sorted(set(pre_support) | set(post_support)):
            pre_value = pre_support.get((source_value, target_value), 0)
            post_value = post_support.get((source_value, target_value), 0)
            source = source_value if "::" in source_value else (
                f"publisher::{source_value}" if family == "publisher-keyword" else source_value
            )
            target = target_value if "::" in target_value else (
                f"keyword::{target_value}" if family == "publisher-keyword" else f"publisher::{target_value}"
            )
            window_edge_change_rows.append(
                {
                    "network_family": family,
                    "source": source,
                    "target": target,
                    "pre_support": pre_value,
                    "post_support": post_value,
                    "support_change": post_value - pre_value,
                    "status": "persistent" if pre_value and post_value else ("emerged" if post_value else "disappeared"),
                }
            )
    window_edge_changes = pl.DataFrame(window_edge_change_rows).sort(
        ["support_change", "post_support", "source", "target"], descending=[True, True, False, False]
    )

    temporal_metrics_all = pl.concat([temporal_metrics, window_metrics], how="diagonal_relaxed")
    turnover_all = pl.concat([yearly_turnover, window_turnover], how="diagonal_relaxed")
    return (
        eligible_years,
        excluded_years,
        publisher_temporal,
        publisher_window_change,
        temporal_metrics_all,
        turnover_all,
        window_metrics,
        window_edge_changes,
        window_rank_change,
        window_turnover,
        yearly_counts,
        yearly_rank_change,
    )


@app.cell
def render_temporal_networks(
    alt,
    eligible_years,
    excluded_years,
    mo,
    pl,
    publisher_temporal,
    publisher_window_change,
    temporal_metrics_all,
    turnover_all,
    window_edge_changes,
    window_metrics,
    window_rank_change,
    window_turnover,
    yearly_counts,
    yearly_rank_change,
):
    yearly_metric_frame = temporal_metrics_all.filter(pl.col("design") == "calendar year")
    edge_chart = alt.Chart(yearly_metric_frame).mark_line(point=True).encode(
        x=alt.X("period:O", title="Eligible calendar year"),
        y=alt.Y("edges:Q", title="Thresholded edges"),
        color=alt.Color("network_family:N", title="Network"),
        tooltip=["period:O", "network_family:N", "reviewed_claims:Q", "active_nodes:Q", "edges:Q", alt.Tooltip("density:Q", format=".4f"), "components:Q"],
    ).properties(width="container", height=320, title="Yearly thresholded network size")
    modularity_chart = alt.Chart(yearly_metric_frame).mark_line(point=True).encode(
        x=alt.X("period:O", title="Eligible calendar year"),
        y=alt.Y("modularity:Q", title="Greedy modularity", scale=alt.Scale(zero=False)),
        color=alt.Color("network_family:N", title="Network"),
        tooltip=["period:O", "network_family:N", alt.Tooltip("modularity:Q", format=".4f"), alt.Tooltip("degree_centralization:Q", format=".4f"), "top_node:N", "top_strength:Q"],
    ).properties(width="container", height=320, title="Yearly modularity and centralization")
    top_rank_changes = (
        window_rank_change
        .filter(pl.col("pre_strength") + pl.col("post_strength") > 0)
        .with_columns(pl.col("rank_change").abs().alias("absolute_rank_change"))
        .sort(["network_family", "absolute_rank_change", "post_strength"], descending=[False, True, True])
        .group_by("network_family", maintain_order=True)
        .head(20)
    )
    yearly_top_rank_changes = (
        yearly_rank_change
        .with_columns(pl.col("rank_change").abs().alias("absolute_rank_change"))
        .sort(["to_year", "network_family", "absolute_rank_change", "to_strength"], descending=[False, False, True, True])
        .group_by("to_year", "network_family", maintain_order=True)
        .head(5)
    )
    strongest_window_edge_changes = (
        window_edge_changes
        .with_columns(pl.col("support_change").abs().alias("absolute_support_change"))
        .sort(["network_family", "absolute_support_change", "post_support"], descending=[False, True, True])
        .group_by("network_family", maintain_order=True)
        .head(15)
    )
    temporal_most_persistent = window_turnover.sort("edge_jaccard", descending=True).row(0, named=True)
    temporal_largest_change = strongest_window_edge_changes.sort(
        "absolute_support_change", descending=True
    ).row(0, named=True)
    temporal_section = mo.vstack(
        [
            mo.md(
                f"Eligible complete years are **{', '.join(str(year) for year in eligible_years)}**. "
                f"The source has **{int(yearly_counts['reviewed_claims'].sum()):,}** dated reviewed claims "
                f"through 2024 before the minimum-year filter. In matched windows, the "
                f"**{temporal_most_persistent['network_family']}** network has the higher edge-set "
                f"Jaccard (**{temporal_most_persistent['edge_jaccard']:.3f}**), while the largest "
                f"absolute relation-support movement is **{temporal_largest_change['source']} ↔ "
                f"{temporal_largest_change['target']}** "
                f"({temporal_largest_change['support_change']:+,} records)."
            ),
            edge_chart,
            mo.ui.tabs({
                "Modularity and centralization": modularity_chart,
                "Edge turnover": mo.ui.table(
                    turnover_all,
                    selection=None,
                    pagination=True,
                    page_size=25,
                    show_column_summaries=False,
                ),
                "Matched-window metrics": mo.hstack(
                    [
                        mo.ui.table(window_metrics, selection=None, pagination=False, show_column_summaries=False),
                        mo.ui.table(window_turnover, selection=None, pagination=False, show_column_summaries=False),
                    ], widths="equal", align="start", wrap=True,
                ),
                "Matched-window rank changes": mo.ui.table(
                    top_rank_changes,
                    selection=None,
                    pagination=True,
                    page_size=20,
                    show_search=True,
                    show_column_summaries=False,
                ),
                "Yearly rank changes": mo.ui.table(
                    yearly_top_rank_changes,
                    selection=None,
                    pagination=True,
                    page_size=25,
                    show_search=True,
                    show_column_summaries=False,
                ),
                "Relation-support changes": mo.ui.table(
                    strongest_window_edge_changes,
                    selection=None,
                    pagination=True,
                    page_size=25,
                    show_search=True,
                    show_column_summaries=False,
                ),
                "Publisher coverage": mo.hstack(
                    [
                        mo.ui.table(publisher_temporal, selection=None, pagination=True, page_size=25, show_column_summaries=False),
                        mo.ui.table(publisher_window_change, selection=None, pagination=False, show_column_summaries=False),
                    ], widths="equal", align="start", wrap=True,
                ),
                "Excluded years": mo.ui.table(
                    excluded_years,
                    selection=None,
                    pagination=True,
                    page_size=20,
                    show_column_summaries=False,
                ),
            }),
            mo.callout(
                mo.md(
                    "Temporal coexistence is not a causal intervention. An edge appears when an "
                    "association crosses the fixed support threshold in this extract, not when a "
                    "new real-world relationship begins."
                ),
                kind="warn",
                title="How to interpret network change",
            ),
        ]
    )
    temporal_section
    return


@app.cell(hide_code=True)
def explain_outputs_and_limits(mo):
    mo.md("""
    ## 6. Methods, sensitivity, audit, and data

    Detailed provenance, threshold sensitivity, actor exclusions, and model audits
    remain available below through progressive disclosure. They support reproducible
    interpretation without competing with the primary findings above.

    Every download is generated eagerly for the static page. Searchable tables
    retain source claims and exclusion decisions for audit; there is no separate
    static download manifest.
    """)
    return


@app.cell
def render_findings_downloads_and_explorers(
    MODEL_ID,
    MODEL_REVISION,
    actor_audit_section,
    actor_edges,
    actor_node_metrics,
    actor_occurrence_frame,
    actor_publisher_edges,
    exclusion_summary,
    keyword_edges,
    keyword_node_metrics,
    mo,
    pl,
    provenance_section,
    publisher_keyword_edges,
    sensitivity_section,
    temporal_metrics_all,
    turnover_all,
    window_edge_changes,
    window_turnover,
):
    publisher_keyword_publisher_nodes = (
        publisher_keyword_edges
        .group_by("source", "publisher_site")
        .agg(
            pl.col("support").sum().alias("strength"),
            pl.col("keyword").n_unique().alias("degree"),
        )
        .rename({"source": "node_id", "publisher_site": "label"})
        .with_columns(
            pl.lit("publisher").alias("node_type"),
            pl.col("strength").alias("support"),
            pl.lit("publisher-keyword").alias("network_family"),
        )
    )
    publisher_keyword_keyword_nodes = (
        publisher_keyword_edges
        .group_by("target", "keyword")
        .agg(
            pl.col("support").sum().alias("strength"),
            pl.col("publisher_site").n_unique().alias("degree"),
        )
        .rename({"target": "node_id", "keyword": "label"})
        .with_columns(
            pl.lit("keyword").alias("node_type"),
            pl.col("strength").alias("support"),
            pl.lit("publisher-keyword").alias("network_family"),
        )
    )
    actor_publisher_actor_nodes = (
        actor_publisher_edges
        .group_by("source", "actor_id")
        .agg(
            pl.col("support").sum().alias("strength"),
            pl.col("publisher_site").n_unique().alias("degree"),
        )
        .rename({"source": "node_id"})
        .join(actor_node_metrics.select("node_id", "label"), on="node_id", how="left")
        .with_columns(
            pl.lit("actor").alias("node_type"),
            pl.col("strength").alias("support"),
            pl.lit("actor-publisher").alias("network_family"),
        )
    )
    actor_publisher_publisher_nodes = (
        actor_publisher_edges
        .group_by("target", "publisher_site")
        .agg(
            pl.col("support").sum().alias("strength"),
            pl.col("actor_id").n_unique().alias("degree"),
        )
        .rename({"target": "node_id", "publisher_site": "label"})
        .with_columns(
            pl.lit("publisher").alias("node_type"),
            pl.col("strength").alias("support"),
            pl.lit("actor-publisher").alias("network_family"),
        )
    )
    combined_nodes = pl.concat(
        [
            keyword_node_metrics.with_columns(pl.lit("keyword co-match").alias("network_family")),
            actor_node_metrics.with_columns(pl.lit("actor co-mention").alias("network_family")),
            publisher_keyword_publisher_nodes,
            publisher_keyword_keyword_nodes,
            actor_publisher_actor_nodes,
            actor_publisher_publisher_nodes,
        ],
        how="diagonal_relaxed",
    ).with_columns(
        pl.col("provenance_flags").list.join("|").alias("provenance_flags"),
        pl.col("model_entity_types").list.join("|").alias("model_entity_types"),
    )
    assert combined_nodes.select("network_family", "node_id").is_duplicated().sum() == 0
    combined_edges = pl.concat(
        [
            keyword_edges.with_columns(pl.lit("keyword co-match").alias("network_family")),
            actor_edges.with_columns(pl.lit("actor co-mention").alias("network_family")),
            publisher_keyword_edges.with_columns(pl.lit("publisher-keyword").alias("network_family")),
            actor_publisher_edges.with_columns(pl.lit("actor-publisher").alias("network_family")),
        ],
        how="diagonal_relaxed",
    )
    actor_occurrence_export = actor_occurrence_frame.with_columns(
        pl.col("model_entity_types").list.join("|").alias("model_entity_types"),
        pl.col("provenance_flags").list.join("|").alias("provenance_flags"),
        pl.lit(MODEL_ID).alias("model_id"),
        pl.lit(MODEL_REVISION).alias("model_revision"),
    )
    temporal_export = temporal_metrics_all.join(
        turnover_all,
        left_on=["design", "network_family", "period"],
        right_on=["design", "network_family", "to_period"],
        how="left",
    )

    downloads = mo.hstack(
        [
            mo.download(combined_nodes.write_csv().encode("utf-8"), filename="network_nodes.csv", mimetype="text/csv", label=f"Node metrics CSV ({combined_nodes.height:,} rows)"),
            mo.download(combined_edges.write_csv().encode("utf-8"), filename="network_edges.csv", mimetype="text/csv", label=f"Edge tables CSV ({combined_edges.height:,} rows)"),
            mo.download(temporal_export.write_csv().encode("utf-8"), filename="network_temporal_metrics.csv", mimetype="text/csv", label=f"Temporal metrics CSV ({temporal_export.height:,} rows)"),
            mo.download(window_edge_changes.write_csv().encode("utf-8"), filename="network_window_edge_changes.csv", mimetype="text/csv", label=f"Window edge changes CSV ({window_edge_changes.height:,} rows)"),
            mo.download(actor_occurrence_export.write_csv().encode("utf-8"), filename="network_actor_occurrences.csv", mimetype="text/csv", label=f"Actor occurrences CSV ({actor_occurrence_export.height:,} rows)"),
            mo.download(exclusion_summary.write_csv().encode("utf-8"), filename="network_claimant_exclusions.csv", mimetype="text/csv", label=f"Claimant exclusions CSV ({exclusion_summary.height:,} rows)"),
        ],
        wrap=True,
    )

    edge_explorer_frame = (
        combined_edges
        .sort(["support", "network_family", "source", "target"], descending=[True, False, False, False])
        .head(200)
    )
    actor_explorer_frame = (
        actor_occurrence_export
        .sort(["max_ner_score", "normalized_actor", "claim_text"], nulls_last=True)
        .head(200)
    )
    edge_explorer = mo.ui.table(
        edge_explorer_frame,
        selection=None,
        pagination=True,
        page_size=200,
        show_search=True,
        show_download=True,
        show_column_summaries=True,
        wrapped_columns=["representative_record"],
        label="Primary edge explorer (top 200 by support)",
    )
    actor_explorer = mo.ui.table(
        actor_explorer_frame,
        selection=None,
        pagination=True,
        page_size=200,
        show_search=True,
        show_download=True,
        show_column_summaries=True,
        wrapped_columns=["claim_text", "provenance_flags", "model_entity_types"],
        label="Actor occurrence audit (200-row deterministic sample)",
    )
    limitations = mo.callout(
        mo.md(
            "**Interpretation limits.** Google indexing and fixed queries define entry into the corpus; "
            "publisher mix is incomplete; claim dates are missing for some records; NER spans are model "
            "predictions; exact-only aliases fragment identities; structured claimants are sparse and noisy; "
            "support thresholds alter topology; retrospective matching can connect a claim to vocabulary that "
            "was configured later. Centrality is not importance, a community is not coordination, and a "
            "publisher edge is not affiliation."
        ),
        kind="warn",
        title="What these networks cannot establish",
    )
    output_section = mo.vstack(
        [
            limitations,
            mo.accordion({
                "NER provenance and analytical grains": provenance_section,
                "Threshold sensitivity": sensitivity_section,
                "Actor identification and exclusion audit": actor_audit_section,
            }, multiple=False),
            mo.md("### Eager research downloads"),
            downloads,
            mo.callout(
                mo.md(
                    "Marimo static tables support at most 200 embedded rows. The explorers show "
                    "deterministic research samples; the eager CSV downloads above contain every "
                    "node, edge, temporal result, actor occurrence, and exclusion decision."
                ),
                kind="info",
                title="Complete artifacts are in the CSV downloads",
            ),
            mo.md("### Primary edge explorer (top 200 by support)"),
            edge_explorer,
            mo.md("### Actor occurrence audit (200-row deterministic sample)"),
            actor_explorer,
        ]
    )
    output_section
    return


if __name__ == "__main__":
    app.run()
