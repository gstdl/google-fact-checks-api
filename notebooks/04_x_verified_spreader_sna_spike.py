import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def introduce_spike(mo):
    mo.md("""
    # Spike: ranking verified accounts in X interaction networks for two fact-checked claims

    This spike checks whether X mentions can become a directed interaction network
    showing which verified accounts engage most with two claims that Indonesian
    fact-checkers have already debunked, *Ijazah Jokowi* and *Eko Patrio promosi judol*.

    The input is an `x_mentions` table with a fixed set of columns. For now it holds
    synthetic data, so the numbers only show that the pipeline runs and say nothing
    about real accounts. A Brandwatch adapter will need to produce the same columns.

    Edges come from retweets, quotes, replies and @mentions. They are not follower ties,
    and the spike does not label whether a post spreads or debunks the claim.
    """)
    return


@app.cell
def import_dependencies():
    import altair as alt
    import marimo as mo
    import networkx as nx
    import polars as pl

    return alt, mo, nx, pl


@app.cell
def load_mentions(pl):
    MENTIONS_PATH = "tmp/x_mentions_synthetic.parquet"
    mentions = pl.read_parquet(MENTIONS_PATH)
    is_synthetic = mentions["is_synthetic"].all()
    return is_synthetic, mentions


@app.cell
def build_edges(mentions, pl):
    amplify_edges = (
        mentions
        .filter(pl.col("parent_author_handle").is_not_null())
        .select(
            pl.col("author_handle").alias("source"),
            pl.col("parent_author_handle").alias("target"),
            "mention_type",
            "topic",
        )
    )
    mention_edges = (
        mentions
        .select(pl.col("author_handle").alias("source"), pl.col("mentioned_handles").alias("target"), "topic")
        .explode("target", empty_as_null=True)
        .drop_nulls("target")
        .with_columns(pl.lit("mention").alias("mention_type"))
    )
    edges = (
        pl.concat([amplify_edges, mention_edges.select(amplify_edges.columns)])
        .filter(pl.col("source") != pl.col("target"))
        .group_by("source", "target", "topic")
        .agg(pl.len().alias("weight"))
    )
    accounts = (
        mentions
        .group_by("author_handle")
        .agg(
            pl.when(pl.col("author_verified").any()).then(pl.lit("verified")).otherwise(pl.lit("not verified")).alias("verified"),
            pl.col("author_followers").max().alias("followers"),
        )
        .rename({"author_handle": "handle"})
    )
    return accounts, edges


@app.cell
def build_graph(accounts, edges, nx, pl):
    topic_graphs = {}
    for _topic in edges["topic"].unique().sort().to_list():
        _graph = nx.DiGraph()
        _graph.add_weighted_edges_from(
            edges.filter(pl.col("topic") == _topic).select("source", "target", "weight").iter_rows()
        )
        # Accounts only ever seen as a target (never authored a mention) have unknown verification.
        _known = {r["handle"]: r for r in accounts.iter_rows(named=True)}
        for _n in _graph:
            _graph.nodes[_n]["verified"] = _known[_n]["verified"] if _n in _known else "unknown"
            _graph.nodes[_n]["followers"] = _known[_n]["followers"] if _n in _known else None
        topic_graphs[_topic] = _graph
    return (topic_graphs,)


@app.cell
def compute_node_metrics(nx, pl, topic_graphs):
    _frames = []
    for _topic, _graph in topic_graphs.items():
        _betweenness = nx.betweenness_centrality(_graph, weight=None)
        _frames.append(pl.DataFrame({
            "topic": _topic,
            "handle": list(_graph),
            "verified": [_graph.nodes[n]["verified"] for n in _graph],
            "followers": [_graph.nodes[n]["followers"] for n in _graph],
            "amplified_in": [_graph.in_degree(n, weight="weight") for n in _graph],
            "amplifies_out": [_graph.out_degree(n, weight="weight") for n in _graph],
            "betweenness": [_betweenness[n] for n in _graph],
        }))
    node_metrics = pl.concat(_frames)
    return (node_metrics,)


@app.cell(hide_code=True)
def render_verified_ranking(is_synthetic, mo, node_metrics, pl):
    top_verified = (
        node_metrics
        .filter(pl.col("verified") == "verified")
        .sort(["topic", "amplified_in"], descending=[False, True])
        .group_by("topic", maintain_order=True)
        .head(5)
    )
    verified_share = (
        node_metrics
        .group_by("topic")
        .agg(
            (pl.col("amplified_in").filter(pl.col("verified") == "verified").sum() / pl.col("amplified_in").sum()).alias("verified_share_of_amplification")
        )
    )
    mo.vstack([
        mo.md(f"""
        ## 1. Share of interactions received by verified accounts

        {"These figures come from synthetic data and only show that the pipeline works." if is_synthetic else ""}
        """),
        verified_share,
        mo.md("Top five verified accounts by interactions received, per topic."),
        top_verified,
    ])
    return


@app.cell(hide_code=True)
def render_network(alt, mo, node_metrics, nx, pl, topic_graphs):
    _charts = []
    for _topic, _graph in topic_graphs.items():
        _positions = nx.spring_layout(_graph, seed=7, weight=None)
        _nodes = (
            node_metrics.filter(pl.col("topic") == _topic)
            .with_columns(
                pl.col("handle").map_elements(lambda h: _positions[h][0], return_dtype=pl.Float64).alias("x"),
                pl.col("handle").map_elements(lambda h: _positions[h][1], return_dtype=pl.Float64).alias("y"),
            )
        )
        _segments = pl.DataFrame([
            {"x": _positions[_u][0], "y": _positions[_u][1], "x2": _positions[_v][0], "y2": _positions[_v][1]}
            for _u, _v in _graph.edges
        ])
        _edge_layer = alt.Chart(_segments).mark_rule(opacity=0.12, strokeWidth=0.5).encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), x2="x2:Q", y2="y2:Q"
        )
        _node_layer = alt.Chart(_nodes).mark_circle(opacity=0.9).encode(
            x="x:Q",
            y="y:Q",
            size=alt.Size("amplified_in:Q", legend=alt.Legend(title="Interactions received")),
            color=alt.Color(
                "verified:N",
                scale=alt.Scale(domain=["verified", "not verified", "unknown"], range=["#d62728", "#9aa5b1", "#d9dee4"]),
                legend=alt.Legend(title="Verified"),
            ),
            tooltip=["handle", "verified", "followers", "amplified_in", "amplifies_out"],
        )
        _charts.append(
            (_edge_layer + _node_layer)
            .properties(title=f"Interaction network for {_topic}, verified accounts in red", width=480, height=420)
        )
    mo.hstack([mo.ui.altair_chart(_c) for _c in _charts])
    return


@app.cell(hide_code=True)
def render_topic_overlap(mo, node_metrics, pl):
    overlap = (
        node_metrics
        .filter(pl.col("verified") == "verified")
        .group_by("handle")
        .agg(pl.col("topic").n_unique().alias("topics"), pl.col("amplified_in").sum().alias("amplified_in"))
        .filter(pl.col("topics") > 1)
        .sort("amplified_in", descending=True)
    )
    mo.vstack([
        mo.md("## 2. Verified accounts that appear in both topics"),
        overlap,
    ])
    return


@app.cell(hide_code=True)
def render_limits(mo):
    mo.md("""
    ## 3. Limits of this spike

    - The network is built from interactions. Brandwatch exposes mentions, so follower ties are unavailable.
    - No post is labelled as spreading or debunking. A quote or reply may challenge the claim, so a high count shows heavy interaction and says nothing about endorsement.
    - The verified flag does not vet identity, and paid checks count. Follower tiers would be a useful second cut.
    - The sample depends on the Brandwatch query and date window, so it says nothing about how common the claims are.
    - The input is synthetic. Swap `tmp/x_mentions_synthetic.parquet` for the adapter output before reading any result.
    """)
    return


if __name__ == "__main__":
    app.run()
