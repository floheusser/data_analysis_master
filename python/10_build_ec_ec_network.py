# Original notebook: 10_build_ec_ec_network.ipynb
# Converted to Python script on: 2026-05-24
# Outputs and markdown cells have been removed.
# Code logic has been preserved as closely as possible.

# --- Cell 1: 1. Imports and Configuration ---
# 1. Imports and Configuration
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/processed")
INPUT_PATH = DATA_DIR / "ec_ec_case_matches.csv"

OUT_DIR           = DATA_DIR / "network" / "ec_ec"
NODES_PATH        = OUT_DIR / "ec_ec_nodes.csv"
EDGES_PATH        = OUT_DIR / "ec_ec_edges.csv"
GRAPHML_PATH      = OUT_DIR / "ec_ec_network.graphml"
GEXF_PATH         = OUT_DIR / "ec_ec_network.gexf"
NODE_METRICS_PATH = OUT_DIR / "ec_ec_node_metrics.csv"

FIGURES_DIR = Path("outputs/figures/ec_ec")

# Node colour constants
COLOR_EC = "#4C72B0"

# Match-strength ordering (higher index = stronger)
STRENGTH_ORDER = {"weak": 0, "medium": 1, "strong": 2}

print("Configuration loaded.")

# --- Cell 2: 2. Load CSV ---
# 2. Load CSV
df = pd.read_csv(INPUT_PATH, dtype=str).fillna("")

print(f"Rows loaded : {len(df):,}")
print(f"Columns     : {list(df.columns)}")
df.head(3)

# --- Cell 3: 3. Filter: Only Valid Matches ---
# 3. Filter: Only Valid Matches
# Keep only rows with processing_status == "matched" (or "ok_pdf" / non-empty meaningful status)
# The file uses "ok_pdf" as the matched status based on the data
matched = df[df["processing_status"].str.strip().ne("") & df["processing_status"].str.strip().ne("no_match")].copy()

# Drop rows with missing source or target case numbers
matched = matched[
    matched["source_ec_case_number"].str.strip().ne("") &
    matched["target_ec_case_number"].str.strip().ne("")
].copy()

# Remove self-citations
matched = matched[
    matched["source_ec_case_number"].str.strip() != matched["target_ec_case_number"].str.strip()
].copy()

print(f"Rows after filtering: {len(matched):,}")
print(f"Unique processing_status values: {matched['processing_status'].unique()}")

# --- Cell 4: 4. Deduplicate to One Edge per Source-Target Pair ---
# 4. Deduplicate to One Edge per Source-Target Pair
# Each raw row represents one pattern match. Multiple rows may exist for the same
# source-target pair (different passages, different patterns). We aggregate them into
# one unique directed edge per `(source_ec_case_number, target_ec_case_number)` pair,
# keeping useful aggregated attributes.
def best_strength(strengths: pd.Series) -> str:
    """Return the strongest match_strength value in a group."""
    ranked = strengths.map(lambda s: STRENGTH_ORDER.get(s, -1))
    best_idx = ranked.idxmax()
    return strengths.loc[best_idx]


edges_raw = (
    matched
    .groupby(["source_ec_case_number", "target_ec_case_number"], sort=False)
    .agg(
        match_count           = ("match_strength", "count"),
        best_match_strength   = ("match_strength", best_strength),
        all_match_strengths   = ("match_strength", lambda s: "|".join(s.unique())),
        example_match_context = ("match_context", "first"),
    )
    .reset_index()
)

print(f"Unique EC-EC pairs (edges): {len(edges_raw):,}")
edges_raw.head(3)

# --- Cell 5: 5. Build Node Table ---
# 5. Build Node Table
# All EC cases that appear as source or target are collected into a single node table.
# Source and target metadata are merged carefully so each case appears only once.
# ── Source node metadata ───────────────────────────────────────────────────────
src_meta = (
    matched[["source_ec_case_number", "source_case_title", "source_date",
             "source_type", "source_document_type"]]
    .drop_duplicates(subset="source_ec_case_number")
    .rename(columns={
        "source_ec_case_number": "case_number",
        "source_case_title"    : "label",
        "source_date"          : "date",
        "source_type"          : "source_type",
        "source_document_type" : "document_type",
    })
    .copy()
)
src_meta["celex_no"] = ""

# ── Target node metadata ───────────────────────────────────────────────────────
tgt_meta = (
    matched[["target_ec_case_number", "target_case_title", "target_celex_no"]]
    .drop_duplicates(subset="target_ec_case_number")
    .rename(columns={
        "target_ec_case_number": "case_number",
        "target_case_title"    : "label",
        "target_celex_no"      : "celex_no",
    })
    .copy()
)
tgt_meta["date"]          = ""
tgt_meta["source_type"]   = ""
tgt_meta["document_type"] = ""

# ── Merge: source takes priority for shared cases ─────────────────────────────
all_meta = pd.concat([src_meta, tgt_meta], ignore_index=True)
all_meta = all_meta.drop_duplicates(subset="case_number", keep="first")

ec_nodes = pd.DataFrame({
    "node_id"      : "ec:" + all_meta["case_number"],
    "node_type"    : "ec",
    "label"        : all_meta["label"].values,
    "case_number"  : all_meta["case_number"].values,
    "celex_no"     : all_meta["celex_no"].values,
    "date"         : all_meta["date"].values,
    "source_type"  : all_meta["source_type"].values,
    "document_type": all_meta["document_type"].values,
})

print(f"EC nodes: {len(ec_nodes):,}")
ec_nodes.head(5)

# --- Cell 6: 6. Build Edge Table ---
# 6. Build Edge Table
edges = edges_raw.copy()
edges.insert(0, "source", "ec:" + edges["source_ec_case_number"])
edges.insert(1, "target", "ec:" + edges["target_ec_case_number"])

print(f"Edges: {len(edges):,}")
edges.head(3)

# --- Cell 7: 7. Build NetworkX Directed Graph ---
# 7. Build NetworkX Directed Graph
G = nx.DiGraph()

# Add EC nodes
for _, row in ec_nodes.iterrows():
    G.add_node(
        row["node_id"],
        node_type    =row["node_type"],
        label        =row["label"],
        case_number  =row["case_number"],
        celex_no     =row["celex_no"],
        date         =row["date"],
        source_type  =row["source_type"],
        document_type=row["document_type"],
    )

# Add edges
for _, row in edges.iterrows():
    G.add_edge(
        row["source"],
        row["target"],
        match_count          =int(row["match_count"]),
        best_match_strength  =row["best_match_strength"],
        all_match_strengths  =row["all_match_strengths"],
        example_match_context=str(row["example_match_context"])[:300],
    )

print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# --- Cell 8: 8. Basic Statistics ---
# 8. Basic Statistics
print(f"EC nodes : {len(ec_nodes):,}")
print(f"Edges    : {len(edges):,}")
print(f"Is directed: {G.is_directed()}")

# --- Cell 9: 9.1 Reciprocity ---
# 9.1 Reciprocity
# **What it measures:**
# Reciprocity is the fraction of edges that are mutual — i.e., if case A cites case B,
# does case B also cite case A? A value of 0 means no mutual citations; 1 means every
# citation is reciprocated.
# **Why it matters for this EC → EC citation network:**
# In a legal citation network, high reciprocity would be unusual (decisions typically cite
# earlier cases, not later ones). Low reciprocity confirms the expected temporal, hierarchical
# structure of the citation graph. Any non-zero reciprocity may indicate cases that were
# decided close in time and cross-reference each other.
reciprocity = nx.reciprocity(G)
print(f"Network reciprocity: {reciprocity:.4f}")
print(f"  → {reciprocity * 100:.2f}% of edges have a reciprocal counterpart")

# --- Cell 10: 9.2 Weakly Connected Components ---
# 9.2 Weakly Connected Components
# **What it measures:**
# A weakly connected component (WCC) is a maximal set of nodes that are connected when
# edge directions are ignored. Two nodes belong to the same WCC if there is an undirected
# path between them.
# **Why it matters for this EC → EC citation network:**
# WCCs reveal whether the EC citation graph forms one large connected cluster or splits into
# several isolated sub-networks. A dominant large WCC means most EC cases are structurally
# linked through the citation graph; small isolated components may represent niche or
# peripheral citation clusters with no connection to the main body of EC antitrust decisions.
wccs = list(nx.weakly_connected_components(G))
wccs_sorted = sorted(wccs, key=len, reverse=True)

print(f"Number of weakly connected components : {len(wccs_sorted):,}")
print(f"Largest WCC size                       : {len(wccs_sorted[0]):,} nodes")
print(f"Nodes in largest WCC (% of total)      : {len(wccs_sorted[0]) / G.number_of_nodes() * 100:.1f}%")

wcc_summary = pd.DataFrame({
    "component_id": range(1, len(wccs_sorted) + 1),
    "size"        : [len(c) for c in wccs_sorted],
})

print("\nWeakly connected component sizes (top 20):")
display(wcc_summary.head(20))

# --- Cell 11: 9.3 Strongly Connected Components ---
# 9.3 Strongly Connected Components
# **What it measures:**
# A strongly connected component (SCC) is a maximal set of nodes where every node can
# reach every other node *following edge directions*. In a directed citation graph, an SCC
# larger than 1 means a group of cases that mutually cite each other (directly or indirectly).
# **Why it matters for this EC → EC citation network:**
# Because citations typically flow forward in time (older cases are cited by newer ones),
# most SCCs are expected to be trivial (size 1). Any non-trivial SCC (size > 1) indicates
# a cycle of mutual citations — cases that reference each other in a loop — which may
# reflect cross-referencing decisions issued close in time or corrections/amendments.
sccs = list(nx.strongly_connected_components(G))
sccs_sorted = sorted(sccs, key=len, reverse=True)

print(f"Number of strongly connected components : {len(sccs_sorted):,}")
print(f"Largest SCC size                         : {len(sccs_sorted[0]):,} nodes")
print(f"Nodes in largest SCC (% of total)        : {len(sccs_sorted[0]) / G.number_of_nodes() * 100:.1f}%")
print(f"Non-trivial SCCs (size > 1)              : {sum(1 for c in sccs_sorted if len(c) > 1):,}")

scc_summary = pd.DataFrame({
    "component_id": range(1, len(sccs_sorted) + 1),
    "size"        : [len(c) for c in sccs_sorted],
})

print("\nStrongly connected component sizes (top 20):")
display(scc_summary.head(20))

# --- Cell 12: --- ---
# ---
# 10. Visualisation (Top 50 Nodes)
# A quick visual impression of the network using the 25 most-cited and 25 most-citing EC cases.
# Nodes are coloured by in-degree (cited) vs out-degree (citing) role for orientation.
from matplotlib.patches import Patch

# Pre-compute degree series for subgraph selection
in_deg_series = pd.Series(
    dict(G.in_degree()),
    name="in_degree",
).sort_values(ascending=False)

out_deg_series = pd.Series(
    dict(G.out_degree()),
    name="out_degree",
).sort_values(ascending=False)

# Select top 25 most-cited and top 25 most-citing nodes
top_cited_ids  = in_deg_series.head(25).index.tolist()
top_citing_ids = out_deg_series.head(25).index.tolist()

sub_nodes = set(top_cited_ids) | set(top_citing_ids)
H = G.subgraph(sub_nodes).copy()

print(f"Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

# Colour: nodes that appear in top-cited get orange; top-citing get blue; both get green
color_map = []
for n in H.nodes():
    in_top  = n in top_cited_ids
    out_top = n in top_citing_ids
    if in_top and out_top:
        color_map.append("#2ca02c")   # green = both
    elif in_top:
        color_map.append("#DD8452")   # orange = highly cited
    else:
        color_map.append("#4C72B0")   # blue = highly citing

# Short labels (case number)
labels = {n: H.nodes[n].get("case_number", n) for n in H.nodes()}

fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(H, seed=42, k=1.5)

nx.draw_networkx_nodes(H, pos, node_color=color_map, node_size=300, alpha=0.9, ax=ax)
nx.draw_networkx_edges(H, pos, edge_color="#aaaaaa", arrows=True,
                       arrowsize=12, width=0.8, alpha=0.7, ax=ax)
nx.draw_networkx_labels(H, pos, labels=labels, font_size=6, ax=ax)

legend_elements = [
    Patch(facecolor="#DD8452", label="Highly cited (in-degree)"),
    Patch(facecolor="#4C72B0", label="Highly citing (out-degree)"),
    Patch(facecolor="#2ca02c", label="Both"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
ax.set_title("EC → EC Citation Network (Top 50 nodes)", fontsize=13)
ax.axis("off")
plt.tight_layout()
plt.show()

# --- Cell 13: 11.1 In-Degree and Out-Degree ---
# 11.1 In-Degree and Out-Degree
# **What it measures:**
# - **In-degree** = number of incoming edges → how often an EC case is *cited by other EC cases*.
# - **Out-degree** = number of outgoing edges → how many other EC cases a document *cites*.
# **Why it matters for this citation network:**
# In-degree is the most direct measure of an EC case's influence within the EC corpus:
# the more other EC decisions reference it, the more foundational it is.
# Out-degree reveals which EC documents draw most heavily on prior EC precedents.
# **Most meaningful for:** both node types are EC cases, so both metrics are equally relevant.
# Compute in- and out-degree for every node
in_degree_all  = dict(G.in_degree())
out_degree_all = dict(G.out_degree())

# ── Top cited EC cases (by in-degree) ─────────────────────────────────────────
ec_in = (
    ec_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(in_degree=lambda df: df.index.map(in_degree_all))
    .sort_values("in_degree", ascending=False)
)

print("Top 20 most cited EC cases (in-degree):")
display(ec_in.head(20))

# --- Cell 14 ---
# ── Top citing EC cases (by out-degree) ───────────────────────────────────────
ec_out = (
    ec_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(out_degree=lambda df: df.index.map(out_degree_all))
    .sort_values("out_degree", ascending=False)
)

print("Top 20 most citing EC cases (out-degree):")
display(ec_out.head(20))

# --- Cell 15: 11.2 PageRank ---
# 11.2 PageRank
# **What it measures:**
# PageRank assigns a prestige score to each node based on the number and quality of incoming links.
# A node receives a higher score if it is cited by nodes that are themselves highly cited.
# **Why it matters for this citation network:**
# Unlike raw in-degree, PageRank accounts for the *importance* of the citing document.
# An EC case cited by many influential EC decisions will rank higher than one cited
# by less prominent documents. This helps identify the most foundational EC precedents.
# **Most meaningful for:** all EC nodes — identifies the most *prestigious* EC cases in the network.
pagerank = nx.pagerank(G, alpha=0.85)

# Top EC cases by PageRank
ec_pr = (
    ec_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(
        in_degree=lambda df: df.index.map(in_degree_all),
        pagerank =lambda df: df.index.map(pagerank),
    )
    .sort_values("pagerank", ascending=False)
)

print("Top 20 EC cases by PageRank:")
display(ec_pr.head(20))

# --- Cell 16: 11.3 Betweenness Centrality ---
# 11.3 Betweenness Centrality
# **What it measures:**
# Betweenness centrality counts how often a node lies on the shortest path between two other nodes.
# A node with high betweenness acts as a *bridge* or *connector* within the network.
# **Why it matters for this citation network:**
# In the EC → EC citation network, a case with high betweenness is not just frequently cited —
# it structurally connects otherwise separate clusters of EC decisions.
# Such cases may represent pivotal precedents that link different areas of EC antitrust practice.
# **Important note:** Betweenness is an additional structural metric, not the primary one.
# In-degree and PageRank remain the main indicators of importance.
# Betweenness highlights *bridge nodes* that may be overlooked by degree-based metrics alone.
betweenness = nx.betweenness_centrality(G, normalized=True)

# Top EC cases by betweenness
ec_bw = (
    ec_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(
        in_degree   =lambda df: df.index.map(in_degree_all),
        betweenness =lambda df: df.index.map(betweenness),
    )
    .sort_values("betweenness", ascending=False)
)

print("Top 20 EC cases by betweenness centrality:")
display(ec_bw.head(20))

# --- Cell 17: --- ---
# ---
# 12. Node-Level Metric Summary Table
# All four core metrics are combined into a single node-level table for easy inspection and export.
# Each row represents one EC node with its `in_degree`, `out_degree`, `pagerank`, and `betweenness`.
node_metrics = ec_nodes[["node_id", "label", "case_number"]].copy()

node_metrics["in_degree"]   = node_metrics["node_id"].map(in_degree_all)
node_metrics["out_degree"]  = node_metrics["node_id"].map(out_degree_all)
node_metrics["pagerank"]    = node_metrics["node_id"].map(pagerank)
node_metrics["betweenness"] = node_metrics["node_id"].map(betweenness)

print(f"Node metrics table: {len(node_metrics):,} rows")
display(node_metrics.sort_values("in_degree", ascending=False).head(20))

# --- Cell 18: --- ---
# ---
# 12A. Metric Summary Statistics and Distributions
# This section provides summary statistics and distribution plots for all node-level metrics
# computed in this notebook. Statistics are based on **all EC nodes** in the network,
# not just the top-10 rankings.
# **Metrics covered:**
# - `in_degree` — how often each EC case is cited by other EC decisions
# - `out_degree` — how many other EC cases each decision cites
# - `pagerank` — prestige-weighted citation score
# - `betweenness` — structural bridge score
from scipy import stats as _scipy_stats

SUMMARY_STATS_PATH = OUT_DIR / "ec_ec_metric_summary_stats.csv"

def _summary_stats(series: pd.Series, name: str) -> dict:
    """Compute summary statistics for a numeric series."""
    s = series.dropna()
    return {
        "metric"  : name,
        "count"   : int(s.count()),
        "mean"    : round(float(s.mean()), 6),
        "std"     : round(float(s.std()), 6),
        "min"     : round(float(s.min()), 6),
        "25%"     : round(float(s.quantile(0.25)), 6),
        "median"  : round(float(s.median()), 6),
        "75%"     : round(float(s.quantile(0.75)), 6),
        "max"     : round(float(s.max()), 6),
        "skewness": round(float(_scipy_stats.skew(s)), 4),
        "kurtosis": round(float(_scipy_stats.kurtosis(s)), 4),
    }

in_degree_series   = pd.Series({nid: in_degree_all[nid]  for nid in ec_nodes["node_id"] if nid in in_degree_all},  name="in_degree")
out_degree_series  = pd.Series({nid: out_degree_all[nid] for nid in ec_nodes["node_id"] if nid in out_degree_all}, name="out_degree")
pagerank_series    = pd.Series({nid: pagerank[nid]        for nid in ec_nodes["node_id"] if nid in pagerank},       name="pagerank")
betweenness_series = pd.Series({nid: betweenness[nid]     for nid in ec_nodes["node_id"] if nid in betweenness},    name="betweenness")

summary_rows = [
    _summary_stats(in_degree_series,   "in_degree"),
    _summary_stats(out_degree_series,  "out_degree"),
    _summary_stats(pagerank_series,    "pagerank"),
    _summary_stats(betweenness_series, "betweenness"),
]

summary_stats_df = pd.DataFrame(summary_rows)
print("Metric Summary Statistics — EC → EC Network")
display(summary_stats_df)

OUT_DIR.mkdir(parents=True, exist_ok=True)
summary_stats_df.to_csv(SUMMARY_STATS_PATH, index=False, encoding="utf-8")
print(f"\nSaved summary stats to: {SUMMARY_STATS_PATH}")

# --- Cell 19: 12A-1. In-Degree and Out-Degree Distributions ---
# 12A-1. In-Degree and Out-Degree Distributions
# Distribution of in-degree and out-degree across all EC nodes. Citation distributions are
# typically highly right-skewed; a logarithmic y-axis is used for readability.
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

for ax, series, xlabel, title in [
    (axes[0], in_degree_series,  "In-degree (number of citing EC decisions)",    "In-Degree Distribution — EC → EC Network"),
    (axes[1], out_degree_series, "Out-degree (number of distinct EC cases cited)", "Out-Degree Distribution — EC → EC Network"),
]:
    ax.set_facecolor("white")
    ax.hist(series, bins=40, color=COLOR_EC, edgecolor="white")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=13, color="black")
    ax.set_xlabel(xlabel, fontsize=11, color="black")
    ax.set_ylabel("Number of EC cases (log scale)", fontsize=11, color="black")
    ax.tick_params(colors="black")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_ec_ec_dist_degree.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved fig_ec_ec_dist_degree.png")

# --- Cell 20: 12A-2. PageRank Distribution ---
# 12A-2. PageRank Distribution
# Distribution of PageRank scores across all EC nodes. Extremely right-skewed;
# log y-axis used for readability.
fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
ax.set_facecolor("white")
ax.hist(pagerank_series, bins=40, color=COLOR_EC, edgecolor="white")
ax.set_yscale("log")
ax.set_title("PageRank Distribution — EC → EC Network", fontsize=14, color="black")
ax.set_xlabel("PageRank score", fontsize=12, color="black")
ax.set_ylabel("Number of EC cases (log scale)", fontsize=12, color="black")
ax.tick_params(colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_ec_ec_dist_pagerank.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved fig_ec_ec_dist_pagerank.png")

# --- Cell 21: 12A-3. Betweenness Centrality Distribution ---
# 12A-3. Betweenness Centrality Distribution
# Distribution of betweenness centrality across all EC nodes. Most nodes have near-zero
# betweenness; a small number of bridge nodes have substantially higher values.
fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
ax.set_facecolor("white")
ax.hist(betweenness_series, bins=40, color=COLOR_EC, edgecolor="white")
ax.set_yscale("log")
ax.set_title("Betweenness Centrality Distribution — EC → EC Network", fontsize=14, color="black")
ax.set_xlabel("Betweenness centrality", fontsize=12, color="black")
ax.set_ylabel("Number of EC cases (log scale)", fontsize=12, color="black")
ax.tick_params(colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_ec_ec_dist_betweenness.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved fig_ec_ec_dist_betweenness.png")

# --- Cell 22: --- ---
# ---
# 13. Summary Tables
# 13A. Top Cited EC Cases
# Ranked by **in-degree** (raw citation count), also showing **PageRank** (prestige-weighted score).
# These are the EC cases most frequently referenced by other EC documents.
top_ec_cited = (
    ec_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(
        in_degree=lambda df: df.index.map(in_degree_all),
        pagerank =lambda df: df.index.map(pagerank),
    )
    .sort_values(["in_degree", "pagerank"], ascending=False)
    .reset_index(drop=True)
)

print("Top 30 most cited EC cases (ranked by in-degree, then PageRank):")
display(top_ec_cited.head(30))

# --- Cell 23: 13B. Top Citing EC Cases ---
# 13B. Top Citing EC Cases
# Ranked by **out-degree** (number of distinct EC cases cited).
# These are the EC documents that draw most heavily on prior EC antitrust precedents.
top_ec_citing = (
    ec_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(
        out_degree=lambda df: df.index.map(out_degree_all),
    )
    .sort_values("out_degree", ascending=False)
    .reset_index(drop=True)
)

print("Top 30 most citing EC cases (ranked by out-degree):")
display(top_ec_citing.head(30))

# --- Cell 24: 13C. Top EC Cases by Betweenness Centrality ---
# 13C. Top EC Cases by Betweenness Centrality
# Ranked by **betweenness centrality** — these are the EC cases that act as structural bridges
# within the EC citation network, connecting otherwise separate clusters of decisions.
# This is a supplementary structural view; in-degree and PageRank remain the primary metrics.
top_ec_betweenness = (
    ec_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(
        betweenness =lambda df: df.index.map(betweenness),
        in_degree   =lambda df: df.index.map(in_degree_all),
        pagerank    =lambda df: df.index.map(pagerank),
    )
    .sort_values("betweenness", ascending=False)
    .reset_index(drop=True)
)

print("Top 10 EC cases by betweenness centrality:")
display(top_ec_betweenness.head(10))

# --- Cell 25: --- ---
# ---
# 14. Community Detection
# Community detection identifies groups of nodes that are more densely connected to each other
# than to the rest of the network. In a citation network, communities often correspond to
# thematic or procedural clusters — groups of EC decisions that frequently cite each other
# but cite outside the group less often.
# **Modularity** measures how well a partition separates the network into such communities.
# A modularity score close to 1 indicates strong community structure; a score near 0 suggests
# no more clustering than expected by chance.
# **Why this is interesting in same-layer legal citation networks:**
# In the EC → EC network, communities may reflect distinct areas of antitrust practice
# (e.g., merger control, cartel enforcement, abuse of dominance) or different time periods.
# This is an exploratory structural metric — it reveals structural groupings in the citation
# graph, but does not definitively classify cases by doctrine or legal area.
# **Method:** We use the Louvain algorithm built into NetworkX (`nx.community.louvain_communities`),
# which is a standard, well-established modularity-based method. Because Louvain requires an
# undirected graph, we convert the directed EC → EC graph to undirected for this step only.
# All other metrics above remain computed on the original directed graph.
from networkx.algorithms import community as nx_community

# Convert to undirected for community detection
G_undirected = G.to_undirected()

# Run Louvain community detection
louvain_sets = nx_community.louvain_communities(G_undirected, seed=42)

# Build node → community id mapping
partition = {}
for comm_id, members in enumerate(louvain_sets):
    for node in members:
        partition[node] = comm_id

# Compute modularity
modularity_score = nx_community.modularity(G_undirected, louvain_sets)

num_communities = len(louvain_sets)
print(f"Number of communities detected : {num_communities}")
print(f"Modularity score               : {modularity_score:.4f}")

# --- Cell 26: 14.1 Community Summary ---
# 14.1 Community Summary
community_sizes = pd.Series(partition).value_counts().sort_index()
community_summary = pd.DataFrame({
    "community_id": community_sizes.index,
    "num_nodes"   : community_sizes.values,
}).sort_values("num_nodes", ascending=False).reset_index(drop=True)

print(f"Community sizes (largest first):")
display(community_summary)

print(f"\nLargest community : {community_summary['num_nodes'].iloc[0]:,} nodes")
print(f"Smallest community: {community_summary['num_nodes'].iloc[-1]:,} nodes")

# --- Cell 27: 14.2 Community-Level Interpretation Table ---
# 14.2 Community-Level Interpretation Table
# For each community: number of nodes, average in-degree, average PageRank,
# and a few example case numbers.
# Build a lookup: node_id → community
node_community = pd.Series(partition, name="community")

community_details = []
for comm_id, size in community_summary[["community_id", "num_nodes"]].values:
    members = [n for n, c in partition.items() if c == comm_id]
    avg_indeg = sum(in_degree_all.get(n, 0) for n in members) / len(members)
    avg_pr    = sum(pagerank.get(n, 0) for n in members) / len(members)
    examples  = ", ".join(
        G.nodes[n].get("case_number", n)
        for n in sorted(members, key=lambda n: in_degree_all.get(n, 0), reverse=True)[:3]
    )
    community_details.append({
        "community_id"   : int(comm_id),
        "num_nodes"      : int(size),
        "avg_in_degree"  : round(avg_indeg, 2),
        "avg_pagerank"   : round(avg_pr, 6),
        "top_3_examples" : examples,
    })

community_details_df = pd.DataFrame(community_details).sort_values("num_nodes", ascending=False).reset_index(drop=True)
print("Community-level summary (sorted by size):")
display(community_details_df)

# --- Cell 28: 14.3 Add Community to Node Metrics ---
# 14.3 Add Community to Node Metrics
node_metrics["community"] = node_metrics["node_id"].map(partition)

print("Node metrics table now includes community assignment:")
display(node_metrics.sort_values("in_degree", ascending=False).head(20))

# --- Cell 29: 14.4 Visualisation with Community Colours ---
# 14.4 Visualisation with Community Colours
# The same top-50 subgraph as in section 10, but now nodes are coloured by community.
import matplotlib.cm as cm

# Reuse the same subgraph H from section 10
comm_ids = sorted(set(partition.get(n, -1) for n in H.nodes()))
cmap = cm.get_cmap("tab20", len(comm_ids))
comm_color_map = {c: cmap(i) for i, c in enumerate(comm_ids)}

node_colors_comm = [comm_color_map[partition.get(n, -1)] for n in H.nodes()]

fig, ax = plt.subplots(figsize=(14, 10))
nx.draw_networkx_nodes(H, pos, node_color=node_colors_comm, node_size=300, alpha=0.9, ax=ax)
nx.draw_networkx_edges(H, pos, edge_color="#aaaaaa", arrows=True,
                       arrowsize=12, width=0.8, alpha=0.7, ax=ax)
nx.draw_networkx_labels(H, pos, labels=labels, font_size=6, ax=ax)

legend_patches = [
    Patch(facecolor=comm_color_map[c], label=f"Community {c}")
    for c in comm_ids
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=8, title="Community")
ax.set_title("EC → EC Citation Network (Top 50 nodes, coloured by community)", fontsize=13)
ax.axis("off")
plt.tight_layout()
plt.show()

# --- Cell 30: --- ---
# ---
# 14A. Descriptive Figures
# The following figures provide a visual summary of the EC → EC network structure and key metrics
# for the results section. All figures are saved to `outputs/figures/ec_ec/`.
import matplotlib.pyplot as plt

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
})
print("Matplotlib style reset to default (white background).")

# --- Cell 31: 14A-1. Weakly Connected Component Size Distribution ---
# 14A-1. Weakly Connected Component Size Distribution
# This bar chart shows how many weakly connected components exist at each size in the EC → EC network.
# Most components are small isolated pairs or tiny clusters; the dominant component contains the bulk of the network.
size_counts = wcc_summary["size"].value_counts().sort_index()

print(f"Number of weakly connected components      : {len(wccs_sorted):,}")
print(f"Largest WCC size                           : {len(wccs_sorted[0]):,} nodes")
print(f"Nodes in largest WCC (% of total)          : {len(wccs_sorted[0]) / G.number_of_nodes() * 100:.1f}%")

fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ax.set_facecolor("white")
ax.bar(size_counts.index.astype(str), size_counts.values, color=COLOR_EC, edgecolor="white")
ax.set_title("Weakly Connected Components in the EC → EC Network", fontsize=14, color="black")
ax.set_xlabel("Component size", fontsize=12, color="black")
ax.set_ylabel("Number of components", fontsize=12, color="black")
ax.tick_params(axis="x", labelrotation=45, labelsize=11, colors="black")
ax.tick_params(axis="y", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_ec_ec_wcc_distribution.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved fig_ec_ec_wcc_distribution.png")

# --- Cell 32: 14A-2. Strongly Connected Components & Reciprocity Summary ---
# 14A-2. Strongly Connected Components & Reciprocity Summary
# This bar chart summarises the key structural properties of the EC → EC network:
# number of SCCs, non-trivial SCCs, size of the largest SCC, and reciprocity percentage.
num_sccs         = len(sccs_sorted)
num_nontrivial   = sum(1 for c in sccs_sorted if len(c) > 1)
largest_scc_size = len(sccs_sorted[0])
reciprocity_pct  = reciprocity * 100

print(f"Number of strongly connected components    : {num_sccs:,}")
print(f"Non-trivial SCCs (size > 1)                : {num_nontrivial:,}")
print(f"Largest SCC size                           : {largest_scc_size:,} nodes")
print(f"Reciprocity                                : {reciprocity_pct:.2f}%")

# Table-style figure: separate counts from percentages to avoid mixed-scale issues
_rows = [
    ["SCCs (total)",       f"{num_sccs:,}"],
    ["Non-trivial SCCs",   f"{num_nontrivial:,}"],
    ["Largest SCC size",   f"{largest_scc_size:,} nodes"],
    ["Reciprocity",        f"{reciprocity_pct:.2f}%"],
]

fig, ax = plt.subplots(figsize=(6, 2.8), facecolor="white")
ax.set_facecolor("white")
ax.axis("off")
table = ax.table(
    cellText=_rows,
    colLabels=["Metric", "Value"],
    cellLoc="left",
    loc="center",
    colWidths=[0.55, 0.35],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:
        cell.set_facecolor("#e8e8e8")
        cell.set_text_props(fontweight="bold", color="black")
    else:
        cell.set_facecolor("white")
        cell.set_text_props(color="black")
ax.set_title("SCC & Reciprocity Summary — EC → EC Network", fontsize=13, color="black", pad=10)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_ec_ec_scc_reciprocity_summary.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved fig_ec_ec_scc_reciprocity_summary.png")

# --- Cell 33: 14A-3. Top 10 Cited EC Cases by In-Degree ---
# 14A-3. Top 10 Cited EC Cases by In-Degree
# This horizontal bar chart shows the ten EC cases most frequently cited by other EC decisions,
# ranked by in-degree (raw citation count). These are the most referenced EC antitrust precedents.
# Step 1: rank ALL nodes by metric, then take top 10
top10_ec_indeg = (
    ec_nodes.set_index("node_id")[["case_number", "label", "celex_no"]]
    .assign(in_degree=lambda df: df.index.map(in_degree_all))
    .sort_values("in_degree", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

# Step 2: build fallback label AFTER selecting top 10
def _make_ec_label(r):
    cn  = str(r.get("case_number", "")).strip()
    lbl = str(r.get("label", "")).strip()
    cx  = str(r.get("celex_no", "")).strip()
    if cn and lbl:
        return f"{cn} – {lbl[:60]}"
    elif cn:
        return cn
    elif cx:
        return cx
    else:
        return str(r.name)

top10_ec_indeg["bar_label"] = top10_ec_indeg.apply(_make_ec_label, axis=1)

# Consistency check
_n_missing_before = top10_ec_indeg["case_number"].str.strip().eq("").sum()
_n_missing_after  = top10_ec_indeg["bar_label"].str.strip().eq("").sum()
print(f"[Consistency] EC In-Degree Top-10: nodes={len(top10_ec_indeg)}, missing label before fallback={_n_missing_before}, after fallback={_n_missing_after}, plotting={'10' if len(top10_ec_indeg)==10 else str(len(top10_ec_indeg))+' (< 10!)'}")

fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
ax.set_facecolor("white")
_max_val = top10_ec_indeg["in_degree"].max()
bars = ax.barh(top10_ec_indeg["bar_label"][::-1], top10_ec_indeg["in_degree"][::-1],
               color=COLOR_EC, edgecolor="white")
for bar, val in zip(bars, top10_ec_indeg["in_degree"][::-1]):
    ax.text(bar.get_width() + _max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{int(val):,}", va="center", ha="left", fontsize=10, color="black")
ax.set_xlim(0, _max_val * 1.15)
ax.set_title("Top 10 Cited EC Cases by In-Degree", fontsize=14, color="black")
ax.set_xlabel("In-degree (number of citing EC decisions)", fontsize=12, color="black")
ax.tick_params(axis="y", labelsize=9, colors="black")
ax.tick_params(axis="x", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_ec_ec_top10_indegree.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved fig_ec_ec_top10_indegree.png")

# --- Cell 34: 14A-4. Top 10 Citing EC Cases by Out-Degree ---
# 14A-4. Top 10 Citing EC Cases by Out-Degree
# This horizontal bar chart shows the ten EC decisions that cite the most distinct other EC cases,
# ranked by out-degree. These are the EC documents most heavily drawing on prior EC antitrust precedents.
# Step 1: rank ALL nodes by metric, then take top 10
top10_ec_outdeg = (
    ec_nodes.set_index("node_id")[["case_number", "label", "celex_no"]]
    .assign(out_degree=lambda df: df.index.map(out_degree_all))
    .sort_values("out_degree", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

# Step 2: build fallback label AFTER selecting top 10
top10_ec_outdeg["bar_label"] = top10_ec_outdeg.apply(_make_ec_label, axis=1)

# Consistency check
_n_missing_before = top10_ec_outdeg["case_number"].str.strip().eq("").sum()
_n_missing_after  = top10_ec_outdeg["bar_label"].str.strip().eq("").sum()
print(f"[Consistency] EC Out-Degree Top-10: nodes={len(top10_ec_outdeg)}, missing label before fallback={_n_missing_before}, after fallback={_n_missing_after}, plotting={'10' if len(top10_ec_outdeg)==10 else str(len(top10_ec_outdeg))+' (< 10!)'}")

fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
ax.set_facecolor("white")
_max_val = top10_ec_outdeg["out_degree"].max()
bars = ax.barh(top10_ec_outdeg["bar_label"][::-1], top10_ec_outdeg["out_degree"][::-1],
               color=COLOR_EC, edgecolor="white")
for bar, val in zip(bars, top10_ec_outdeg["out_degree"][::-1]):
    ax.text(bar.get_width() + _max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{int(val):,}", va="center", ha="left", fontsize=10, color="black")
ax.set_xlim(0, _max_val * 1.15)
ax.set_title("Top 10 Citing EC Cases by Out-Degree", fontsize=14, color="black")
ax.set_xlabel("Out-degree (number of distinct EC cases cited)", fontsize=12, color="black")
ax.tick_params(axis="y", labelsize=9, colors="black")
ax.tick_params(axis="x", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_ec_ec_top10_outdegree.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved fig_ec_ec_top10_outdegree.png")

# --- Cell 35: 14A-5. Top 10 EC Cases by PageRank ---
# 14A-5. Top 10 EC Cases by PageRank
# This horizontal bar chart shows the ten EC cases with the highest PageRank score in the EC → EC network.
# PageRank accounts for the prestige of citing documents, so cases cited by influential EC decisions rank higher.
# Step 1: rank ALL nodes by metric, then take top 10
top10_ec_pr = (
    ec_nodes.set_index("node_id")[["case_number", "label", "celex_no"]]
    .assign(
        in_degree=lambda df: df.index.map(in_degree_all),
        pagerank =lambda df: df.index.map(pagerank),
    )
    .sort_values("pagerank", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

# Step 2: build fallback label AFTER selecting top 10
top10_ec_pr["bar_label"] = top10_ec_pr.apply(_make_ec_label, axis=1)

# Consistency check
_n_missing_before = top10_ec_pr["case_number"].str.strip().eq("").sum()
_n_missing_after  = top10_ec_pr["bar_label"].str.strip().eq("").sum()
print(f"[Consistency] EC PageRank Top-10: nodes={len(top10_ec_pr)}, missing label before fallback={_n_missing_before}, after fallback={_n_missing_after}, plotting={'10' if len(top10_ec_pr)==10 else str(len(top10_ec_pr))+' (< 10!)'}")

fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
ax.set_facecolor("white")
_max_val = top10_ec_pr["pagerank"].max()
bars = ax.barh(top10_ec_pr["bar_label"][::-1], top10_ec_pr["pagerank"][::-1],
               color=COLOR_EC, edgecolor="white")
for bar, val in zip(bars, top10_ec_pr["pagerank"][::-1]):
    ax.text(bar.get_width() + _max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}", va="center", ha="left", fontsize=10, color="black")
ax.set_xlim(0, _max_val * 1.15)
ax.set_title("Top 10 EC Cases by PageRank", fontsize=14, color="black")
ax.set_xlabel("PageRank score", fontsize=12, color="black")
ax.tick_params(axis="y", labelsize=9, colors="black")
ax.tick_params(axis="x", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_ec_ec_top10_pagerank.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved fig_ec_ec_top10_pagerank.png")

# --- Cell 36: 14A-6. Top 10 EC Cases by Betweenness Centrality ---
# 14A-6. Top 10 EC Cases by Betweenness Centrality
# This horizontal bar chart shows the ten EC cases with the highest betweenness centrality in the EC → EC network.
# These cases act as structural bridges, connecting otherwise separate clusters of EC antitrust decisions.
# Step 1: rank ALL nodes by metric, then take top 10
top10_ec_bw = (
    ec_nodes.set_index("node_id")[["case_number", "label", "celex_no"]]
    .assign(betweenness=lambda df: df.index.map(betweenness))
    .sort_values("betweenness", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

# Step 2: build fallback label AFTER selecting top 10
top10_ec_bw["bar_label"] = top10_ec_bw.apply(_make_ec_label, axis=1)

# Consistency check
_n_missing_before = top10_ec_bw["case_number"].str.strip().eq("").sum()
_n_missing_after  = top10_ec_bw["bar_label"].str.strip().eq("").sum()
print(f"[Consistency] EC Betweenness Top-10: nodes={len(top10_ec_bw)}, missing label before fallback={_n_missing_before}, after fallback={_n_missing_after}, plotting={'10' if len(top10_ec_bw)==10 else str(len(top10_ec_bw))+' (< 10!)'}")

fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
ax.set_facecolor("white")
_max_val = top10_ec_bw["betweenness"].max()
bars = ax.barh(top10_ec_bw["bar_label"][::-1], top10_ec_bw["betweenness"][::-1],
               color=COLOR_EC, edgecolor="white")
for bar, val in zip(bars, top10_ec_bw["betweenness"][::-1]):
    ax.text(bar.get_width() + _max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}", va="center", ha="left", fontsize=10, color="black")
ax.set_xlim(0, _max_val * 1.15)
ax.set_title("Top 10 EC Cases by Betweenness Centrality", fontsize=14, color="black")
ax.set_xlabel("Betweenness centrality", fontsize=12, color="black")
ax.tick_params(axis="y", labelsize=9, colors="black")
ax.tick_params(axis="x", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_ec_ec_top10_betweenness.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved fig_ec_ec_top10_betweenness.png")

# --- Cell 37: 14A-7. Community Size Distribution ---
# 14A-7. Community Size Distribution
# This bar chart shows the size distribution of communities detected by the Louvain algorithm in the EC → EC network.
# Each bar represents one community; the height shows how many EC cases belong to it.
print(f"Number of communities detected : {num_communities:,}")
print(f"Modularity score               : {modularity_score:.4f}")
print(f"Largest community              : {community_summary['num_nodes'].iloc[0]:,} nodes")
print(f"Second largest community       : {community_summary['num_nodes'].iloc[1]:,} nodes" if len(community_summary) > 1 else "")

comm_sizes_sorted = community_summary["num_nodes"].values
_top_n = min(20, len(comm_sizes_sorted))
_top_sizes = comm_sizes_sorted[:_top_n]
_top_labels = [str(i + 1) for i in range(_top_n)]

fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
ax.set_facecolor("white")
bars = ax.bar(_top_labels, _top_sizes, color=COLOR_EC, edgecolor="white")
for bar, val in zip(bars, _top_sizes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(_top_sizes) * 0.01,
            f"{val:,}", ha="center", va="bottom", fontsize=8, color="black")
ax.set_ylim(0, max(_top_sizes) * 1.15)
_subtitle = f" (largest {_top_n} shown)" if len(comm_sizes_sorted) > _top_n else ""
ax.set_title(f"Community Size Distribution — EC → EC Network (Louvain, modularity={modularity_score:.3f})",
             fontsize=13, color="black")
ax.set_xlabel("Community rank", fontsize=12, color="black")
ax.set_ylabel("Number of EC cases", fontsize=12, color="black")
ax.tick_params(axis="x", labelsize=10, colors="black")
ax.tick_params(axis="y", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_ec_ec_community_size_distribution.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved fig_ec_ec_community_size_distribution.png")

# --- Cell 38: --- ---
# ---
# 15. Export
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Node and edge tables
ec_nodes.to_csv(NODES_PATH, index=False, encoding="utf-8")
print(f"Saved {len(ec_nodes):,} nodes to: {NODES_PATH}")

edges.to_csv(EDGES_PATH, index=False, encoding="utf-8")
print(f"Saved {len(edges):,} edges to: {EDGES_PATH}")

# GraphML
nx.write_graphml(G, str(GRAPHML_PATH))
print(f"Saved GraphML to: {GRAPHML_PATH}")

# GEXF with Gephi-compatible viz colours
for node in G.nodes():
    G.nodes[node]["viz"] = {"color": {"r": 76, "g": 114, "b": 176, "a": 1.0}}
nx.write_gexf(G, str(GEXF_PATH))
print(f"Saved GEXF  to: {GEXF_PATH}")

# Node metrics (includes community column)
node_metrics.to_csv(NODE_METRICS_PATH, index=False, encoding="utf-8")
print(f"Saved node metrics to: {NODE_METRICS_PATH}")

