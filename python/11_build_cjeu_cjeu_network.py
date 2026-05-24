# Original notebook: 11_build_cjeu_cjeu_network.ipynb
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
INPUT_PATH = DATA_DIR / "cjeu_cjeu_case_matches.csv"

OUT_DIR           = DATA_DIR / "network" / "cjeu_cjeu"
NODES_PATH        = OUT_DIR / "cjeu_cjeu_nodes.csv"
EDGES_PATH        = OUT_DIR / "cjeu_cjeu_edges.csv"
GRAPHML_PATH      = OUT_DIR / "cjeu_cjeu_network.graphml"
NODE_METRICS_PATH = OUT_DIR / "cjeu_cjeu_node_metrics.csv"

GEXF_PATH   = OUT_DIR / "cjeu_cjeu_network.gexf"
FIGURES_DIR = Path("outputs/figures/cjeu_cjeu")

# ── Node colour map (by case type) ───────────────────────────────────────────
COLOR_MAP = {
    "c_case": {"hex": "#1f77b4", "rgb": {"r": 31,  "g": 119, "b": 180, "a": 1.0}},
    "t_case": {"hex": "#ff7f0e", "rgb": {"r": 255, "g": 127, "b": 14,  "a": 1.0}},
    "other":  {"hex": "#999999", "rgb": {"r": 153, "g": 153, "b": 153, "a": 1.0}},
}

# ── Single main colour for all ranking bar charts ─────────────────────────────
COLOR_CJEU = "#4C72B0"

# ── Shared matplotlib style ───────────────────────────────────────────────────
plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor":"white",
    "text.color":       "black",
    "axes.labelcolor":  "black",
    "xtick.color":      "black",
    "ytick.color":      "black",
    "axes.edgecolor":   "black",
})

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print("Configuration loaded.")

# --- Cell 2: 2. Load CSV ---
# 2. Load CSV
df = pd.read_csv(INPUT_PATH, dtype=str).fillna("")

print(f"Rows loaded : {len(df):,}")
print(f"Columns     : {list(df.columns)}")
df.head(3)

# --- Cell 3: 3. Filter: Only Valid Matches ---
# 3. Filter: Only Valid Matches
matched = df[df["processing_status"].str.strip() == "matched"].copy()

# Drop rows with missing source or target celex ids
matched = matched[
    matched["source_celex_id"].str.strip().ne("") &
    matched["target_celex_id"].str.strip().ne("")
].copy()

# Remove self-citations
matched = matched[
    matched["source_celex_id"].str.strip() != matched["target_celex_id"].str.strip()
].copy()

print(f"Rows after filtering: {len(matched):,}")
print(f"Unique processing_status values: {matched['processing_status'].unique()}")

# --- Cell 4: 4. Deduplicate to One Edge per Source-Target Pair ---
# 4. Deduplicate to One Edge per Source-Target Pair
# Each raw row represents one pattern match. Multiple rows may exist for the same
# source-target pair. We aggregate them into one unique directed edge per
# `(source_celex_id, target_celex_id)` pair, keeping useful aggregated attributes.
edges_raw = (
    matched
    .groupby(["source_celex_id", "target_celex_id"], sort=False)
    .agg(
        source_case_number    = ("source_case_number", "first"),
        target_case_number    = ("target_case_number", "first"),
        match_count           = ("citation_style", "count"),
        citation_styles       = ("citation_style", lambda s: "|".join(s.unique())),
        example_match_context = ("match_context", "first"),
    )
    .reset_index()
)

print(f"Unique CJEU-CJEU pairs (edges): {len(edges_raw):,}")
edges_raw.head(3)

# --- Cell 5: 5. Build Node Table ---
# 5. Build Node Table
# All CJEU cases that appear as source or target are collected into a single node table.
# Source and target metadata are merged carefully so each case appears only once.
# ── Source node metadata ───────────────────────────────────────────────────────
src_meta = (
    matched[["source_celex_id", "source_cellar_id", "source_case_number",
             "source_title", "source_document_date"]]
    .drop_duplicates(subset="source_celex_id")
    .rename(columns={
        "source_celex_id"      : "celex_id",
        "source_cellar_id"     : "cellar_id",
        "source_case_number"   : "case_number",
        "source_title"         : "label",
        "source_document_date" : "date",
    })
    .copy()
)

# ── Target node metadata ───────────────────────────────────────────────────────
tgt_meta = (
    matched[["target_celex_id", "target_cellar_id", "target_case_number", "target_title"]]
    .drop_duplicates(subset="target_celex_id")
    .rename(columns={
        "target_celex_id"    : "celex_id",
        "target_cellar_id"   : "cellar_id",
        "target_case_number" : "case_number",
        "target_title"       : "label",
    })
    .copy()
)
tgt_meta["date"] = ""

# ── Merge: source metadata takes priority ─────────────────────────────────────
all_meta = pd.concat([src_meta, tgt_meta], ignore_index=True)
all_meta = all_meta.drop_duplicates(subset="celex_id", keep="first")

cjeu_nodes = pd.DataFrame({
    "node_id"    : "cjeu:" + all_meta["celex_id"],
    "node_type"  : "cjeu",
    "label"      : all_meta["label"].values,
    "celex_id"   : all_meta["celex_id"].values,
    "cellar_id"  : all_meta["cellar_id"].values,
    "case_number": all_meta["case_number"].values,
    "date"       : all_meta["date"].values,
})

print(f"CJEU nodes: {len(cjeu_nodes):,}")
cjeu_nodes.head(3)

# --- Cell 6: 6. Build Edge Table ---
# 6. Build Edge Table
edges = edges_raw.copy()
edges.insert(0, "source", "cjeu:" + edges["source_celex_id"])
edges.insert(1, "target", "cjeu:" + edges["target_celex_id"])

print(f"Edges: {len(edges):,}")
edges.head(3)

# --- Cell 7: 7. Build NetworkX Directed Graph ---
# 7. Build NetworkX Directed Graph
def classify_case(case_number: str) -> str:
    """Classify a CJEU case number into c_case, t_case, or other."""
    cn = str(case_number).strip()
    if cn.startswith("C-"):
        return "c_case"
    elif cn.startswith("T-"):
        return "t_case"
    else:
        return "other"


G = nx.DiGraph()

for _, row in cjeu_nodes.iterrows():
    ntype = classify_case(row["case_number"])
    G.add_node(
        row["node_id"],
        node_type=ntype,
        node_color=COLOR_MAP[ntype]["hex"],
        label=row["label"],
        celex_id=row["celex_id"],
        cellar_id=row["cellar_id"],
        case_number=row["case_number"],
        date=row["date"],
    )

for _, row in edges.iterrows():
    G.add_edge(
        row["source"],
        row["target"],
        match_count=int(row["match_count"]),
        citation_styles=row["citation_styles"],
        example_match_context=str(row["example_match_context"])[:300],
    )

print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# --- Cell 8: 8. Basic Statistics ---
# 8. Basic Statistics
print(f"CJEU nodes : {len(cjeu_nodes):,}")
print(f"Edges      : {len(edges):,}")
print(f"Is directed: {G.is_directed()}")

# --- Cell 9: --- ---
# ---
# 9. Citation Style Distribution
# A quick look at how many matches come from each citation style (`modern_prefixed` vs `old_unprefixed`).
# This is a descriptive note, not a major analysis section.
style_counts = matched["citation_style"].value_counts().reset_index()
style_counts.columns = ["citation_style", "match_count"]

print("Citation style distribution (raw match rows):")
display(style_counts)

# --- Cell 10 ---
_sc = matched["citation_style"].value_counts()
_label_map = {
    "old_unprefixed":  "Old / unprefixed",
    "modern_prefixed": "Modern / prefixed",
}
_labels_plot = [_label_map.get(s, s) for s in _sc.index]
_values_plot = _sc.values

fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
ax.set_facecolor("white")
bars = ax.bar(_labels_plot, _values_plot, color=COLOR_CJEU, edgecolor="white")
for bar, val in zip(bars, _values_plot):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(_values_plot) * 0.01,
            f"{val:,}", ha="center", va="bottom", fontsize=11, color="black")
ax.set_ylim(0, max(_values_plot) * 1.15)
ax.set_title("CJEU → CJEU Citation Style Distribution", fontsize=13, color="black")
ax.set_ylabel("Number of matches", fontsize=11, color="black")
ax.set_xlabel("Citation style", fontsize=11, color="black")
ax.tick_params(colors="black")
ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_citation_style_distribution.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_citation_style_distribution.png")

# --- Cell 11: 10.1 Reciprocity ---
# 10.1 Reciprocity
# **What it measures:**
# Reciprocity is the fraction of edges that are mutual — if case A cites case B, does case B also cite case A?
# A value of 0 means no mutual citations; 1 means every citation is reciprocated.
# **Why it matters:**
# In a legal citation network, high reciprocity would be unusual (decisions typically cite earlier cases).
# Low reciprocity confirms the expected temporal, hierarchical structure of case law.
recip = nx.reciprocity(G)
print(f"Network reciprocity: {recip:.4f}")
print(f"({recip*100:.2f}% of edges are mutual citations)")

# --- Cell 12: 10.2 Weakly Connected Components ---
# 10.2 Weakly Connected Components
# **What it measures:**
# A weakly connected component (WCC) is a maximal set of nodes reachable from each other
# if edge directions are ignored. A large dominant WCC means most CJEU cases are structurally
# linked through the citation graph.
# **Why it matters:**
# WCCs reveal whether the network forms one large connected cluster or splits into isolated sub-networks.
# Small isolated components may indicate peripheral citation clusters.
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

# --- Cell 13 ---
# WCC distribution figure — show only components of size <= 20 to avoid the large component
# dominating the chart; summarise the large component in the title
_wcc_sizes = [len(c) for c in wccs_sorted]
_largest   = _wcc_sizes[0]
_small_sizes = [s for s in _wcc_sizes if s < _largest]

# Count occurrences of each small-component size
_size_counts = pd.Series(_small_sizes).value_counts().sort_index()

print(f"Largest WCC: {_largest:,} nodes (shown in title, excluded from bar chart)")
print(f"Remaining components: {len(_small_sizes):,} (sizes 1–{max(_small_sizes) if _small_sizes else 0})")

if _size_counts.empty:
    print("All nodes are in a single component — no distribution chart needed.")
else:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(_size_counts.index.astype(str), _size_counts.values, color=COLOR_CJEU, edgecolor="white")
    ax.set_title(
        f"CJEU → CJEU Weakly Connected Components\n"
        f"(largest component: {_largest:,} nodes excluded; {len(_small_sizes):,} smaller components shown)",
        fontsize=12, color="black"
    )
    ax.set_xlabel("Component size (number of nodes)", fontsize=11, color="black")
    ax.set_ylabel("Number of components", fontsize=11, color="black")
    ax.tick_params(axis="x", labelrotation=45, labelsize=10, colors="black")
    ax.tick_params(axis="y", labelsize=11, colors="black")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_wcc_distribution.png",
                dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.show()
    print("Saved: fig_cjeu_cjeu_wcc_distribution.png")

# --- Cell 14: 10.3 Strongly Connected Components ---
# 10.3 Strongly Connected Components
# **What it measures:**
# A strongly connected component (SCC) is a maximal set of nodes where every node can reach
# every other node following directed edges. In a citation network, large SCCs indicate
# cycles of mutual citation.
# **Why it matters:**
# Most legal citation networks are nearly acyclic (older cases cannot cite newer ones),
# so most SCCs will be trivial (size 1). A non-trivial SCC indicates mutual or circular citation patterns.
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

# --- Cell 15 ---
# SCC & Reciprocity summary — table-style figure (avoids mixed-scale issues)
_num_sccs        = len(sccs_sorted)
_num_nontrivial  = sum(1 for c in sccs_sorted if len(c) > 1)
_largest_scc     = len(sccs_sorted[0])
_scc_share       = _largest_scc / G.number_of_nodes() * 100
_recip_pct       = recip * 100

_rows = [
    ["SCCs (total)",             f"{_num_sccs:,}"],
    ["Non-trivial SCCs (>1)",    f"{_num_nontrivial:,}"],
    ["Largest SCC size",         f"{_largest_scc:,} nodes"],
    ["Largest SCC (% of nodes)", f"{_scc_share:.2f}%"],
    ["Reciprocity",              f"{_recip_pct:.2f}%"],
]

fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="white")
ax.set_facecolor("white")
ax.axis("off")
table = ax.table(
    cellText=_rows,
    colLabels=["Metric", "Value"],
    cellLoc="left",
    loc="center",
    colWidths=[0.6, 0.35],
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
ax.set_title("CJEU → CJEU SCC and Reciprocity Summary", fontsize=13, color="black", pad=10)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_scc_reciprocity_summary.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_scc_reciprocity_summary.png")

# --- Cell 16: --- ---
# ---
# 11. Visualisation (Top 50 Nodes)
# A quick visual impression using the 25 most-cited and 25 most-citing CJEU cases.
# Nodes are coloured by role: orange = highly cited, blue = highly citing, green = both.
from matplotlib.patches import Patch

in_deg_series  = pd.Series(dict(G.in_degree()),  name="in_degree").sort_values(ascending=False)
out_deg_series = pd.Series(dict(G.out_degree()), name="out_degree").sort_values(ascending=False)

top_cited_ids  = in_deg_series.head(25).index.tolist()
top_citing_ids = out_deg_series.head(25).index.tolist()

sub_nodes = set(top_cited_ids) | set(top_citing_ids)
H = G.subgraph(sub_nodes).copy()

print(f"Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

color_map_vis = [H.nodes[n]["node_color"] for n in H.nodes()]

labels = {n: H.nodes[n].get("case_number", n) for n in H.nodes()}

fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(H, seed=42, k=1.5)

nx.draw_networkx_nodes(H, pos, node_color=color_map_vis, node_size=300, alpha=0.9, ax=ax)
nx.draw_networkx_edges(H, pos, edge_color="#aaaaaa", arrows=True,
                       arrowsize=12, width=0.8, alpha=0.7, ax=ax)
nx.draw_networkx_labels(H, pos, labels=labels, font_size=6, ax=ax)

legend_elements = [
    Patch(facecolor=COLOR_MAP["c_case"]["hex"], label="C cases"),
    Patch(facecolor=COLOR_MAP["t_case"]["hex"], label="T cases"),
    Patch(facecolor=COLOR_MAP["other"]["hex"],  label="Other / unclassified"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
ax.set_title("CJEU → CJEU Citation Network (Top 50 nodes, coloured by case type)", fontsize=13)
ax.axis("off")
plt.tight_layout()
plt.show()

# --- Cell 17: 12.1 In-Degree and Out-Degree ---
# 12.1 In-Degree and Out-Degree
# **What it measures:**
# - **In-degree** = number of incoming edges → how often a CJEU case is *cited by other CJEU cases*.
# - **Out-degree** = number of outgoing edges → how many other CJEU cases a document *cites*.
# **Why it matters:**
# In-degree is the most direct measure of a case's influence within the CJEU corpus.
# Out-degree reveals which CJEU documents draw most heavily on prior CJEU precedents.
# **Most meaningful for:** both metrics apply equally to all CJEU nodes.
in_degree_all  = dict(G.in_degree())
out_degree_all = dict(G.out_degree())

cjeu_in = (
    cjeu_nodes.set_index("node_id")[["case_number", "label", "date"]]
    .assign(in_degree=lambda df: df.index.map(in_degree_all))
    .sort_values("in_degree", ascending=False)
)

print("Top 20 most cited CJEU cases (in-degree):")
display(cjeu_in.head(20))

# --- Cell 18 ---
cjeu_out = (
    cjeu_nodes.set_index("node_id")[["case_number", "label", "date"]]
    .assign(out_degree=lambda df: df.index.map(out_degree_all))
    .sort_values("out_degree", ascending=False)
)

print("Top 20 most citing CJEU cases (out-degree):")
display(cjeu_out.head(20))

# --- Cell 19 ---
# ── Top 10 by In-Degree figure ────────────────────────────────────────────────
# Step 1: rank ALL nodes by metric, then take top 10
_df_indeg = (
    cjeu_nodes.set_index("node_id")[["case_number", "celex_id"]]
    .assign(in_degree=lambda df: df.index.map(in_degree_all))
    .sort_values("in_degree", ascending=False)
)
top10_indeg_raw = _df_indeg.head(10).copy()

# Step 2: build fallback label AFTER selecting top 10
import re as _re

def _derive_case_number_from_celex(celex_id: str) -> str:
    """Derive a readable case number from a CELEX ID, e.g. 61973CJ0040 -> C-40/73."""
    m = _re.match(r'^6(\d{4})(CJ|CC|CO|TJ|TC|TO|FJ|FC|FO)(\d+)$', celex_id.strip())
    if not m:
        return ""
    year_str, proc, num_str = m.group(1), m.group(2), m.group(3)
    short_year = year_str[-2:]
    num = int(num_str)
    prefix = "T" if proc in ("TJ","TC","TO") else ("F" if proc in ("FJ","FC","FO") else "C")
    return f"{prefix}-{num}/{short_year}"

def _make_label(row):
    cn = str(row.get("case_number", "")).strip()
    cx = str(row.get("celex_id", "")).strip()
    nid = str(row.name).strip()
    if cn and cn not in ("-", ""):
        return cn
    derived = _derive_case_number_from_celex(cx) if cx else ""
    if derived:
        return derived
    elif cx:
        return cx
    else:
        return nid

# Label source stats will be printed per figure below

top10_indeg_raw["bar_label"] = top10_indeg_raw.apply(_make_label, axis=1)

# Consistency check
_n_existing_cn  = (top10_indeg_raw["case_number"].str.strip().ne("") & top10_indeg_raw["case_number"].str.strip().ne("-")).sum()
_n_derived_cn   = top10_indeg_raw.apply(lambda r: bool(_derive_case_number_from_celex(str(r.get("celex_id","")).strip())) and (str(r.get("case_number","")).strip() in ("","-")), axis=1).sum()
_n_celex_fb     = top10_indeg_raw.apply(lambda r: not bool(_derive_case_number_from_celex(str(r.get("celex_id","")).strip())) and bool(str(r.get("celex_id","")).strip()) and (str(r.get("case_number","")).strip() in ("","-")), axis=1).sum()
_n_nodeid_fb    = (top10_indeg_raw["bar_label"] == top10_indeg_raw.index.to_series()).sum()
_n_missing_after = (top10_indeg_raw["bar_label"].str.strip().eq("")).sum()
print(f"[Consistency] In-Degree Top-10: nodes={len(top10_indeg_raw)}, existing_cn={_n_existing_cn}, derived_cn={_n_derived_cn}, celex_fallback={_n_celex_fb}, nodeid_fallback={_n_nodeid_fb}, missing_after={_n_missing_after}, plotting={'10' if len(top10_indeg_raw)==10 else str(len(top10_indeg_raw))+' (< 10!)'}")

top10_indeg = top10_indeg_raw.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ax.set_facecolor("white")
_max_val = top10_indeg["in_degree"].max()
bars = ax.barh(top10_indeg["bar_label"], top10_indeg["in_degree"],
               color=COLOR_CJEU, edgecolor="white")
for bar, val in zip(bars, top10_indeg["in_degree"]):
    ax.text(bar.get_width() + _max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{int(val):,}", va="center", ha="left", fontsize=10, color="black")
ax.set_xlim(0, _max_val * 1.15)
ax.set_title("Top 10 CJEU Cases by In-Degree", fontsize=14, color="black")
ax.set_xlabel("In-degree (number of citing CJEU cases)", fontsize=12, color="black")
ax.tick_params(axis="y", labelsize=9, colors="black")
ax.tick_params(axis="x", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_top10_indegree.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_top10_indegree.png")

# --- Cell 20 ---
# ── Top 10 by Out-Degree figure ───────────────────────────────────────────────
# Step 1: rank ALL nodes by metric, then take top 10
_df_outdeg = (
    cjeu_nodes.set_index("node_id")[["case_number", "celex_id"]]
    .assign(out_degree=lambda df: df.index.map(out_degree_all))
    .sort_values("out_degree", ascending=False)
)
top10_outdeg_raw = _df_outdeg.head(10).copy()
top10_outdeg_raw["bar_label"] = top10_outdeg_raw.apply(_make_label, axis=1)

_n_existing_cn  = (top10_outdeg_raw["case_number"].str.strip().ne("") & top10_outdeg_raw["case_number"].str.strip().ne("-")).sum()
_n_derived_cn   = top10_outdeg_raw.apply(lambda r: bool(_derive_case_number_from_celex(str(r.get("celex_id","")).strip())) and (str(r.get("case_number","")).strip() in ("","-")), axis=1).sum()
_n_celex_fb     = top10_outdeg_raw.apply(lambda r: not bool(_derive_case_number_from_celex(str(r.get("celex_id","")).strip())) and bool(str(r.get("celex_id","")).strip()) and (str(r.get("case_number","")).strip() in ("","-")), axis=1).sum()
_n_nodeid_fb    = (top10_outdeg_raw["bar_label"] == top10_outdeg_raw.index.to_series()).sum()
_n_missing_after = (top10_outdeg_raw["bar_label"].str.strip().eq("")).sum()
print(f"[Consistency] Out-Degree Top-10: nodes={len(top10_outdeg_raw)}, existing_cn={_n_existing_cn}, derived_cn={_n_derived_cn}, celex_fallback={_n_celex_fb}, nodeid_fallback={_n_nodeid_fb}, missing_after={_n_missing_after}, plotting={'10' if len(top10_outdeg_raw)==10 else str(len(top10_outdeg_raw))+' (< 10!)'}")

top10_outdeg = top10_outdeg_raw.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ax.set_facecolor("white")
_max_val = top10_outdeg["out_degree"].max()
bars = ax.barh(top10_outdeg["bar_label"], top10_outdeg["out_degree"],
               color=COLOR_CJEU, edgecolor="white")
for bar, val in zip(bars, top10_outdeg["out_degree"]):
    ax.text(bar.get_width() + _max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{int(val):,}", va="center", ha="left", fontsize=10, color="black")
ax.set_xlim(0, _max_val * 1.15)
ax.set_title("Top 10 CJEU Cases by Out-Degree", fontsize=14, color="black")
ax.set_xlabel("Out-degree (number of cited CJEU cases)", fontsize=12, color="black")
ax.tick_params(axis="y", labelsize=9, colors="black")
ax.tick_params(axis="x", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_top10_outdegree.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_top10_outdegree.png")

# --- Cell 21: 12.2 PageRank ---
# 12.2 PageRank
# **What it measures:**
# PageRank assigns a prestige score to each node based on the number and quality of incoming links.
# A node receives a higher score if it is cited by nodes that are themselves highly cited.
# **Why it matters:**
# Unlike raw in-degree, PageRank accounts for the *importance* of the citing document.
# A CJEU case cited by many influential CJEU judgments will rank higher than one cited
# by less prominent cases. This identifies the most foundational CJEU precedents.
# **Most meaningful for:** all CJEU nodes — identifies the most *prestigious* cases in the network.
pagerank = nx.pagerank(G, alpha=0.85)

cjeu_pr = (
    cjeu_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(
        in_degree=lambda df: df.index.map(in_degree_all),
        pagerank =lambda df: df.index.map(pagerank),
    )
    .sort_values("pagerank", ascending=False)
)

print("Top 20 CJEU cases by PageRank:")
display(cjeu_pr.head(20))

# --- Cell 22 ---
# ── Top 10 by PageRank figure ─────────────────────────────────────────────────
# Step 1: rank ALL nodes by metric, then take top 10
_df_pr = (
    cjeu_nodes.set_index("node_id")[["case_number", "celex_id"]]
    .assign(pagerank=lambda df: df.index.map(pagerank))
    .sort_values("pagerank", ascending=False)
)
top10_pr_raw = _df_pr.head(10).copy()
top10_pr_raw["bar_label"] = top10_pr_raw.apply(_make_label, axis=1)

_n_existing_cn  = (top10_pr_raw["case_number"].str.strip().ne("") & top10_pr_raw["case_number"].str.strip().ne("-")).sum()
_n_derived_cn   = top10_pr_raw.apply(lambda r: bool(_derive_case_number_from_celex(str(r.get("celex_id","")).strip())) and (str(r.get("case_number","")).strip() in ("","-")), axis=1).sum()
_n_celex_fb     = top10_pr_raw.apply(lambda r: not bool(_derive_case_number_from_celex(str(r.get("celex_id","")).strip())) and bool(str(r.get("celex_id","")).strip()) and (str(r.get("case_number","")).strip() in ("","-")), axis=1).sum()
_n_nodeid_fb    = (top10_pr_raw["bar_label"] == top10_pr_raw.index.to_series()).sum()
_n_missing_after = (top10_pr_raw["bar_label"].str.strip().eq("")).sum()
print(f"[Consistency] PageRank Top-10: nodes={len(top10_pr_raw)}, existing_cn={_n_existing_cn}, derived_cn={_n_derived_cn}, celex_fallback={_n_celex_fb}, nodeid_fallback={_n_nodeid_fb}, missing_after={_n_missing_after}, plotting={'10' if len(top10_pr_raw)==10 else str(len(top10_pr_raw))+' (< 10!)'}")

top10_pr = top10_pr_raw.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ax.set_facecolor("white")
_max_val = top10_pr["pagerank"].max()
bars = ax.barh(top10_pr["bar_label"], top10_pr["pagerank"],
               color=COLOR_CJEU, edgecolor="white")
for bar, val in zip(bars, top10_pr["pagerank"]):
    ax.text(bar.get_width() + _max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}", va="center", ha="left", fontsize=10, color="black")
ax.set_xlim(0, _max_val * 1.15)
ax.set_title("Top 10 CJEU Cases by PageRank", fontsize=14, color="black")
ax.set_xlabel("PageRank score", fontsize=12, color="black")
ax.tick_params(axis="y", labelsize=9, colors="black")
ax.tick_params(axis="x", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_top10_pagerank.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_top10_pagerank.png")

# --- Cell 23: 12.3 HITS (Hubs and Authorities) ---
# 12.3 HITS (Hubs and Authorities)
# **What it measures:**
# HITS computes two scores per node:
# - **Hub score**: how well a node points to good authorities (high out-degree to important cases).
# - **Authority score**: how well a node is pointed to by good hubs (cited by important citing cases).
# **Why it matters:**
# In a CJEU citation network, authority score identifies the most foundational precedents,
# while hub score identifies the cases that most systematically reference key precedents.
# This complements PageRank by separating the *cited* role from the *citing* role.
# **Most meaningful for:** both roles are relevant in a same-layer citation network.
hits_hub, hits_authority = nx.hits(G, max_iter=500)

cjeu_hits_auth = (
    cjeu_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(
        in_degree =lambda df: df.index.map(in_degree_all),
        authority =lambda df: df.index.map(hits_authority),
        hub       =lambda df: df.index.map(hits_hub),
    )
    .sort_values("authority", ascending=False)
)

print("Top 20 CJEU cases by HITS authority score:")
display(cjeu_hits_auth.head(20))

# --- Cell 24 ---
# ── Top 10 by HITS Authority figure ──────────────────────────────────────────
# Step 1: rank ALL nodes by metric, then take top 10
_df_auth = (
    cjeu_nodes.set_index("node_id")[["case_number", "celex_id"]]
    .assign(authority=lambda df: df.index.map(hits_authority))
    .sort_values("authority", ascending=False)
)
top10_auth_raw = _df_auth.head(10).copy()
top10_auth_raw["bar_label"] = top10_auth_raw.apply(_make_label, axis=1)

_n_existing_cn  = (top10_auth_raw["case_number"].str.strip().ne("") & top10_auth_raw["case_number"].str.strip().ne("-")).sum()
_n_derived_cn   = top10_auth_raw.apply(lambda r: bool(_derive_case_number_from_celex(str(r.get("celex_id","")).strip())) and (str(r.get("case_number","")).strip() in ("","-")), axis=1).sum()
_n_celex_fb     = top10_auth_raw.apply(lambda r: not bool(_derive_case_number_from_celex(str(r.get("celex_id","")).strip())) and bool(str(r.get("celex_id","")).strip()) and (str(r.get("case_number","")).strip() in ("","-")), axis=1).sum()
_n_nodeid_fb    = (top10_auth_raw["bar_label"] == top10_auth_raw.index.to_series()).sum()
_n_missing_after = (top10_auth_raw["bar_label"].str.strip().eq("")).sum()
print(f"[Consistency] HITS Authority Top-10: nodes={len(top10_auth_raw)}, existing_cn={_n_existing_cn}, derived_cn={_n_derived_cn}, celex_fallback={_n_celex_fb}, nodeid_fallback={_n_nodeid_fb}, missing_after={_n_missing_after}, plotting={'10' if len(top10_auth_raw)==10 else str(len(top10_auth_raw))+' (< 10!)'}")

top10_auth = top10_auth_raw.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ax.set_facecolor("white")
_max_val = top10_auth["authority"].max()
bars = ax.barh(top10_auth["bar_label"], top10_auth["authority"],
               color=COLOR_CJEU, edgecolor="white")
for bar, val in zip(bars, top10_auth["authority"]):
    ax.text(bar.get_width() + _max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}", va="center", ha="left", fontsize=10, color="black")
ax.set_xlim(0, _max_val * 1.15)
ax.set_title("Top 10 CJEU Cases by HITS Authority Score", fontsize=14, color="black")
ax.set_xlabel("HITS authority score", fontsize=12, color="black")
ax.tick_params(axis="y", labelsize=9, colors="black")
ax.tick_params(axis="x", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_top10_hits_authority.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_top10_hits_authority.png")

# --- Cell 25: 12.4 Betweenness Centrality ---
# 12.4 Betweenness Centrality
# **What it measures:**
# Betweenness centrality counts how often a node lies on the shortest path between two other nodes.
# A node with high betweenness acts as a *bridge* or *connector* within the network.
# **Why it matters:**
# In the CJEU citation network, a case with high betweenness is not just frequently cited —
# it structurally connects otherwise separate clusters of CJEU decisions.
# Such cases may represent pivotal precedents that link different areas of CJEU case law.
# **Important note:** Betweenness is a supplementary structural metric.
# In-degree and PageRank remain the primary indicators of importance.
betweenness = nx.betweenness_centrality(G, normalized=True)

cjeu_bw = (
    cjeu_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(
        in_degree   =lambda df: df.index.map(in_degree_all),
        betweenness =lambda df: df.index.map(betweenness),
    )
    .sort_values("betweenness", ascending=False)
)

print("Top 20 CJEU cases by betweenness centrality:")
display(cjeu_bw.head(20))

# --- Cell 26 ---
# ── Top 10 by Betweenness figure ──────────────────────────────────────────────
# Step 1: rank ALL nodes by metric, then take top 10
_df_bw = (
    cjeu_nodes.set_index("node_id")[["case_number", "celex_id"]]
    .assign(betweenness=lambda df: df.index.map(betweenness))
    .sort_values("betweenness", ascending=False)
)
top10_bw_raw = _df_bw.head(10).copy()
top10_bw_raw["bar_label"] = top10_bw_raw.apply(_make_label, axis=1)

_n_existing_cn  = (top10_bw_raw["case_number"].str.strip().ne("") & top10_bw_raw["case_number"].str.strip().ne("-")).sum()
_n_derived_cn   = top10_bw_raw.apply(lambda r: bool(_derive_case_number_from_celex(str(r.get("celex_id","")).strip())) and (str(r.get("case_number","")).strip() in ("","-")), axis=1).sum()
_n_celex_fb     = top10_bw_raw.apply(lambda r: not bool(_derive_case_number_from_celex(str(r.get("celex_id","")).strip())) and bool(str(r.get("celex_id","")).strip()) and (str(r.get("case_number","")).strip() in ("","-")), axis=1).sum()
_n_nodeid_fb    = (top10_bw_raw["bar_label"] == top10_bw_raw.index.to_series()).sum()
_n_missing_after = (top10_bw_raw["bar_label"].str.strip().eq("")).sum()
print(f"[Consistency] Betweenness Top-10: nodes={len(top10_bw_raw)}, existing_cn={_n_existing_cn}, derived_cn={_n_derived_cn}, celex_fallback={_n_celex_fb}, nodeid_fallback={_n_nodeid_fb}, missing_after={_n_missing_after}, plotting={'10' if len(top10_bw_raw)==10 else str(len(top10_bw_raw))+' (< 10!)'}")

top10_bw = top10_bw_raw.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ax.set_facecolor("white")
_max_val = top10_bw["betweenness"].max()
bars = ax.barh(top10_bw["bar_label"], top10_bw["betweenness"],
               color=COLOR_CJEU, edgecolor="white")
for bar, val in zip(bars, top10_bw["betweenness"]):
    ax.text(bar.get_width() + _max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}", va="center", ha="left", fontsize=10, color="black")
ax.set_xlim(0, _max_val * 1.15)
ax.set_title("Top 10 CJEU Cases by Betweenness Centrality", fontsize=14, color="black")
ax.set_xlabel("Betweenness centrality", fontsize=12, color="black")
ax.tick_params(axis="y", labelsize=9, colors="black")
ax.tick_params(axis="x", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_top10_betweenness.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_top10_betweenness.png")

# --- Cell 27: --- ---
# ---
# 13. Node-Level Metric Summary Table
# All core metrics are combined into a single node-level table for easy inspection and export.
# Each row represents one CJEU node with all computed metrics.
node_metrics = cjeu_nodes[["node_id", "label", "case_number", "date"]].copy()

node_metrics["in_degree"]      = node_metrics["node_id"].map(in_degree_all)
node_metrics["out_degree"]     = node_metrics["node_id"].map(out_degree_all)
node_metrics["pagerank"]       = node_metrics["node_id"].map(pagerank)
node_metrics["hits_hub"]       = node_metrics["node_id"].map(hits_hub)
node_metrics["hits_authority"] = node_metrics["node_id"].map(hits_authority)
node_metrics["betweenness"]    = node_metrics["node_id"].map(betweenness)

print(f"Node metrics table: {len(node_metrics):,} rows")
display(node_metrics.sort_values("in_degree", ascending=False).head(20))

# --- Cell 28: --- ---
# ---
# 13A. Metric Summary Statistics and Distributions
# This section provides summary statistics and distribution plots for all node-level metrics
# computed in this notebook. Statistics are based on **all CJEU nodes** in the network,
# not just the top-10 rankings.
# **Metrics covered:**
# - `in_degree` — how often each CJEU case is cited by other CJEU cases
# - `out_degree` — how many other CJEU cases each document cites
# - `pagerank` — prestige-weighted citation score
# - `betweenness` — structural bridge score
# - `hits_authority` — HITS authority score (cited by good hubs)
# - `hits_hub` — HITS hub score (points to good authorities)
from scipy import stats as _scipy_stats

SUMMARY_STATS_PATH = OUT_DIR / "cjeu_cjeu_metric_summary_stats.csv"

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

in_degree_series   = pd.Series({nid: in_degree_all[nid]    for nid in cjeu_nodes["node_id"] if nid in in_degree_all},    name="in_degree")
out_degree_series  = pd.Series({nid: out_degree_all[nid]   for nid in cjeu_nodes["node_id"] if nid in out_degree_all},   name="out_degree")
pagerank_series    = pd.Series({nid: pagerank[nid]          for nid in cjeu_nodes["node_id"] if nid in pagerank},         name="pagerank")
betweenness_series = pd.Series({nid: betweenness[nid]       for nid in cjeu_nodes["node_id"] if nid in betweenness},      name="betweenness")
authority_series   = pd.Series({nid: hits_authority[nid]    for nid in cjeu_nodes["node_id"] if nid in hits_authority},   name="hits_authority")
hub_series         = pd.Series({nid: hits_hub[nid]          for nid in cjeu_nodes["node_id"] if nid in hits_hub},         name="hits_hub")

summary_rows = [
    _summary_stats(in_degree_series,   "in_degree"),
    _summary_stats(out_degree_series,  "out_degree"),
    _summary_stats(pagerank_series,    "pagerank"),
    _summary_stats(betweenness_series, "betweenness"),
    _summary_stats(authority_series,   "hits_authority"),
    _summary_stats(hub_series,         "hits_hub"),
]

summary_stats_df = pd.DataFrame(summary_rows)
print("Metric Summary Statistics — CJEU → CJEU Network")
display(summary_stats_df)

OUT_DIR.mkdir(parents=True, exist_ok=True)
summary_stats_df.to_csv(SUMMARY_STATS_PATH, index=False, encoding="utf-8")
print(f"\nSaved summary stats to: {SUMMARY_STATS_PATH}")

# --- Cell 29: 13A-1. In-Degree and Out-Degree Distributions ---
# 13A-1. In-Degree and Out-Degree Distributions
# Distribution of in-degree and out-degree across all CJEU nodes. Citation distributions are
# typically highly right-skewed; a logarithmic y-axis is used for readability.
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

for ax, series, xlabel, title in [
    (axes[0], in_degree_series,  "In-degree (number of citing CJEU cases)",     "In-Degree Distribution — CJEU → CJEU Network"),
    (axes[1], out_degree_series, "Out-degree (number of distinct CJEU cases cited)", "Out-Degree Distribution — CJEU → CJEU Network"),
]:
    ax.set_facecolor("white")
    ax.hist(series, bins=40, color=COLOR_CJEU, edgecolor="white")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=13, color="black")
    ax.set_xlabel(xlabel, fontsize=11, color="black")
    ax.set_ylabel("Number of CJEU cases (log scale)", fontsize=11, color="black")
    ax.tick_params(colors="black")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_dist_degree.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_dist_degree.png")

# --- Cell 30: 13A-2. PageRank Distribution ---
# 13A-2. PageRank Distribution
# Distribution of PageRank scores across all CJEU nodes. Extremely right-skewed;
# log y-axis used for readability.
fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
ax.set_facecolor("white")
ax.hist(pagerank_series, bins=40, color=COLOR_CJEU, edgecolor="white")
ax.set_yscale("log")
ax.set_title("PageRank Distribution — CJEU → CJEU Network", fontsize=14, color="black")
ax.set_xlabel("PageRank score", fontsize=12, color="black")
ax.set_ylabel("Number of CJEU cases (log scale)", fontsize=12, color="black")
ax.tick_params(colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_dist_pagerank.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_dist_pagerank.png")

# --- Cell 31: 13A-3. Betweenness Centrality Distribution ---
# 13A-3. Betweenness Centrality Distribution
# Distribution of betweenness centrality across all CJEU nodes. Most nodes have near-zero
# betweenness; a small number of bridge nodes have substantially higher values.
fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
ax.set_facecolor("white")
ax.hist(betweenness_series, bins=40, color=COLOR_CJEU, edgecolor="white")
ax.set_yscale("log")
ax.set_title("Betweenness Centrality Distribution — CJEU → CJEU Network", fontsize=14, color="black")
ax.set_xlabel("Betweenness centrality", fontsize=12, color="black")
ax.set_ylabel("Number of CJEU cases (log scale)", fontsize=12, color="black")
ax.tick_params(colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_dist_betweenness.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_dist_betweenness.png")

# --- Cell 32: 13A-4. HITS Authority and Hub Score Distributions ---
# 13A-4. HITS Authority and Hub Score Distributions
# Distribution of HITS authority and hub scores across all CJEU nodes.
# Authority scores identify the most-cited foundational cases; hub scores identify
# the cases that most systematically reference key precedents.
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

for ax, series, xlabel, title in [
    (axes[0], authority_series, "HITS authority score", "HITS Authority Distribution — CJEU → CJEU Network"),
    (axes[1], hub_series,       "HITS hub score",       "HITS Hub Distribution — CJEU → CJEU Network"),
]:
    ax.set_facecolor("white")
    ax.hist(series, bins=40, color=COLOR_CJEU, edgecolor="white")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=13, color="black")
    ax.set_xlabel(xlabel, fontsize=11, color="black")
    ax.set_ylabel("Number of CJEU cases (log scale)", fontsize=11, color="black")
    ax.tick_params(colors="black")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_dist_hits.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_dist_hits.png")

# --- Cell 33: --- ---
# ---
# 14. Summary Tables
# 14A. Top Cited CJEU Cases
# Ranked by **in-degree** (raw citation count), also showing **PageRank** and **HITS authority**.
# These are the CJEU cases most frequently referenced by other CJEU documents.
top_cited = (
    cjeu_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(
        in_degree =lambda df: df.index.map(in_degree_all),
        pagerank  =lambda df: df.index.map(pagerank),
        authority =lambda df: df.index.map(hits_authority),
    )
    .sort_values(["in_degree", "pagerank"], ascending=False)
    .reset_index(drop=True)
)

print("Top 30 most cited CJEU cases (ranked by in-degree, then PageRank):")
display(top_cited.head(30))

# --- Cell 34: 14B. Top Citing CJEU Cases ---
# 14B. Top Citing CJEU Cases
# Ranked by **out-degree** (number of distinct CJEU cases cited), also showing **HITS hub score**.
# These are the CJEU documents that draw most heavily on prior CJEU precedents.
top_citing = (
    cjeu_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(
        out_degree=lambda df: df.index.map(out_degree_all),
        hub       =lambda df: df.index.map(hits_hub),
    )
    .sort_values("out_degree", ascending=False)
    .reset_index(drop=True)
)

print("Top 30 most citing CJEU cases (ranked by out-degree):")
display(top_citing.head(30))

# --- Cell 35: 14C. Top Bridge Cases by Betweenness ---
# 14C. Top Bridge Cases by Betweenness
# Ranked by **betweenness centrality** — these are the CJEU cases that act as structural bridges
# within the citation network, connecting otherwise separate clusters of decisions.
top_bridges = (
    cjeu_nodes.set_index("node_id")[["case_number", "label"]]
    .assign(
        betweenness=lambda df: df.index.map(betweenness),
        in_degree  =lambda df: df.index.map(in_degree_all),
        pagerank   =lambda df: df.index.map(pagerank),
    )
    .sort_values("betweenness", ascending=False)
    .reset_index(drop=True)
)

print("Top 20 CJEU bridge cases by betweenness centrality:")
display(top_bridges.head(20))

# --- Cell 36: --- ---
# ---
# 15. Community Detection
# Community detection identifies groups of nodes that are more densely connected to each other
# than to the rest of the network. In a citation network, communities often correspond to
# thematic or procedural clusters — groups of CJEU decisions that frequently cite each other
# but cite outside the group less often.
# **Modularity** measures how well a partition separates the network into such communities.
# A modularity score close to 1 indicates strong community structure; a score near 0 suggests
# no more clustering than expected by chance.
# **Why this is interesting in same-layer legal citation networks:**
# In the CJEU → CJEU network, communities may reflect distinct areas of EU law
# (e.g., competition law, fundamental rights, internal market) or different time periods of case law.
# This is an exploratory structural metric — it reveals structural groupings in the citation
# graph, but does not definitively classify cases by doctrine or legal area.
# **Method:** We use the Louvain algorithm built into NetworkX (`nx.community.louvain_communities`),
# which is a standard, well-established modularity-based method. Because Louvain requires an
# undirected graph, we convert the directed CJEU → CJEU graph to undirected for this step only.
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

# --- Cell 37: 15.1 Community Summary ---
# 15.1 Community Summary
community_sizes = pd.Series(partition).value_counts().sort_index()
community_summary = pd.DataFrame({
    "community_id": community_sizes.index,
    "num_nodes"   : community_sizes.values,
}).sort_values("num_nodes", ascending=False).reset_index(drop=True)

print("Community sizes (largest first):")
display(community_summary)

print(f"\nLargest community : {community_summary['num_nodes'].iloc[0]:,} nodes")
print(f"Smallest community: {community_summary['num_nodes'].iloc[-1]:,} nodes")

# --- Cell 38: 15.2 Community-Level Interpretation Table ---
# 15.2 Community-Level Interpretation Table
# For each community: number of nodes, average in-degree, average PageRank,
# and a few example case numbers.
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

# --- Cell 39: 15.3 Add Community to Node Metrics ---
# 15.3 Add Community to Node Metrics
node_metrics["community"] = node_metrics["node_id"].map(partition)

print("Node metrics table now includes community assignment:")
display(node_metrics.sort_values("in_degree", ascending=False).head(20))

# --- Cell 40: 15.4 Community Size Distribution ---
# 15.4 Community Size Distribution
comm_sizes_sorted = community_summary["num_nodes"].values

print(f"Number of communities detected : {num_communities:,}")
print(f"Modularity score               : {modularity_score:.4f}")
print(f"Largest community size         : {comm_sizes_sorted[0]:,} nodes")
print(f"Top 5 community sizes          : {list(comm_sizes_sorted[:5])}")

_top_n = min(20, len(comm_sizes_sorted))
_top_sizes = comm_sizes_sorted[:_top_n]
_top_labels = [str(i + 1) for i in range(_top_n)]

fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
ax.set_facecolor("white")
bars = ax.bar(_top_labels, _top_sizes, color=COLOR_CJEU, edgecolor="white")
for bar, val in zip(bars, _top_sizes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(_top_sizes) * 0.01,
            f"{val:,}", ha="center", va="bottom", fontsize=8, color="black")
ax.set_ylim(0, max(_top_sizes) * 1.15)
_subtitle = f" (largest {_top_n} shown)" if len(comm_sizes_sorted) > _top_n else ""
ax.set_title(
    f"CJEU → CJEU Community Size Distribution\n"
    f"(Louvain, modularity = {modularity_score:.4f})",
    fontsize=13, color="black"
)
ax.set_xlabel("Community rank", fontsize=12, color="black")
ax.set_ylabel("Number of CJEU cases", fontsize=12, color="black")
ax.tick_params(axis="x", labelsize=10, colors="black")
ax.tick_params(axis="y", labelsize=11, colors="black")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig_cjeu_cjeu_community_size_distribution.png",
            dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
plt.show()
print("Saved: fig_cjeu_cjeu_community_size_distribution.png")

# --- Cell 41: 15.5 Visualisation with Community Colours ---
# 15.5 Visualisation with Community Colours
# The same top-50 subgraph as in section 11, but now nodes are coloured by community.
import matplotlib.cm as cm

# Reuse the same subgraph H from section 11
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
ax.set_title("CJEU → CJEU Citation Network (Top 50 nodes, coloured by community)", fontsize=13)
ax.axis("off")
plt.tight_layout()
plt.show()

# --- Cell 42: --- ---
# ---
# 16. Export
OUT_DIR.mkdir(parents=True, exist_ok=True)

cjeu_nodes.to_csv(NODES_PATH, index=False, encoding="utf-8")
print(f"Saved {len(cjeu_nodes):,} nodes to: {NODES_PATH}")

edges.to_csv(EDGES_PATH, index=False, encoding="utf-8")
print(f"Saved {len(edges):,} edges to: {EDGES_PATH}")

nx.write_graphml(G, str(GRAPHML_PATH))
print(f"Saved GraphML to: {GRAPHML_PATH}")

# GEXF with Gephi-compatible viz colours
for node in G.nodes():
    ntype = G.nodes[node].get("node_type", "other")
    rgb   = COLOR_MAP.get(ntype, COLOR_MAP["other"])["rgb"]
    G.nodes[node]["viz"] = {"color": rgb}
nx.write_gexf(G, str(GEXF_PATH))
print(f"Saved GEXF  to: {GEXF_PATH}")

# Node metrics (includes community column)
node_metrics.to_csv(NODE_METRICS_PATH, index=False, encoding="utf-8")
print(f"Saved node metrics to: {NODE_METRICS_PATH}")

