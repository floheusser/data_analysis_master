# Original notebook: 08_summary_statistics.ipynb
# Converted to Python script on: 2026-05-24
# Outputs and markdown cells have been removed.
# Code logic has been preserved as closely as possible.

# --- Cell 1: 1. Imports and Configuration ---
# 1. Imports and Configuration
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR   = Path("data/processed")
FIGURE_DIR = Path("outputs/figures/summary")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", "{:.2f}".format)

# ── Consistent visual style ───────────────────────────────────────────────────
plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.edgecolor":    "black",
    "text.color":        "black",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "axes.grid":         True,
    "grid.color":        "#eeeeee",
    "grid.linestyle":    "-",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
})

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


print("Configuration loaded.")

# --- Cell 2: 2. Load Datasets ---
# 2. Load Datasets
def load_csv(path, **kwargs):
    p = Path(path)
    if not p.exists():
        print(f"  [MISSING] {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, dtype=str, encoding="utf-8", **kwargs)
    print(f"  [OK] {p.name}: {len(df):,} rows, {df.shape[1]} columns")
    return df

print("Loading datasets...")
ec_master         = load_csv(DATA_DIR / "ec_antitrust_master.csv")
cjeu_cases        = load_csv(DATA_DIR / "cjeu_cases.csv")
cjeu_ec_matches   = load_csv(DATA_DIR / "cjeu_ec_case_matches.csv")
ec_ec_matches     = load_csv(DATA_DIR / "ec_ec_case_matches.csv")
cjeu_cjeu_matches = load_csv(DATA_DIR / "cjeu_cjeu_case_matches.csv")

# --- Cell 3: 3. Classify CJEU Cases and Deduplicate Pairs ---
# 3. Classify CJEU Cases and Deduplicate Pairs
_CELEX_RE = re.compile(r"^6(?P<year>\d{4})(?P<court>CJ|TJ|FJ)(?P<number>\d+)$", re.IGNORECASE)

def classify_cjeu_celex(celex_id):
    m = _CELEX_RE.match(str(celex_id).strip())
    if not m:
        return "other"
    return "c_case" if m.group("court").upper() == "CJ" else "t_case"

if not cjeu_cases.empty:
    cjeu_cases["group"] = cjeu_cases["celex_id"].apply(classify_cjeu_celex)

def dedup_pairs(df, src_col, tgt_col):
    if df.empty:
        return pd.DataFrame(columns=[src_col, tgt_col])
    return df[[src_col, tgt_col]].drop_duplicates().reset_index(drop=True)

cjeu_ec_pairs   = dedup_pairs(cjeu_ec_matches,   "cjeu_celex_id",         "ec_case_number")
ec_ec_pairs     = dedup_pairs(ec_ec_matches,      "source_ec_case_number", "target_ec_case_number")
cjeu_cjeu_pairs = dedup_pairs(cjeu_cjeu_matches,  "source_celex_id",       "target_celex_id")

print("Classification and deduplication done.")

# --- Cell 4: --- ---
# ---
# Figure 1 – Corpus Overview
# The bar chart below shows the number of cases in each dataset collected for this thesis.
grp_counts = cjeu_cases["group"].value_counts() if not cjeu_cases.empty else pd.Series(dtype=int)
n_ec    = len(ec_master)
n_cjeu  = len(cjeu_cases)
n_c     = int(grp_counts.get("c_case", 0))
n_t     = int(grp_counts.get("t_case", 0))
n_other = int(grp_counts.get("other", 0))

labels = ["EC cases", "CJEU cases (total)", "C cases (CJ)", "T/F cases (TJ/FJ)"]
values = [n_ec, n_cjeu, n_c, n_t]
if n_other > 0:
    labels.append("Unclassified CJEU")
    values.append(n_other)

bar_colors = COLORS[:len(labels)]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
bars = ax.barh(range(len(labels)), values[::-1], color=bar_colors[::-1], edgecolor="white", height=0.55)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels[::-1])
for bar, val in zip(bars, values[::-1]):
    ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", fontsize=10)
ax.set_xlabel("Number of records", labelpad=8)
ax.set_title("Corpus Overview", pad=12)
ax.set_xlim(0, max(values) * 1.22)
ax.grid(axis="x", color="#eeeeee")
ax.grid(axis="y", visible=False)
fig.subplots_adjust(left=0.30, right=0.95, top=0.88, bottom=0.15)
fig.savefig(FIGURE_DIR / "fig_corpus_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white")
plt.close(fig)

# --- Cell 5: --- ---
# ---
# Figure 2 – Temporal Coverage
# The line chart shows the number of cases collected per year for each group. A decade-level bar chart is added for readability.
yr_series = {}

if not ec_master.empty and "date" in ec_master.columns:
    ec_master["year"] = pd.to_datetime(ec_master["date"], errors="coerce").dt.year
    s = ec_master["year"].dropna().astype(int).value_counts().sort_index()
    if not s.empty:
        yr_series["EC cases"] = s

if not cjeu_cases.empty and "document_date" in cjeu_cases.columns:
    cjeu_cases["year"] = pd.to_datetime(cjeu_cases["document_date"], errors="coerce").dt.year
    for grp, lbl in [("c_case", "C cases (CJ)"), ("t_case", "T/F cases (TJ/FJ)")]:
        sub = cjeu_cases.loc[cjeu_cases["group"] == grp, "year"].dropna().astype(int)
        if not sub.empty:
            yr_series[lbl] = sub.value_counts().sort_index()

if yr_series:
    # ── Year-level line chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for (lbl, s), col in zip(yr_series.items(), COLORS):
        ax.plot(s.index, s.values, marker="o", markersize=3, linewidth=1.5,
                label=lbl, color=col)
    ax.set_title("Cases per Year by Dataset Group", pad=12)
    ax.set_xlabel("Year", labelpad=8)
    ax.set_ylabel("Number of cases", labelpad=8)
    ax.legend(title="Dataset", framealpha=0.9)
    all_years = sorted(set().union(*[set(s.index) for s in yr_series.values()]))
    year_ticks = [y for y in all_years if y % 5 == 0]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(y) for y in year_ticks], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_temporal_year.png", dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)

    # ── Decade-level grouped bar chart ────────────────────────────────────────
    decade_data = {}
    for lbl, s in yr_series.items():
        dec = (s.index // 10 * 10)
        decade_data[lbl] = s.groupby(dec).sum()

    all_decades = sorted(set().union(*[d.index for d in decade_data.values()]))
    x     = np.arange(len(all_decades))
    n_grp = len(decade_data)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for i, (lbl, d) in enumerate(decade_data.items()):
        vals   = [int(d.get(dec, 0)) for dec in all_decades]
        offset = (i - n_grp / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width=width, label=lbl,
               color=COLORS[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 2,
                        f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}s" for d in all_decades], rotation=0)
    ax.set_title("Cases per Decade by Dataset Group", pad=12)
    ax.set_xlabel("Decade", labelpad=8)
    ax.set_ylabel("Number of cases", labelpad=8)
    ax.legend(title="Dataset", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_temporal_decade.png", dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
else:
    print("No date data available.")

# --- Cell 6: --- ---
# ---
# Figure 3 – Metadata Completeness
# The chart shows the percentage of non-missing values for the most important metadata fields in each dataset.
def completeness(df, fields):
    result = {}
    for col in fields:
        if col in df.columns:
            pct = df[col].replace("", pd.NA).notna().mean() * 100
            result[col] = round(pct, 1)
        else:
            result[col] = 0.0
    return result

ec_fields   = ["ec_case_number", "date", "celex_no", "document_url"]
cjeu_fields = ["celex_id", "document_date", "cellar_id", "court_level"]

ec_comp   = completeness(ec_master,  ec_fields)
cjeu_comp = completeness(cjeu_cases, cjeu_fields)

ec_labels   = ["Case number", "Date", "CELEX no.", "Document URL"]
cjeu_labels = ["CELEX ID", "Document date", "Cellar ID", "Court level"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("white")
fig.suptitle("Metadata Completeness by Dataset", fontsize=14)

for ax, comp, field_labels, title, col in [
    (axes[0], ec_comp,   ec_labels,   "EC Cases",   COLORS[0]),
    (axes[1], cjeu_comp, cjeu_labels, "CJEU Cases", COLORS[1]),
]:
    ax.set_facecolor("white")
    vals = list(comp.values())
    bars = ax.barh(range(len(field_labels)), vals[::-1], color=col, edgecolor="white", height=0.5)
    ax.set_yticks(range(len(field_labels)))
    ax.set_yticklabels(field_labels[::-1])
    for bar, v in zip(bars, vals[::-1]):
        ax.text(min(v + 1, 101), bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", ha="left", fontsize=10)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Completeness (%)", labelpad=8)
    ax.set_title(title, pad=10)
    ax.grid(axis="x", color="#eeeeee")
    ax.grid(axis="y", visible=False)

fig.tight_layout()
fig.savefig(FIGURE_DIR / "fig_metadata_completeness.png", dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white")
plt.close(fig)

# --- Cell 7: --- ---
# ---
# Figure 4 – Match Overview
# The grouped bar chart compares the three citation match datasets across four dimensions: raw matches, unique source–target pairs, unique source documents, and unique target documents.
def match_stats(raw_df, pairs_df, src_col, tgt_col):
    if raw_df.empty:
        return {"Raw matches": 0, "Unique pairs": 0, "Unique sources": 0, "Unique targets": 0}
    return {
        "Raw matches":    len(raw_df),
        "Unique pairs":   len(pairs_df),
        "Unique sources": pairs_df[src_col].nunique() if not pairs_df.empty else 0,
        "Unique targets": pairs_df[tgt_col].nunique() if not pairs_df.empty else 0,
    }

datasets = {
    "CJEU→EC":   match_stats(cjeu_ec_matches,   cjeu_ec_pairs,   "cjeu_celex_id",         "ec_case_number"),
    "EC→EC":     match_stats(ec_ec_matches,     ec_ec_pairs,     "source_ec_case_number", "target_ec_case_number"),
    "CJEU→CJEU": match_stats(cjeu_cjeu_matches, cjeu_cjeu_pairs, "source_celex_id",       "target_celex_id"),
}

metrics = ["Raw matches", "Unique pairs", "Unique sources", "Unique targets"]
x_metric_labels = ["Raw\nmatches", "Unique\npairs", "Unique\nsources", "Unique\ntargets"]
x     = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

global_max = max(max(s.values()) for s in datasets.values())

for i, (lbl, stats) in enumerate(datasets.items()):
    vals   = [stats[m] for m in metrics]
    offset = (i - 1) * width
    bars   = ax.bar(x + offset, vals, width=width, label=lbl,
                    color=COLORS[i], edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + global_max * 0.01,
                    f"{v:,}", ha="center", va="bottom", fontsize=8, rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(x_metric_labels, fontsize=10)
ax.set_ylabel("Count", labelpad=8)
ax.set_title("Citation Match Overview by Dataset", pad=12)
ax.legend(title="Dataset", framealpha=0.9)
ax.set_ylim(0, global_max * 1.25)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "fig_match_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white")
plt.close(fig)

# Compact summary table
display(pd.DataFrame(datasets, index=metrics).T)

# --- Cell 8: --- ---
# ---
# Figure 5 – Citation Distributions
# The histograms below show how citations are distributed across source and target documents for each match dataset. The x-axis shows the number of citations per document; the y-axis shows how many documents fall into each bin.
dist_data = []

if not cjeu_ec_pairs.empty:
    dist_data.append((cjeu_ec_pairs.groupby("cjeu_celex_id").size(),      "CJEU→EC: Citations made per source (CJEU)",      0))
    dist_data.append((cjeu_ec_pairs.groupby("ec_case_number").size(),     "CJEU→EC: Citations received per target (EC)",    0))

if not ec_ec_pairs.empty:
    dist_data.append((ec_ec_pairs.groupby("source_ec_case_number").size(), "EC→EC: Citations made per source",               1))
    dist_data.append((ec_ec_pairs.groupby("target_ec_case_number").size(), "EC→EC: Citations received per target",           1))

if not cjeu_cjeu_pairs.empty:
    dist_data.append((cjeu_cjeu_pairs.groupby("source_celex_id").size(),  "CJEU→CJEU: Citations made per source",           2))
    dist_data.append((cjeu_cjeu_pairs.groupby("target_celex_id").size(),  "CJEU→CJEU: Citations received per target",       2))

if dist_data:
    n     = len(dist_data)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Citation Count Distributions per Document", fontsize=14, y=1.01)
    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, (s, lbl, col_idx) in zip(axes_flat, dist_data):
        ax.set_facecolor("white")
        ax.hist(s.values, bins=30, color=COLORS[col_idx], edgecolor="white")
        ax.set_title(lbl, fontsize=11, pad=8)
        ax.set_xlabel("Number of citations per document", labelpad=6)
        ax.set_ylabel("Number of documents", labelpad=6)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.savefig(FIGURE_DIR / "fig_citation_distributions.png", dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
else:
    print("No citation data available.")

# --- Cell 9: --- ---
# ---
# Figure 6 – CJEU→CJEU Target Type Breakdown
# This figure classifies the citation targets in the CJEU→CJEU match dataset into three types:
# - **Old CJEU**: early cases without a letter prefix (e.g. `6/64`)
# - **C cases**: Court of Justice cases (prefix `C-`)
# - **T/F cases**: General Court / Court of First Instance cases (prefix `T-` or `F-`)
# Classification is derived directly from the match data, not from `cjeu_cases.csv`.
def classify_target_type(row):
    flag = str(row.get("is_old_case_citation", "")).strip().lower()
    if flag in ("true", "1", "yes"):
        return "Old CJEU"
    cn = str(row.get("target_case_number", "")).strip()
    if re.match(r"^\d+/\d+$", cn):
        return "Old CJEU"
    if cn.startswith("C-"):
        return "C cases"
    if cn.startswith("T-") or cn.startswith("F-"):
        return "T/F cases"
    celex = str(row.get("target_celex_id", "")).strip()
    m = re.match(r"^6\d{4}(CJ|TJ|FJ)\d+$", celex, re.IGNORECASE)
    if m:
        return "C cases" if m.group(1).upper() == "CJ" else "T/F cases"
    return "Other"

def classify_source_type(celex_id):
    m = re.match(r"^6\d{4}(CJ|TJ|FJ)\d+$", str(celex_id).strip(), re.IGNORECASE)
    if not m:
        return "Other"
    return "C cases" if m.group(1).upper() == "CJ" else "T/F cases"

tmp_edges = pd.DataFrame()

if not cjeu_cjeu_matches.empty:
    tmp = cjeu_cjeu_matches.copy()
    tmp["target_type"] = tmp.apply(classify_target_type, axis=1)
    tmp["source_type"] = tmp["source_celex_id"].apply(classify_source_type)

    edge_cols = ["source_celex_id", "target_celex_id", "source_type", "target_type"]
    tmp_edges = tmp[edge_cols].drop_duplicates()

    type_counts = tmp_edges["target_type"].value_counts()
    type_order  = [t for t in ["Old CJEU", "C cases", "T/F cases", "Other"] if t in type_counts.index]
    vals        = [int(type_counts[t]) for t in type_order]
    colors      = [COLORS[i % len(COLORS)] for i in range(len(type_order))]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.bar(type_order, vals, color=colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Citation target type", labelpad=8)
    ax.set_ylabel("Unique citation edges", labelpad=8)
    ax.set_title("CJEU→CJEU: Citation Targets by Case Type", pad=12)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", color="#eeeeee")
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels(type_order, fontsize=10)
    ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_cjeu_cjeu_target_types.png", dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
else:
    print("No CJEU->CJEU match data available.")

# --- Cell 10: --- ---
# ---
# Figure 7 – CJEU→CJEU Source × Target Type Heatmap (Optional)
# This heatmap shows the number of unique citation edges broken down by source type (rows) and target type (columns) within the CJEU→CJEU dataset.
if not tmp_edges.empty:
    crosstab  = pd.crosstab(tmp_edges["source_type"], tmp_edges["target_type"])
    keep_rows = [t for t in ["C cases", "T/F cases", "Other"] if t in crosstab.index]
    keep_cols = [t for t in ["Old CJEU", "C cases", "T/F cases", "Other"] if t in crosstab.columns]
    crosstab  = crosstab.loc[keep_rows, keep_cols]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(crosstab.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns, fontsize=11, rotation=15, ha="right")
    ax.set_yticklabels(crosstab.index, fontsize=11)
    ax.set_xlabel("Citation target type", labelpad=10)
    ax.set_ylabel("Citation source type", labelpad=10)
    ax.set_title("CJEU→CJEU: Source × Target Type (unique edges)", pad=12)

    vmax = crosstab.values.max()
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            val        = int(crosstab.iloc[i, j])
            text_color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=10, color=text_color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of unique citation edges", labelpad=10)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_cjeu_cjeu_heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
else:
    print("No CJEU->CJEU data available for heatmap.")

