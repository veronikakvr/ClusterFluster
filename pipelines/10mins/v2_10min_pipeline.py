# %%
# =============================================================================
# BEHAVIORAL CLUSTERING PIPELINE  v2
# Social Open Field — W_ Animals
# Groups: control, prenatal stress (PNS), early life stress (ELS),
#         prenatal + early life stress (PNELS)
#
# Pipeline overview:
#   0.  Environment setup
#   1.  Data loading & time binning
#   2.  Filtering & imputation
#   3.  Scaling → UMAP → HDBSCAN     (uses BEST params from Section 3a)
#   3a. PARAMETER SEARCH (UMAP × HDBSCAN grid, multi-metric)  ← NEW
#   4.  Cluster validation metrics
#   5.  Colour palette
#   6.  UMAP scatter
#   7.  Behavioral profiles (radar + heatmap)
#   8.  Cluster composition (chi-square + Cramér's V)
#   9.  Kruskal-Wallis + Dunn post-hoc
#  10.  Violin plots with significance brackets
#  11.  PERMANOVA per cluster (condition, sex)
#  12.  Temporal dynamics (polar plots)
#  13.  Transition analysis — counts, Markov matrices,
#       NETWORK GRAPHS (per condition × sex)               ← RESTORED
#  14.  Behavioral entropy
#  15.  Cluster–biomarker correlations (e.g. cFos)
#  16.  Most-distinct cluster selection
# =============================================================================

# %%
# ── 0. Environment (must come first, before numpy is imported) ────────────────
import os
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
 

# %%
# ── Imports ───────────────────────────────────────────────────────────────────
import math, random, warnings, itertools
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx
import scikit_posthocs as sp
 
from scipy.stats import kruskal, chi2_contingency, entropy as shannon_entropy, spearmanr
from scipy.spatial.distance import pdist, squareform
 
from sklearn.experimental import enable_iterative_imputer   # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)
 
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
import statsmodels.api as sm
 
import umap.umap_ as umap
import hdbscan
from hdbscan.validity import validity_index
 
from skbio.stats.distance import permanova, DistanceMatrix
from pypalettes import load_cmap
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# %%
# =============================================================================
# CONFIGURATION  — edit this block only
# =============================================================================
DATA_PATH      = "/Users/veronika/Nextcloud/SOWMYA_PNAS_revision/master_combined_S_revisions_Social2_FINAL2.csv"
BIOMARKER_PATH = None   # e.g. "/path/to/cfos_data.csv"
 
INTERVAL_S         = 2
LAST_LABEL         = "0 days 00:09:58 - 0 days 00:10:00"
EXCLUDE_IDS        = []
EXCLUDE_CONDITIONS = []
RANDOM_STATE       = 42

# --- Data processing settings ─────────────────────────────────────────────────
# selection of cohorts to include in the analysis (e.g. only controls and PNS)
INCLUDE_CONDITIONS = ["ctrl", "PNELS"] # "ctrl", "PNS", "ELS", "PNELS"
INCLUDE_COHORTS = ["RNAseq"] # "cFOS" or set to None to include all
INCLUDE_SEXES       = ["M", "F"]   # set to None to include all


# ── Final UMAP / HDBSCAN parameters (overridden by param search if RUN_PARAM_SEARCH=True)
UMAP_N_NEIGHBORS         = 15 #10
UMAP_MIN_DIST            = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 500 #300
HDBSCAN_MIN_SAMPLES      = 90 #80
 
# ── Parameter search settings ─────────────────────────────────────────────────
RUN_PARAM_SEARCH = True    # set False to skip and use values above directly
 
UMAP_N_NEIGHBORS_GRID  = [5, 10, 15, 30]
UMAP_MIN_DIST_GRID     = [0.0, 0.05, 0.1]
HDBSCAN_MCS_GRID       = [100, 200, 300, 500, 600, 700]   # min_cluster_size
HDBSCAN_MS_GRID        = [20, 50, 60, 80, 90, 100, 120]      # min_samples
# Composite score weights for parameter selection
W_DBCV   = 0.40   # DBCV   (higher = better)
W_SIL    = 0.25   # Silhouette (higher = better)
W_NOISE  = 0.20   # noise fraction penalty (lower = better → negated)
W_NCLU   = 0.15   # number-of-clusters stability (prefer 4–10)
 
# ── Visual / labelling config ─────────────────────────────────────────────────
CONDITION_COLORS = {
    "ctrl":  "#3bf7a9",
    "PNELS": "#8c88ff",
    "PNS":   "#d37370",
    "ELS":   "#bfac00",
}
CONDITION_ORDER = ["ctrl", "PNS", "ELS", "PNELS"]

BEHAVIOR_GROUPS = {
    "social_W": [
        'W_B_nose2tail', 'W_B_nose2body', 'W_B_following'
    ],
    "social_B": [
        'B_W_nose2nose', 'B_W_nose2tail', 'B_W_nose2body', 'B_W_following',
        'B_W_sidebyside', 'B_W_sidereside'
    ],
    "arena_W": [
        'W_climb-arena', 'W_sniff-arena'
    ],
    "arena_B": [
        'B_climb-arena', 'B_sniff-arena'
    ],
    "states_W": [
        'W_immobility', 'W_stat-lookaround', 'W_stat-active',
        'W_stat-passive', 'W_moving', 'W_sniffing'
    ],
    "states_B": [
        'B_immobility', 'B_stat-lookaround', 'B_stat-active',
        'B_stat-passive', 'B_moving', 'B_sniffing'
    ]
}

SELECTED_GROUPS = ["social_W", "arena_W", "states_W"]  # <- change THIS only

BEHAVIOR_COLS = [
    col for group in SELECTED_GROUPS
    for col in BEHAVIOR_GROUPS[group]
]
 
#BEHAVIOR_COLS = [
#    'B_W_nose2nose', 'B_W_sidebyside', 'B_W_sidereside',
#    'W_B_nose2tail',  'W_B_nose2body',  'W_B_following',
#    'W_climb-arena',  'W_sniff-arena',  'W_immobility',
#    'W_stat-lookaround', 'W_stat-active', 'W_stat-passive',
#    'W_moving',       'W_sniffing',
#    'B_W_nose2tail',  'B_W_nose2body',  'B_W_following',
#    'B_climb-arena',  'B_sniff-arena',  'B_immobility',
#    'B_stat-lookaround', 'B_stat-active', 'B_stat-passive',
#    'B_moving',       'B_sniffing',
#]
 
LABEL_MAP = {
    "W_B_following": "W following",  "W_stat-active": "W stat-active",
    "W_B_nose2body": "W nose2body",  "B_W_sidebyside": "sidebyside",
    "W_sniff-arena": "W sniff-arena","W_immobility":  "W immobility",
    "W_sniffing":    "W sniffing",   "W_climb-arena": "W climb-arena",
    "B_W_nose2nose": "nose2nose",    "B_W_sidereside": "sidereside",
    "W_B_nose2tail": "W nose2tail",  "W_stat-passive": "W stat-passive",
    "W_moving":      "W moving",     "W_stat-lookaround": "W stat-lookaround",
    "B_W_nose2tail": "B nose2tail",  "B_W_nose2body": "B nose2body",
    "B_W_following": "B following",  "B_climb-arena": "B climb-arena",
    "B_sniff-arena": "B sniff-arena","B_immobility":  "B immobility",
    "B_stat-lookaround": "B stat-lookaround",
    "B_stat-active": "B stat-active","B_stat-passive": "B stat-passive",
    "B_moving":      "B moving",     "B_sniffing":    "B sniffing",
    "W_B_nose2tail": "W nose2tail",  "W_B_nose2body": "W nose2body",
}
 
TOP_K = 3   # most distinct clusters to highlight in Section 16

# %%
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
 
def p_to_symbol(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "#"
    return "ns"
 
def cramers_v(chi2_stat, n, min_dim):
    return np.sqrt(chi2_stat / (n * (min_dim - 1)))
 
def add_sig_brackets(ax, pairs, y_max, y_step=None):
    if y_step is None:
        y_step = max(abs(y_max) * 0.07, 0.01)
    for idx, (i, j) in enumerate(pairs):
        y = y_max + y_step * (idx + 1)
        ax.plot([i, i, j, j],
                [y, y + y_step * 0.4, y + y_step * 0.4, y], lw=1.5, c="k")
        ax.text((i + j) / 2, y + y_step * 0.45, "*",
                ha="center", va="bottom", fontsize=14, weight="bold")
 
def scale_array(arr, lo, hi):
    """Min-max scale an array to [lo, hi]."""
    rng = arr.max() - arr.min()
    if rng < 1e-9:
        return np.full_like(arr, (lo + hi) / 2, dtype=float)
    return lo + (hi - lo) * (arr - arr.min()) / rng

# %%
# =============================================================================
# SECTION 1 — DATA LOADING & PREPROCESSING
# =============================================================================
print("=" * 60)
print("SECTION 1 — Loading data")
print("=" * 60)
 
df_raw = pd.read_csv(DATA_PATH)
print(f"  Raw shape: {df_raw.shape}")
 
df_raw["Time"] = pd.to_timedelta(df_raw["Time"])
 
# ── 2-second interval bins ────────────────────────────────────────────────────
df_raw["Interval_bin"]   = (df_raw["Time"].dt.total_seconds() // INTERVAL_S).astype(int)
df_raw["Interval_start"] = pd.to_timedelta(df_raw["Interval_bin"] * INTERVAL_S, unit="s")
df_raw["Interval_end"]   = pd.to_timedelta((df_raw["Interval_bin"] + 1) * INTERVAL_S, unit="s")
df_raw["Interval_label"] = (df_raw["Interval_start"].astype(str) + " - " +
                             df_raw["Interval_end"].astype(str))

# Filter to include only specified conditions and sexes
if INCLUDE_CONDITIONS is not None:
    df_raw = df_raw[df_raw["condition"].isin(INCLUDE_CONDITIONS)]
if INCLUDE_SEXES is not None:
    df_raw = df_raw[df_raw["sex"].isin(INCLUDE_SEXES)]
if INCLUDE_COHORTS is not None:
    df_raw = df_raw[df_raw["cohort"].isin(INCLUDE_COHORTS)]
 
# ── Aggregate: mean per animal × 2-s bin ─────────────────────────────────────
GROUP_COLS = ["Interval_bin", "Interval_label", "sex", "experimental_id", "condition"]
agg_df = (
    df_raw
    .groupby(GROUP_COLS)[BEHAVIOR_COLS]
    .mean()
    .reset_index()
)
 
print(f"  Aggregated shape: {agg_df.shape}")
print(f"  Missing values:\n{agg_df[BEHAVIOR_COLS].isnull().sum().to_string()}")
 


# %%
# =============================================================================
# SECTION 2 — FILTERING & IMPUTATION
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2 — Filtering & imputation")
print("=" * 60)
 
# Filter
filtered_df = agg_df.copy()
if EXCLUDE_IDS:
    filtered_df = filtered_df[~filtered_df["experimental_id"].isin(EXCLUDE_IDS)]
if EXCLUDE_CONDITIONS:
    filtered_df = filtered_df[~filtered_df["condition"].isin(EXCLUDE_CONDITIONS)]
filtered_df = filtered_df[filtered_df["Interval_label"] <= LAST_LABEL]
filtered_df = filtered_df.sort_values(["experimental_id", "Interval_label"]).reset_index(drop=True)
 
print(f"  Filtered shape: {filtered_df.shape}")
print(f"  Conditions: {filtered_df['condition'].value_counts().to_dict()}")
 
# Multiple imputation (IterativeImputer, Bayesian Ridge internally)
df_imp = filtered_df.fillna(0)
#imputer  = IterativeImputer(random_state=RANDOM_STATE, max_iter=10)
#df_imp   = filtered_df.copy()
#df_imp[BEHAVIOR_COLS] = imputer.fit_transform(filtered_df[BEHAVIOR_COLS])
 
print(f"  Post-imputation NaN: {df_imp[BEHAVIOR_COLS].isnull().sum().sum()}")
 

# %%
# =============================================================================
# SECTION 3a — PARAMETER SEARCH  (UMAP × HDBSCAN grid)
# =============================================================================
# Strategy
# ─────────
# We sweep a grid of UMAP (n_neighbors, min_dist) × HDBSCAN
# (min_cluster_size, min_samples) combinations and score each with four
# complementary metrics:
#
#   • DBCV          — density-based validity (HDBSCAN-native, range −1→1,
#                     higher is better; the most meaningful metric for HDBSCAN)
#   • Silhouette    — mean inter/intra-cluster distance (−1→1, higher better)
#   • Noise %       — fraction of points labelled as noise (penalised)
#   • n_clusters    — number of clusters found; we softly prefer 4–10
#                     (biological plausibility) via a tent function
#
# These four are normalised to [0,1] and combined with configurable weights
# (W_DBCV, W_SIL, W_NOISE, W_NCLU) into a single composite score.
# The best-scoring parameter set is then used for the main analysis.
#
# Tip: the search can be slow for large grids. Narrow the ranges once you have
# a rough idea of good values and re-run.
# =============================================================================


# %%
# %%
# =============================================================================
# SECTION 3 — SCALING, UMAP, HDBSCAN
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3 — Scaling → UMAP → HDBSCAN")
print("=" * 60)
 
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(df_imp[BEHAVIOR_COLS])
print(X_scaled.shape)

# %%
if RUN_PARAM_SEARCH:
    print("\n" + "=" * 60)
    print("SECTION 3a — Parameter search (UMAP × HDBSCAN grid)")
    print("=" * 60)

    grid = list(itertools.product(
        UMAP_N_NEIGHBORS_GRID,
        UMAP_MIN_DIST_GRID,
        HDBSCAN_MCS_GRID,
        HDBSCAN_MS_GRID,
    ))
    print(f"  Grid size: {len(grid)} combinations — this may take a few minutes …")

    search_rows = []
    umap_cache = {}  # 🔥 CACHE

    for nn, md, mcs, ms in grid:
        try:
            # ── UMAP (cached) ────────────────────────────────────────────────
            key = (nn, md)
            if key not in umap_cache:
                umap_cache[key] = umap.UMAP(
                    n_components=2,
                    n_neighbors=nn,
                    min_dist=md,
                    metric="euclidean",
                    random_state=RANDOM_STATE,
                    verbose=False,
                ).fit_transform(X_scaled)

            emb_s = umap_cache[key]

            # ── HDBSCAN ──────────────────────────────────────────────────────
            cl_s  = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            lbl_s = cl_s.fit_predict(emb_s)

            n_clu   = len(set(lbl_s)) - (1 if -1 in lbl_s else 0)
            noise_f = (lbl_s == -1).mean()

            if n_clu < 2:
                continue  # cleaner: skip useless configs

            # ── DBCV (subsample for speed) ───────────────────────────────────
            if len(emb_s) > 10000:
                idx = np.random.choice(len(emb_s), 10000, replace=False)
                dbcv_s = validity_index(
                    emb_s[idx].astype(np.float64),
                    lbl_s[idx]
                )
            else:
                dbcv_s = validity_index(
                    emb_s.astype(np.float64),
                    lbl_s
                )

            # ── Silhouette (non-noise only, safe sampling) ───────────────────
            mask_s = lbl_s != -1
            n_samples = mask_s.sum()

            if n_samples > 1:
                sample_size = min(10000, n_samples)
                sil_s = silhouette_score(
                    emb_s[mask_s],
                    lbl_s[mask_s],
                    sample_size=sample_size,
                    random_state=RANDOM_STATE
                )
            else:
                sil_s = np.nan

            search_rows.append(dict(
                nn=nn, md=md, mcs=mcs, ms=ms,
                n_clusters=n_clu,
                noise_frac=noise_f,
                dbcv=dbcv_s,
                silhouette=sil_s,
                composite=np.nan
            ))

        except Exception as exc:
            search_rows.append(dict(
                nn=nn, md=md, mcs=mcs, ms=ms,
                n_clusters=0,
                noise_frac=1.0,
                dbcv=np.nan,
                silhouette=np.nan,
                composite=np.nan,
                error=str(exc)
            ))

    srch = pd.DataFrame(search_rows)

    print(f"  Valid combinations: {srch[['dbcv','silhouette']].dropna().shape[0]} / {len(srch)}")

    # ── Handle NaNs more gracefully (penalize, don't discard) ────────────────
    valid = srch.copy()

    valid["dbcv"] = valid["dbcv"].fillna(valid["dbcv"].min())
    valid["silhouette"] = valid["silhouette"].fillna(valid["silhouette"].min())

    # ── Normalisation ────────────────────────────────────────────────────────
    valid["dbcv_n"]  = scale_array(valid["dbcv"].values, 0, 1)
    valid["sil_n"]   = scale_array(valid["silhouette"].values, 0, 1)
    valid["noise_n"] = 1 - scale_array(valid["noise_frac"].values, 0, 1)

    nc_vals         = valid["n_clusters"].values.astype(float)
    tent            = 1 - np.abs(nc_vals - 6) / 6
    valid["nclu_n"] = np.clip(tent, 0, 1)

    valid["composite"] = (
        W_DBCV  * valid["dbcv_n"] +
        W_SIL   * valid["sil_n"] +
        W_NOISE * valid["noise_n"] +
        W_NCLU  * valid["nclu_n"]
    )

    srch.update(valid[["composite"]])

    if valid.empty:
        raise ValueError("All parameter combinations failed. Check errors in 'srch'.")

    best = valid.sort_values("composite", ascending=False).iloc[0]

    print("\n  Top 10 parameter combinations:")
    cols_show = ["nn", "md", "mcs", "ms", "n_clusters",
                 "noise_frac", "dbcv", "silhouette", "composite"]
    print(valid.sort_values("composite", ascending=False)
               .head(10)[cols_show].to_string(index=False))

    print(f"\n  ★ Best combination selected:")
    print(f"    UMAP  n_neighbors={int(best.nn)}, min_dist={best.md}")
    print(f"    HDBSCAN min_cluster_size={int(best.mcs)}, min_samples={int(best.ms)}")
    print(f"    → n_clusters={int(best.n_clusters)}, noise={best.noise_frac:.1%}, "
          f"DBCV={best.dbcv:.3f}, Sil={best.silhouette:.3f}, "
          f"Composite={best.composite:.3f}")
# Override main parameters
UMAP_N_NEIGHBORS         = int(best.nn)
UMAP_MIN_DIST            = float(best.md)
HDBSCAN_MIN_CLUSTER_SIZE = int(best.mcs)
HDBSCAN_MIN_SAMPLES      = int(best.ms)

# %%
# ── Visualisation 1: composite score heatmap (per UMAP setting) ──────────
# One sub-heatmap per (n_neighbors, min_dist) pair, axes = mcs × ms
umap_pairs = list(itertools.product(UMAP_N_NEIGHBORS_GRID, UMAP_MIN_DIST_GRID))
n_up = len(umap_pairs)
ncols_h = 3
nrows_h = math.ceil(n_up / ncols_h)
 
fig, axes_h = plt.subplots(nrows_h, ncols_h,
                            figsize=(6 * ncols_h, 5 * nrows_h),
                            squeeze=False)
axes_h_flat = axes_h.flatten()
 
for idx_up, (nn_h, md_h) in enumerate(umap_pairs):
    sub_h = valid[(valid["nn"] == nn_h) & (valid["md"] == md_h)]
    if sub_h.empty:
        axes_h_flat[idx_up].axis("off")
        continue
    pivot_h = sub_h.pivot(index="mcs", columns="ms", values="composite")
    ax_h = axes_h_flat[idx_up]
    sns.heatmap(pivot_h, annot=True, fmt=".2f", cmap="YlGnBu",
                vmin=0, vmax=1, ax=ax_h,
                annot_kws={"size": 9, "weight": "bold"},
                cbar_kws={"label": "Composite score"})
    ax_h.set_title(f"UMAP  n_neigh={nn_h}, min_dist={md_h}",
                    fontsize=11, weight="bold")
    ax_h.set_xlabel("HDBSCAN min_samples", fontsize=10, weight="bold")
    ax_h.set_ylabel("HDBSCAN min_cluster_size", fontsize=10, weight="bold")
 
for i in range(n_up, nrows_h * ncols_h):
    axes_h_flat[i].axis("off")
 
plt.suptitle("Parameter Search — Composite Score (DBCV × Silhouette × Noise × n_clusters)",
                fontsize=13, weight="bold", y=1.01)
plt.tight_layout()
#plt.savefig("ParamSearch_composite_heatmap.pdf", bbox_inches="tight")
plt.show()
 

# %%
# ── Visualisation 2: individual metric profiles for top 20 combos ────────
top20 = valid.sort_values("composite", ascending=False).head(20).copy()
top20["label"] = (top20.apply(
    lambda r: f"nn={int(r.nn)}\nmd={r.md}\nmcs={int(r.mcs)}\nms={int(r.ms)}", axis=1))
 
fig, axes_m = plt.subplots(2, 2, figsize=(14, 9))
metric_info = [
    ("dbcv",        "DBCV (higher better)",       "steelblue"),
    ("silhouette",  "Silhouette (higher better)",  "seagreen"),
    ("noise_frac",  "Noise fraction (lower better)","tomato"),
    ("n_clusters",  "n_clusters",                   "darkorange"),
]
for ax_m, (col_m, title_m, col_c) in zip(axes_m.flatten(), metric_info):
    ax_m.barh(range(len(top20)), top20[col_m].values[::-1],
                color=col_c, alpha=0.75, edgecolor="black")
    ax_m.set_yticks(range(len(top20)))
    ax_m.set_yticklabels(top20["label"].values[::-1], fontsize=4)
    ax_m.set_title(title_m, fontsize=12, weight="bold")
    ax_m.axvline(top20[col_m].values[0], color="black",
                    lw=1.5, linestyle="--", label="best")
    ax_m.legend(fontsize=9)
 
plt.suptitle("Top 20 combinations — individual metrics", fontsize=13, weight="bold")
plt.tight_layout()
#plt.savefig("ParamSearch_top20_metrics.pdf", bbox_inches="tight")
plt.show()
 

# %%
# ── Visualisation 3: scatter DBCV vs Silhouette, bubble = noise ──────────
fig, ax_sc = plt.subplots(figsize=(8, 6))
sc = ax_sc.scatter(valid["dbcv"], valid["silhouette"],
                    c=valid["composite"], cmap="YlOrRd",
                    s=200 * (1 - valid["noise_frac"]) + 20,
                    edgecolors="grey", linewidths=0.5, alpha=0.8)
ax_sc.scatter(best.dbcv, best.silhouette, marker="*", s=400,
                color="black", zorder=5, label="Best combo")
plt.colorbar(sc, ax=ax_sc, label="Composite score")
ax_sc.set_xlabel("DBCV", fontsize=14, weight="bold")
ax_sc.set_ylabel("Silhouette score", fontsize=14, weight="bold")
ax_sc.set_title("DBCV vs Silhouette\n(bubble size ∝ 1 − noise fraction)",
                fontsize=13, weight="bold")
ax_sc.legend(fontsize=11)
ax_sc.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
#plt.savefig("ParamSearch_DBCV_vs_Silhouette.pdf", bbox_inches="tight")
plt.show()
 
# Save full search table
#srch.to_csv("ParamSearch_full_results.csv", index=False)
#print("  Full search results saved → ParamSearch_full_results.csv")
 
#else:
 #   print("\n  [Param search skipped — using values from CONFIG block]")

# %%
# =============================================================================
# SECTION 3 — FINAL SCALING → UMAP → HDBSCAN
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3 — Scaling → UMAP → HDBSCAN  (final run)")
print("=" * 60)
print(f"  UMAP  n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}")
print(f"  HDBSCAN  mcs={HDBSCAN_MIN_CLUSTER_SIZE}, ms={HDBSCAN_MIN_SAMPLES}")
 
reducer = umap.UMAP(
    n_components=2, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
    metric="euclidean", random_state=RANDOM_STATE, verbose=False,
)
embedding = reducer.fit_transform(X_scaled)
 
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples=HDBSCAN_MIN_SAMPLES,
)
labels = clusterer.fit_predict(embedding)
df_imp["Cluster"] = labels
 
n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
noise_pct        = (labels == -1).mean() * 100
print(f"  Clusters found : {n_clusters_found}")
print(f"  Noise points   : {noise_pct:.1f}%")
 

# %%
# =============================================================================
# SECTION 4 — CLUSTER VALIDATION
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4 — Cluster validation metrics")
print("=" * 60)
 
dbcv_final = validity_index(embedding.astype(np.float64), clusterer.labels_)
print(f"  DBCV (validity index)    : {dbcv_final:.3f}   (range −1→1, higher better)")
 
mask_valid = df_imp["Cluster"] != -1
X_cl       = embedding[mask_valid]
y_cl       = df_imp["Cluster"][mask_valid].values
 
if len(np.unique(y_cl)) > 1:
    sil_f = silhouette_score(X_cl, y_cl)
    db_f  = davies_bouldin_score(X_cl, y_cl)
    ch_f  = calinski_harabasz_score(X_cl, y_cl)
    print(f"  Silhouette score         : {sil_f:.3f}   (higher → better)")
    print(f"  Davies-Bouldin index     : {db_f:.3f}   (lower  → better)")
    print(f"  Calinski-Harabasz index  : {ch_f:.1f}")
 
probs = clusterer.probabilities_
print(f"  Membership prob mean/med : {probs.mean():.3f} / {np.median(probs):.3f}")
print(f"  Noise fraction           : {noise_pct:.1f}%")

# %%
# =============================================================================
# SECTION 5 — COLOUR PALETTE
# =============================================================================
df_no_outliers = df_imp[df_imp["Cluster"] != -1].copy().reset_index(drop=True)
cluster_ids    = sorted(df_no_outliers["Cluster"].unique())
_cmap          = load_cmap("Tableau_10", cmap_type="discrete")
cluster_to_color     = {c: _cmap(i) for i, c in enumerate(cluster_ids)}
cluster_to_color_str = {str(int(k)): v for k, v in cluster_to_color.items()}
 
bcols_disp = [LABEL_MAP.get(c, c) for c in BEHAVIOR_COLS]
df_display = df_no_outliers.rename(columns=LABEL_MAP)
 
# Condition × sex combined palette (used across several sections)
cs_palette = {}
for cond in CONDITION_ORDER:
    for sx in ["M", "F"]:
        key  = f"{cond}-{sx}"
        base = mpl.colors.to_rgb(CONDITION_COLORS.get(cond, "#888888"))
        cs_palette[key] = tuple(x * (0.6 if sx == "M" else 1.0) for x in base)

# %%
# =============================================================================
# SECTION 6 — UMAP SCATTER
# =============================================================================
emb_valid = embedding[mask_valid]
 
fig, ax = plt.subplots(figsize=(10, 8))
sns.set_style("ticks")
sns.scatterplot(
    x=emb_valid[:, 0], y=emb_valid[:, 1],
    hue=df_no_outliers["Cluster"],
    style=df_no_outliers["condition"],
    palette=cluster_to_color,
    alpha=0.65, ax=ax,
)
ax.set_xlabel("UMAP 1", fontsize=18, weight="bold")
ax.set_ylabel("UMAP 2", fontsize=18, weight="bold")
ax.tick_params(labelsize=14)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["bottom", "left"]].set_linewidth(2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=12,
           frameon=False, markerscale=2, title="")
plt.tight_layout()
#plt.savefig("UMAP_scatter.pdf", bbox_inches="tight")
plt.show()

# %%
# =============================================================================
# SECTION 7 — BEHAVIORAL PROFILES
# =============================================================================
cluster_summary_mean = df_display.groupby("Cluster")[bcols_disp].mean()
 
# ── 7a. Radar / spider plot ───────────────────────────────────────────────────
N      = len(bcols_disp)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
 
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
for cluster in cluster_summary_mean.index:
    vals = cluster_summary_mean.loc[cluster].tolist() + [cluster_summary_mean.loc[cluster, bcols_disp[0]]]
    ax.plot(angles, vals, label=f"Cluster {cluster}",
            color=cluster_to_color[cluster], linewidth=2.5)
    ax.fill(angles, vals, color=cluster_to_color[cluster], alpha=0.08)
 
ax.set_xticks(angles[:-1])
ax.set_xticklabels(bcols_disp, size=10, color="black", weight="bold")
ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1.15), fontsize=13, frameon=False)
plt.tight_layout()
#plt.savefig("Radar_behavioral_profiles.pdf", bbox_inches="tight")
plt.show()
 
# ── 7b. Heatmap ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, max(4, len(cluster_ids) * 1.2)))
sns.heatmap(cluster_summary_mean, annot=True, fmt=".2f", cmap="viridis",
            ax=ax, annot_kws={"size": 11, "weight": "bold"})
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                   fontsize=13, weight="bold")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13, weight="bold")
ax.set_title("Cluster Behavioral Profiles", fontsize=16, weight="bold")
plt.tight_layout()
#plt.savefig("Heatmap_behavioral_profiles.pdf", bbox_inches="tight")
plt.show()
 

# %%
# =============================================================================
# SECTION 7 — BEHAVIORAL PROFILES
# =============================================================================
cluster_summary_mean = df_display.groupby("Cluster")[bcols_disp].mean()
 
# ── 7a. Radar / spider plot ───────────────────────────────────────────────────
N      = len(bcols_disp)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))

for cluster in cluster_summary_mean.index:
    vals = cluster_summary_mean.loc[cluster].tolist() + [cluster_summary_mean.loc[cluster, bcols_disp[0]]]
    ax.plot(angles, vals, label=f"Cluster {cluster}",
            color=cluster_to_color[cluster], linewidth=2.5)
    ax.fill(angles, vals, color=cluster_to_color[cluster], alpha=0.08)

# Labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(bcols_disp, size=14, color="black", weight="bold", rotation=45)
ax.tick_params(axis='x', pad=20)

# Scale
ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10, color="grey", weight="bold")

# Legend
ax.legend(loc="upper left", bbox_to_anchor=(1.00, 1.00), fontsize=13, frameon=False)

plt.tight_layout()
plt.show()
 
 

# %%
# =============================================================================
# SECTION 8 — CLUSTER COMPOSITION (chi-square + Cramér's V)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 8 — Cluster composition")
print("=" * 60)
 
for factor in ["condition", "sex"]:
    ct               = pd.crosstab(df_no_outliers["Cluster"], df_no_outliers[factor])
    chi2, p, dof, _  = chi2_contingency(ct)
    cv               = cramers_v(chi2, ct.values.sum(), min(ct.shape))
    print(f"\n  [{factor.upper()}]  χ²={chi2:.2f}  p={p:.4g}  Cramér's V={cv:.3f}")
    print(ct)
 
# Stacked bar — condition × sex proportion per cluster
df_no_outliers["condition_sex"] = df_no_outliers["condition"] + "-" + df_no_outliers["sex"]
counts_cs = (df_no_outliers
             .groupby(["Cluster", "condition_sex"])
             .size()
             .reset_index(name="Count"))
counts_cs["Total"]      = counts_cs.groupby("Cluster")["Count"].transform("sum")
counts_cs["Proportion"] = counts_cs["Count"] / counts_cs["Total"]
 
# Build ordered group list dynamically from the data
existing_cs = counts_cs["condition_sex"].unique()
desired_order = [f"{c}-{s}" for c in CONDITION_ORDER for s in ["M", "F"]]
cs_order = [g for g in desired_order if g in existing_cs]
 
plot_data_cs = (counts_cs
                .pivot(index="Cluster", columns="condition_sex", values="Proportion")
                .fillna(0)
                .reindex(columns=cs_order, fill_value=0))
 
# Auto-generate palette from condition + sex base colors
cs_palette = {}
for c in CONDITION_ORDER:
    for s in ["M", "F"]:
        key = f"{c}-{s}"
        base = mpl.colors.to_rgb(CONDITION_COLORS.get(c, "#888888"))
        cs_palette[key] = tuple(x * (0.6 if s == "M" else 1.0) for x in base)
 
cs_colors = [cs_palette.get(g, "#aaaaaa") for g in cs_order]
 
ax = plot_data_cs.plot(kind="bar", stacked=True, figsize=(10, 6), color=cs_colors)
handles = [Patch(facecolor=cs_palette.get(g, "#aaa"), label=g) for g in cs_order]
ax.legend(handles=handles, bbox_to_anchor=(1, 1), loc="upper left", fontsize=13, frameon=False)
ax.set_xlabel("Cluster", fontsize=16, weight="bold")
ax.set_ylabel("Proportion", fontsize=16, weight="bold")
ax.tick_params(axis="x", labelsize=14, rotation=0)
plt.yticks(fontsize=14)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
#plt.savefig("ClusterComposition_stacked.pdf", bbox_inches="tight")
plt.show()
 

# %%
# =============================================================================
# SECTION 8 — CLUSTER COMPOSITION - NEW
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 8 — Cluster composition (chi-square + Cramér's V)")
print("=" * 60)
 
for factor in ["condition", "sex"]:
    ct             = pd.crosstab(df_no_outliers["Cluster"], df_no_outliers[factor])
    chi2_v, p, *_  = chi2_contingency(ct)
    cv             = cramers_v(chi2_v, ct.values.sum(), min(ct.shape))
    print(f"\n  [{factor.upper()}]  χ²={chi2_v:.2f}  p={p:.4g}  Cramér's V={cv:.3f}")
    print(ct)
 
df_no_outliers["condition_sex"] = (df_no_outliers["condition"] + "-"
                                   + df_no_outliers["sex"])
counts_cs = (df_no_outliers.groupby(["Cluster", "condition_sex"])
             .size().reset_index(name="Count"))
counts_cs["Total"]      = counts_cs.groupby("Cluster")["Count"].transform("sum")
counts_cs["Proportion"] = counts_cs["Count"] / counts_cs["Total"]
 
existing_cs = counts_cs["condition_sex"].unique()
desired_order = [f"{c}-{s}" for c in CONDITION_ORDER for s in ["M", "F"]]
cs_order = [g for g in desired_order if g in existing_cs]
 
plot_cs = (counts_cs
           .pivot(index="Cluster", columns="condition_sex", values="Proportion")
           .fillna(0).reindex(columns=cs_order, fill_value=0))
 
cs_colors = [cs_palette.get(g, "#aaaaaa") for g in cs_order]
ax = plot_cs.plot(kind="bar", stacked=True, figsize=(10, 6), color=cs_colors)
handles = [Patch(facecolor=cs_palette.get(g, "#aaa"), label=g) for g in cs_order]
ax.legend(handles=handles, bbox_to_anchor=(1, 1), loc="upper left",
          fontsize=13, frameon=False)
ax.set_xlabel("Cluster", fontsize=16, weight="bold")
ax.set_ylabel("Proportion", fontsize=16, weight="bold")
ax.tick_params(axis="x", labelsize=14, rotation=0)
plt.yticks(fontsize=14)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
#plt.savefig("ClusterComposition_stacked.pdf", bbox_inches="tight")
plt.show()

# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.mnlogit("Cluster ~ condition * sex", data=df_no_outliers)
result = model.fit()
print(result.summary())

# %%
np.exp(result.params)

# %%
pred_probs = result.predict(df_no_outliers)
pred_df = pred_probs.copy()
pred_df["condition"] = df_no_outliers["condition"].values
pred_df["sex"] = df_no_outliers["sex"].values
pred_mean = (
    pred_df
    .groupby(["condition", "sex"])
    .mean()
    .reset_index()
)

pred_long = pred_mean.melt(
    id_vars=["condition", "sex"],
    var_name="Cluster",
    value_name="Probability"
)

pred_long["Cluster"] = pred_long["Cluster"].astype(int)

plot_df = (
    pred_long
    .pivot_table(index=["condition", "sex"], columns="Cluster", values="Probability")
)

plot_df.plot(kind="bar", stacked=True, figsize=(10, 6), color=[cluster_to_color_str.get(str(c), "#aaa") for c in plot_df.columns])
plt.ylabel("Predicted probability")
plt.xlabel("Condition / Sex")
plt.xticks(rotation=0)
plt.legend(title="Cluster", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()

# %%
diff = pred_mean.copy()
diff = diff.set_index(["condition", "sex"])

diff_plot = diff.loc["PNELS"] - diff.loc["ctrl"]
diff_plot.T.plot(kind="bar", figsize=(8,5))

plt.axhline(0, linestyle="--")
plt.ylabel("Δ Probability (PNELS - Control)")
plt.title("Effect of PNELS on cluster occupancy")
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# SECTION 9 — KRUSKAL-WALLIS + DUNN POST-HOC
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 9 — Kruskal-Wallis + Dunn post-hoc per behavioral feature")
print("=" * 60)
 
kw_results = []
for col in BEHAVIOR_COLS:
    groups_kw = [g[col].dropna().values
                 for _, g in df_no_outliers.groupby("Cluster") if _ != -1]
    if len(groups_kw) < 2:
        continue
    H, p = kruskal(*groups_kw)
    n     = sum(len(g) for g in groups_kw)
    k     = len(groups_kw)
    eta2  = (H - k + 1) / (n - k)
    kw_results.append({"feature": col, "H": H, "p_raw": p, "eta2": eta2})
 
kw_df = pd.DataFrame(kw_results)
_, p_fdr, _, _ = multipletests(kw_df["p_raw"], alpha=0.05, method="fdr_bh")
kw_df["p_fdr"]       = p_fdr
kw_df["sig_fdr"]     = p_fdr < 0.05
kw_df["effect_size"] = pd.cut(kw_df["eta2"],
                               bins=[-np.inf, 0.01, 0.06, 0.14, np.inf],
                               labels=["negligible", "small", "medium", "large"])
 
print(kw_df[["feature", "H", "p_fdr", "eta2", "effect_size",
             "sig_fdr"]].sort_values("p_fdr").to_string(index=False))
 
# Dunn pairwise for significant features
print("\n  Pairwise Dunn tests (FDR-BH) for significant features:")
dunn_results_all = {}
for col in kw_df.loc[kw_df["sig_fdr"], "feature"]:
    dunn_mat = sp.posthoc_dunn(df_no_outliers, val_col=col,
                               group_col="Cluster", p_adjust="fdr_bh")
    dunn_results_all[col] = dunn_mat
    print(f"\n  {col}:\n{dunn_mat.round(3)}")


# %%
# =============================================================================
# SECTION 9 — KRUSKAL-WALLIS + DUNN POST-HOC - NEW
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 9 — Kruskal-Wallis + Dunn post-hoc")
print("=" * 60)
 
kw_results = []
for col in BEHAVIOR_COLS:
    grps = [g[col].dropna().values
            for _, g in df_no_outliers.groupby("Cluster")]
    if len(grps) < 2:
        continue
    H, p  = kruskal(*grps)
    n_tot = sum(len(g) for g in grps)
    k_tot = len(grps)
    eta2  = (H - k_tot + 1) / (n_tot - k_tot)
    kw_results.append({"feature": col, "H": H, "p_raw": p, "eta2": eta2})
 
kw_df = pd.DataFrame(kw_results)
_, p_fdr, _, _ = multipletests(kw_df["p_raw"], alpha=0.05, method="fdr_bh")
kw_df["p_fdr"]       = p_fdr
kw_df["sig_fdr"]     = p_fdr < 0.05
kw_df["effect_size"] = pd.cut(kw_df["eta2"],
                               bins=[-np.inf, 0.01, 0.06, 0.14, np.inf],
                               labels=["negligible", "small", "medium", "large"])
print(kw_df[["feature","H","p_fdr","eta2","effect_size","sig_fdr"]]
      .sort_values("p_fdr").to_string(index=False))
 
dunn_results_all = {}
for col in kw_df.loc[kw_df["sig_fdr"], "feature"]:
    dunn_results_all[col] = sp.posthoc_dunn(
        df_no_outliers, val_col=col, group_col="Cluster", p_adjust="fdr_bh")

# %%
# =============================================================================
# SECTION 10 — VIOLIN PLOTS WITH SIGNIFICANCE BRACKETS
# =============================================================================
ncols_vio = 4
nvars     = len(BEHAVIOR_COLS)
nrows_vio = math.ceil(nvars / ncols_vio)
 
fig, axs = plt.subplots(nrows_vio, ncols_vio,
                        figsize=(5 * ncols_vio, 5 * nrows_vio), sharey=False)
axs = axs.flatten()
 
for idx, col in enumerate(BEHAVIOR_COLS):
    ax = axs[idx]
    sns.violinplot(x="Cluster", y=col, data=df_no_outliers,
                   inner="box", palette=cluster_to_color_str, ax=ax)
    ax.set_title(LABEL_MAP.get(col, col), fontsize=11, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("mean / 2 s" if idx % ncols_vio == 0 else "")
    ax.tick_params(labelsize=11)
 
    if col in dunn_results_all:
        dm       = dunn_results_all[col]
        sig_pairs = [(i, j) for i in range(len(cluster_ids))
                     for j in range(len(cluster_ids))
                     if i < j and dm.loc[cluster_ids[i], cluster_ids[j]] < 0.05]
        add_sig_brackets(ax, sig_pairs, df_no_outliers[col].max())
 
for i in range(nvars, nrows_vio * ncols_vio):
    fig.delaxes(axs[i])
 
plt.tight_layout()
#plt.savefig("Violins_per_behavior.pdf", bbox_inches="tight")
plt.show()
 

# %%
# =============================================================================
# SECTION 11 — PERMANOVA (condition & sex within each cluster)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 11 — PERMANOVA per cluster")
print("=" * 60)
 
perm_rows = []
for cid in cluster_ids:
    sub = df_no_outliers[df_no_outliers["Cluster"] == cid].copy()
    sub.index = sub.index.astype(str)
    X_perm = sub[BEHAVIOR_COLS].values
    dm_perm = DistanceMatrix(squareform(pdist(X_perm, "euclidean")),
                             ids=sub.index.tolist())
    for factor in ["condition", "sex"]:
        if sub[factor].nunique() < 2:
            continue
        grouping = sub.loc[list(dm_perm.ids), factor]
        res      = permanova(dm_perm, grouping, permutations=999)
        perm_rows.append({
            "Cluster": cid, "Factor": factor,
            "pseudo-F": res["test statistic"], "p-value": res["p-value"],
            "n": len(sub),
        })
 
perm_df = pd.DataFrame(perm_rows)
if len(perm_df) > 0:
    _, p_corr, _, _ = multipletests(perm_df["p-value"], method="fdr_bh")
    perm_df["p_fdr"]       = p_corr
    perm_df["significant"] = p_corr < 0.05
    print(perm_df.to_string(index=False))

# %%
# =============================================================================
# SECTION 11 — PERMANOVA - NEW
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 11 — PERMANOVA per cluster")
print("=" * 60)
 
perm_rows = []
for cid in cluster_ids:
    sub = df_no_outliers[df_no_outliers["Cluster"] == cid].copy()
    sub.index = sub.index.astype(str)
    dm_p = DistanceMatrix(squareform(pdist(sub[BEHAVIOR_COLS].values, "euclidean")),
                          ids=sub.index.tolist())
    for factor in ["condition", "sex"]:
        if sub[factor].nunique() < 2:
            continue
        res = permanova(dm_p, sub.loc[list(dm_p.ids), factor], permutations=999)
        perm_rows.append({"Cluster": cid, "Factor": factor,
                          "pseudo-F": res["test statistic"],
                          "p-value":  res["p-value"], "n": len(sub)})
 
perm_df = pd.DataFrame(perm_rows)
if len(perm_df) > 0:
    _, p_corr, _, _ = multipletests(perm_df["p-value"], method="fdr_bh")
    perm_df["p_fdr"]       = p_corr
    perm_df["significant"] = p_corr < 0.05
    print(perm_df.to_string(index=False))

# %%
# =============================================================================
# SECTION 12 — TEMPORAL DYNAMICS (polar plots)
# =============================================================================
# Proportion of each cluster over time (30-second bins)
 
time_counts = (
    df_no_outliers.groupby(["Interval_bin", "Cluster"])
    .size().reset_index(name="count")
)
time_total  = (
    df_no_outliers.groupby("Interval_bin")
    .size().reset_index(name="total")
)
time_prop = time_counts.merge(time_total, on="Interval_bin")
time_prop["cluster_prop"] = time_prop["count"] / time_prop["total"]
 
# Bin into 30-second windows
time_prop["bin_30s"] = (time_prop["Interval_bin"] // 15) * 30
df_agg30 = (time_prop.groupby(["bin_30s", "Cluster"], as_index=False)
            ["cluster_prop"].mean())
 
intervals_30 = sorted(df_agg30["bin_30s"].unique())
n_int30      = len(intervals_30)
theta30      = 2 * np.pi * np.arange(n_int30) / n_int30
max_t_s      = intervals_30[-1] + 30
 
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(12, 12))
for cid in cluster_ids:
    y = np.zeros(n_int30)
    for _, row in df_agg30[df_agg30["Cluster"] == cid].iterrows():
        bi = intervals_30.index(row["bin_30s"])
        y[bi] = row["cluster_prop"]
    tc = np.append(theta30, theta30[0])
    yc = np.append(y, y[0])
    ax.plot(tc, yc, color=cluster_to_color[cid], lw=3,
            label=f"Cluster {cid}", alpha=0.9)
    ax.fill(tc, yc, color=cluster_to_color[cid], alpha=0.12)
 
major_min = np.arange(0, min(10, max_t_s // 60 + 1), 2)
tick_locs  = 2 * np.pi * (major_min * 60 / max_t_s)
ax.set_xticks(tick_locs)
ax.set_xticklabels([f"{m} min" for m in major_min], fontsize=20, weight="bold")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.tick_params(axis="y", labelsize=14, labelcolor="grey")
ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), fontsize=16, frameon=False)
ax.set_title("Cluster Representation Across Time", fontsize=22, weight="bold", va="bottom")
plt.tight_layout()
#plt.savefig("Polar_temporal_dynamics.pdf", bbox_inches="tight")
plt.show()

# %%
# =============================================================================
# SECTION 13 — TRANSITION ANALYSIS
#   13a  Counts + Markov matrices
#   13b  Transition counts per animal (boxplot + ANOVA)
#   13c  Transition heatmaps (condition × sex, 2-D grid)
#   13d  NETWORK TRANSITION GRAPHS  ← RESTORED
# =============================================================================
df_tsrc = df_no_outliers.sort_values(["experimental_id", "Interval_bin"])
 
all_trans = []
for exp_id, sub in df_tsrc.groupby("experimental_id"):
    cseq = sub["Cluster"].tolist()
    cond = sub["condition"].iloc[0]
    sx   = sub["sex"].iloc[0]
    all_trans += [(exp_id, cond, sx, fr, to)
                  for fr, to in zip(cseq[:-1], cseq[1:])]
 
transitions_df = pd.DataFrame(all_trans,
    columns=["experimental_id", "condition", "sex", "from_cluster", "to_cluster"])
transitions_df["condition_sex"] = (transitions_df["condition"] + "-"
                                   + transitions_df["sex"])
 
# Counts by condition × sex × from × to
tc_all = (transitions_df
          .groupby(["condition", "sex", "condition_sex",
                    "from_cluster", "to_cluster"])
          .size().reset_index(name="count"))
# ── Markov transition matrices ────────────────────────────────────────────────
clusters_tr = sorted(
    set(transitions_df["from_cluster"]) | set(transitions_df["to_cluster"]))
c2i = {c: i for i, c in enumerate(clusters_tr)}
 
transition_matrices = {}
for (cond, sx), grp in transitions_df.groupby(["condition", "sex"]):
    n = len(clusters_tr)
    cnt = np.zeros((n, n), dtype=int)
    for _, row in grp.iterrows():
        cnt[c2i[row["from_cluster"]], c2i[row["to_cluster"]]] += 1
    rs = cnt.sum(axis=1, keepdims=True)
    prob = np.divide(cnt, rs, out=np.zeros_like(cnt, dtype=float), where=rs > 0)
    transition_matrices[(cond, sx)] = pd.DataFrame(
        prob, index=clusters_tr, columns=clusters_tr)

# %%
# ── 13d. Plot Markov heatmaps (2 × n_conditions layout) ─────────────────────
sexes      = sorted(transitions_df["sex"].unique())
conds_plot = [c for c in CONDITION_ORDER
              if c in transitions_df["condition"].unique()]
 
all_prob_vals = np.concatenate(
    [m.values.flatten() for m in transition_matrices.values()])
vmin_m, vmax_m = 0, np.nanmax(all_prob_vals)
 
for sex in sexes:
    ncols_m = len(conds_plot)
    fig, axes_m = plt.subplots(1, ncols_m,
                               figsize=(6 * ncols_m, 5), sharey=True)
    if ncols_m == 1:
        axes_m = [axes_m]
    for ax_m, cond in zip(axes_m, conds_plot):
        key = (cond, sex)
        if key not in transition_matrices:
            ax_m.axis("off")
            continue
        mat  = transition_matrices[key]
        base = mpl.colors.to_rgb(CONDITION_COLORS.get(cond, "#888888"))
        cmap_m = LinearSegmentedColormap.from_list(
            "custom", ["white", mpl.colors.to_hex(base)])
        sns.heatmap(mat, annot=True, fmt=".2f", cmap=cmap_m,
                    vmin=vmin_m, vmax=vmax_m, ax=ax_m,
                    annot_kws={"size": 12, "weight": "bold"})
        ax_m.set_title(f"{cond} – {sex}", fontsize=14, weight="bold")
        ax_m.set_xlabel("To Cluster", fontsize=12, weight="bold")
        ax_m.set_ylabel("From Cluster", fontsize=12, weight="bold")
    plt.suptitle(f"Transition Probability Matrices — Sex: {sex}",
                 fontsize=16, weight="bold", y=1.02)
    plt.tight_layout()
    #plt.savefig(f"Markov_transition_{sex}.pdf", bbox_inches="tight")
    plt.show()
 

# %%
# ── 13b Transitions per animal (boxplot + 2-way ANOVA) ───────────────────────
trans_per_animal = (
    transitions_df.groupby(["experimental_id", "sex", "condition"])
    ["from_cluster"].count().reset_index(name="count"))
trans_per_animal["Group"] = (trans_per_animal["condition"] + "-"
                              + trans_per_animal["sex"])
 
#g_vals   = sorted(trans_per_animal["Group"].unique())
g_vals = [
    "ctrl-M",
    "PNELS-M",
    "ctrl-F",
    "PNELS-F"
]
pal_trans = {g: cs_palette.get(g, "#888888") for g in g_vals}
 
fig, ax = plt.subplots(figsize=(max(8, len(g_vals) * 1.5), 6))
sns.boxplot(x="Group", y="count", data=trans_per_animal,
            order=g_vals, palette=pal_trans, ax=ax)
sns.stripplot(x="Group", y="count", data=trans_per_animal,
              order=g_vals, color="#1a1a2e", size=5, jitter=True,
              alpha=0.6, ax=ax)
ax.set_ylabel("Transitions per animal", fontsize=16, weight="bold")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, weight="bold",
                   rotation=30, ha="right")
ax.tick_params(axis="y", labelsize=13)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_linewidth(2)
plt.tight_layout()
#plt.savefig("Transitions_per_animal.pdf", bbox_inches="tight")
plt.show()
 
print("\n  Two-way ANOVA — transition counts per animal:")
model_trans = ols("count ~ C(condition) * C(sex)", data=trans_per_animal).fit()
print(sm.stats.anova_lm(model_trans, typ=2))

# %%
# ── 13c Transition probability heatmaps ──────────────────────────────────────
all_prob_vals = np.concatenate([m.values.flatten()
                                for m in transition_matrices.values()])
vmin_m, vmax_m = 0, np.nanmax(all_prob_vals)
 
sexes_plot = sorted(transitions_df["sex"].unique())
conds_plot = [c for c in CONDITION_ORDER
              if c in transitions_df["condition"].unique()]
 
for sx in sexes_plot:
    ncols_m = len(conds_plot)
    fig, axes_m = plt.subplots(1, ncols_m,
                               figsize=(6 * ncols_m, 5), sharey=True)
    if ncols_m == 1:
        axes_m = [axes_m]
    for ax_m, cond in zip(axes_m, conds_plot):
        key = (cond, sx)
        if key not in transition_matrices:
            ax_m.axis("off"); continue
        mat  = transition_matrices[key]
        base = mpl.colors.to_rgb(CONDITION_COLORS.get(cond, "#888888"))
        cmap_m = LinearSegmentedColormap.from_list(
            "c", ["white", mpl.colors.to_hex(base)])
        sns.heatmap(mat, annot=True, fmt=".2f", cmap=cmap_m,
                    vmin=vmin_m, vmax=vmax_m, ax=ax_m,
                    annot_kws={"size": 14, "weight": "bold"})
        ax_m.set_title(f"{cond} — {sx}", fontsize=14, weight="bold")
        ax_m.set_xlabel("To cluster", fontsize=12, weight="bold")
        ax_m.set_ylabel("From cluster" if cond == conds_plot[0] else "",
                        fontsize=12, weight="bold")
        ax_m.set_xticklabels(ax_m.get_xticklabels(),
                              fontsize=12, weight="bold", rotation=0)
        ax_m.set_yticklabels(ax_m.get_yticklabels(),
                              fontsize=12, weight="bold", rotation=0)
    plt.suptitle(f"Transition Probability Matrices — {sx}",
                 fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    #plt.savefig(f"Markov_transition_{sx}.pdf", bbox_inches="tight")
    plt.show()

# %%
# ── 13d NETWORK TRANSITION GRAPHS ─────────────────────────────────────────────
# One directed graph per (condition, sex) group.
# Node size   ∝ total in+out transition weight (globally normalised)
# Edge width  ∝ transition count (globally normalised, power-scaled for contrast)
# Node colour = cluster colour (Tableau_10 palette)
# Self-loops are hidden (stay-in-cluster transitions add noise visually)
# All groups share the same node-size and edge-width scale for comparability.
# Layout seed is fixed so the same clusters land in the same positions across
# panels, making cross-group comparison easier.
 
print("\n  Drawing network transition graphs …")
 
# ── Pre-compute global normalisation ranges ───────────────────────────────────
_all_edge_w   = tc_all["count"].values.astype(float)
_global_e_min = _all_edge_w.min()
_global_e_max = _all_edge_w.max()
 
_all_node_inc = []
for (cond, sx) in transition_matrices:
    sub_tc = tc_all[(tc_all["condition"] == cond) & (tc_all["sex"] == sx)]
    G_tmp  = nx.DiGraph()
    for _, row in sub_tc.iterrows():
        G_tmp.add_edge(row["from_cluster"], row["to_cluster"],
                       weight=row["count"])
    for node in G_tmp.nodes():
        in_w  = sum(d["weight"] for _, _, d in G_tmp.in_edges(node,  data=True))
        out_w = sum(d["weight"] for _, _, d in G_tmp.out_edges(node, data=True))
        _all_node_inc.append(in_w + out_w)
 
_global_n_min = min(_all_node_inc)
_global_n_max = max(_all_node_inc)
 
NODE_SZ_MIN, NODE_SZ_MAX = 400,  3000
EDGE_W_MIN,  EDGE_W_MAX  = 1,    40
EDGE_POWER               = 1.5   # >1 increases contrast between thin/thick edges
 
def _norm(val, lo, hi, vmin, vmax):
    if vmax == vmin:
        return (lo + hi) / 2
    return lo + (hi - lo) * (val - vmin) / (vmax - vmin)
 
def _norm_arr(arr, lo, hi, vmin, vmax):
    if vmax == vmin:
        return np.full(len(arr), (lo + hi) / 2)
    return lo + (hi - lo) * ((arr - vmin) / (vmax - vmin)) ** EDGE_POWER
 
# ── Helper: draw one network panel ───────────────────────────────────────────
def draw_network(ax, G, pos, node_inc, title):
    nodes   = list(G.nodes())
    n_cols  = [cluster_to_color.get(n, "#aaaaaa") for n in nodes]
    n_sizes = [_norm(node_inc.get(n, 0),
                     NODE_SZ_MIN, NODE_SZ_MAX,
                     _global_n_min, _global_n_max) for n in nodes]
 
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=n_cols, node_size=n_sizes, alpha=0.92)
 
    edges_no_sl = [(u, v) for u, v in G.edges() if u != v]
    if edges_no_sl:
        ew_raw = np.array([G[u][v]["weight"] for u, v in edges_no_sl], dtype=float)
        ew     = _norm_arr(ew_raw, EDGE_W_MIN, EDGE_W_MAX,
                           _global_e_min, _global_e_max)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges_no_sl, width=ew,
            alpha=0.72, arrows=True,
            arrowstyle="-|>", arrowsize=20,
            connectionstyle="arc3,rad=0.12",  # slight arc avoids overlap
        )
        elabels = {(u, v): G[u][v]["weight"] for u, v in edges_no_sl}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=elabels, ax=ax, font_size=10)
 
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=17, font_weight="bold")
    ax.set_title(title, fontsize=16, weight="bold")
    ax.axis("off")
 
 
# ── Compute a single shared spring-layout using the FULL transition graph ─────
G_full = nx.DiGraph()
for _, row in tc_all.iterrows():
    if row["from_cluster"] != row["to_cluster"]:
        w_existing = G_full[row["from_cluster"]][row["to_cluster"]]["weight"] \
            if G_full.has_edge(row["from_cluster"], row["to_cluster"]) else 0
        G_full.add_edge(row["from_cluster"], row["to_cluster"],
                        weight=w_existing + row["count"])
 
shared_pos = nx.spring_layout(G_full, seed=RANDOM_STATE, weight="weight")
 
# ── Plot: 2 rows (sexes) × n_conditions columns ──────────────────────────────
ncols_net = len(conds_plot)
nrows_net = len(sexes_plot)
fig, axes_net = plt.subplots(nrows_net, ncols_net,
                              figsize=(7 * ncols_net, 7 * nrows_net),
                              squeeze=False)
 
for r_idx, sx in enumerate(sexes_plot):
    for c_idx, cond in enumerate(conds_plot):
        ax_net = axes_net[r_idx][c_idx]
        key    = (cond, sx)
 
        if key not in transition_matrices:
            ax_net.axis("off")
            continue
 
        sub_tc = tc_all[(tc_all["condition"] == cond) & (tc_all["sex"] == sx)]
        G = nx.DiGraph()
        for _, row in sub_tc.iterrows():
            G.add_edge(row["from_cluster"], row["to_cluster"],
                       weight=row["count"])
 
        node_inc = {}
        for node in G.nodes():
            in_w  = sum(d["weight"] for _, _, d in G.in_edges(node,  data=True))
            out_w = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
            node_inc[node] = in_w + out_w
 
        # Use shared positions; add any missing nodes at centroid
        pos_use = {n: shared_pos.get(n, np.array([0.0, 0.0]))
                   for n in G.nodes()}
 
        draw_network(ax_net, G, pos_use, node_inc,
                     title=f"{cond} — {sx}")
 
plt.suptitle("Cluster Transition Networks\n"
             "(node size ∝ total transitions · edge width ∝ transition count)",
             fontsize=16, weight="bold", y=1.01)
plt.tight_layout()
plt.savefig("TransitionNetworks_grid.pdf", bbox_inches="tight")
plt.show()
 

# %%
# ── Also save one figure per sex (larger, cleaner for reports) ────────────────
for sx in sexes_plot:
    fig_s, axes_s = plt.subplots(1, ncols_net,
                                  figsize=(7 * ncols_net, 7), squeeze=False)
    for c_idx, cond in enumerate(conds_plot):
        ax_s = axes_s[0][c_idx]
        key  = (cond, sx)
        if key not in transition_matrices:
            ax_s.axis("off"); continue
        sub_tc = tc_all[(tc_all["condition"] == cond) & (tc_all["sex"] == sx)]
        G_s    = nx.DiGraph()
        for _, row in sub_tc.iterrows():
            G_s.add_edge(row["from_cluster"], row["to_cluster"],
                         weight=row["count"])
        ni_s = {}
        for node in G_s.nodes():
            ni_s[node] = (sum(d["weight"] for _, _, d in G_s.in_edges(node,  data=True))
                        + sum(d["weight"] for _, _, d in G_s.out_edges(node, data=True)))
        draw_network(ax_s, G_s, {n: shared_pos.get(n, np.array([0.0, 0.0]))
                                  for n in G_s.nodes()}, ni_s,
                     title=f"$\\bf{{{cond}}}$")
    fig_s.suptitle(f"Transition Networks — {sx}",
                   fontsize=18, weight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"TransitionNetworks_{sx}.pdf", bbox_inches="tight")
    plt.show()

# %%
# ── 13d NETWORK TRANSITION GRAPHS ─────────────────────────────────────────────
# Style: one standalone figure per (condition, sex) group, matching the
# original poster-quality look:
#   • node size  ∝ total in+out weight  — normalised GLOBALLY so panels are
#     directly comparable across groups
#   • edge width ∝ transition count     — also globally normalised
#   • straight arrows  (-|>),  no arcs
#   • raw count shown as edge label
#   • node colour = Tableau_10 cluster palette
#   • self-loops hidden
#   • fixed spring_layout seed (per-group, same seed → reproducible)
#
# Additionally a 2-row × n-condition summary grid is saved for quick overview.
# =============================================================================
 
print("\n  Drawing network transition graphs …")
 
# ── Global normalisation ranges (so all groups share the same scale) ──────────
_all_edge_w   = tc_all["count"].values.astype(float)
_global_e_min = _all_edge_w.min()
_global_e_max = _all_edge_w.max()
 
_all_node_inc = []
for (cond, sx) in transition_matrices:
    sub_tc = tc_all[(tc_all["condition"] == cond) & (tc_all["sex"] == sx)]
    G_tmp  = nx.DiGraph()
    for _, row in sub_tc.iterrows():
        G_tmp.add_edge(row["from_cluster"], row["to_cluster"],
                       weight=row["count"])
    for node in G_tmp.nodes():
        inc = (sum(d["weight"] for _, _, d in G_tmp.in_edges(node, data=True)) +
               sum(d["weight"] for _, _, d in G_tmp.out_edges(node, data=True)))
        _all_node_inc.append(inc)
 
_global_n_min = min(_all_node_inc)
_global_n_max = max(_all_node_inc)
 
# Size / width bounds
NODE_SZ_MIN, NODE_SZ_MAX = 500,  3000 # 500, 3000
EDGE_W_MIN,  EDGE_W_MAX  = 1,    10 # 1, 16
 
 
def _linscale(val, lo, hi, vmin, vmax):
    """Linear scale a single value to [lo, hi]."""
    if vmax == vmin:
        return (lo + hi) / 2
    return lo + (hi - lo) * (val - vmin) / (vmax - vmin)
 
 
def _linscale_arr(arr, lo, hi, vmin, vmax):
    """Linear scale an array to [lo, hi]."""
    if vmax == vmin:
        return np.full(len(arr), (lo + hi) / 2, dtype=float)
    return lo + (hi - lo) * (arr - vmin) / (vmax - vmin)
 
 
# ── One standalone figure per group ──────────────────────────────────────────
group_keys = [(cond, sx)
              for sx in sexes_plot
              for cond in conds_plot
              if (cond, sx) in transition_matrices]
 
for cond, sx in group_keys:
    sub_tc = tc_all[(tc_all["condition"] == cond) & (tc_all["sex"] == sx)]
 
    # Build directed graph
    G = nx.DiGraph()
 
    for _, row in sub_tc.iterrows():
        G.add_edge(row["from_cluster"], row["to_cluster"],
                   weight=row["count"])
 
    # Node incidence (in + out weights)
    node_incidence = {}
    for node in G.nodes():
        node_incidence[node] = (
            sum(d["weight"] for _, _, d in G.in_edges(node,  data=True)) +
            sum(d["weight"] for _, _, d in G.out_edges(node, data=True)))
 
    # Node colours and globally-scaled sizes
    node_colors = [cluster_to_color.get(n, "#aaaaaa") for n in G.nodes()]
    incidences  = np.array([node_incidence[n] for n in G.nodes()])
 
    scaled_sizes = _linscale_arr(incidences,
                                  NODE_SZ_MIN, NODE_SZ_MAX,
                                  _global_n_min, _global_n_max)
 
    # ── EDGES (RAW VALUES + GLOBAL ROBUST SCALING) ──────────────────────────
    edges_no_sl = [(u, v) for u, v in G.edges() if u != v]
 
    if edges_no_sl:
        ew_raw = np.array(
            [G[u][v]["weight"] for u, v in edges_no_sl],
            dtype=float
        )
 
        edge_widths = _linscale_arr(
            ew_raw,
            EDGE_W_MIN, EDGE_W_MAX,
            _global_e_min, _global_e_max
        )
    else:
        edge_widths = []
 
    # Layout
    pos = nx.spring_layout(G, seed=RANDOM_STATE, weight="weight")
 
    # ── DRAW ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
 
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=scaled_sizes,
        alpha=0.9
    )
 
    if edges_no_sl:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges_no_sl,
            width=edge_widths,
            alpha=0.7,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=22
        )
 
        edge_labels = {(u, v): G[u][v]["weight"] for u, v in edges_no_sl}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=11
        )
 
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=16,
        font_weight="bold"
    )
 
    ax.set_title(f"Cluster Transition Network\n{cond} — {sx}",
                 fontsize=18, weight="bold")
 
    ax.axis("off")
    plt.tight_layout()
 
    fname = f"TransitionNetwork_{cond}_{sx}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.show()
 
    print(f"    Saved → {fname}")
 
 
# ── SUMMARY GRID ─────────────────────────────────────────────────────────────
ncols_net = len(conds_plot)
nrows_net = len(sexes_plot)
 
fig_grid, axes_grid = plt.subplots(
    nrows_net, ncols_net,
    figsize=(8 * ncols_net, 8 * nrows_net),
    squeeze=False
)
 
for r_idx, sx in enumerate(sexes_plot):
    for c_idx, cond in enumerate(conds_plot):
 
        ax_g = axes_grid[r_idx][c_idx]
        key = (cond, sx)
 
        if key not in transition_matrices:
            ax_g.axis("off")
            continue
 
        sub_tc = tc_all[(tc_all["condition"] == cond) &
                        (tc_all["sex"] == sx)]
 
        G_g = nx.DiGraph()
 
        for _, row in sub_tc.iterrows():
            G_g.add_edge(row["from_cluster"], row["to_cluster"],
                         weight=row["count"])
 
        # node incidence
        ni_g = {
            node: (
                sum(d["weight"] for _, _, d in G_g.in_edges(node, data=True)) +
                sum(d["weight"] for _, _, d in G_g.out_edges(node, data=True))
            )
            for node in G_g.nodes()
        }
 
        nc_g = [cluster_to_color.get(n, "#aaaaaa") for n in G_g.nodes()]
        inc_g = np.array([ni_g[n] for n in G_g.nodes()])
 
        sz_g = _linscale_arr(
            inc_g,
            NODE_SZ_MIN, NODE_SZ_MAX,
            _global_n_min, _global_n_max
        )
 
        esl_g = [(u, v) for u, v in G_g.edges() if u != v]
 
        pos_g = nx.spring_layout(G_g, seed=RANDOM_STATE, weight="weight")
 
        nx.draw_networkx_nodes(
            G_g, pos_g, ax=ax_g,
            node_color=nc_g,
            node_size=sz_g,
            alpha=0.9
        )
 
        if esl_g:
            ew_g_raw = np.array(
                [G_g[u][v]["weight"] for u, v in esl_g],
                dtype=float
            )
 
            ew_gs = _linscale_arr(
                ew_g_raw,
                EDGE_W_MIN, EDGE_W_MAX,
                _global_e_min, _global_e_max
            )
 
            nx.draw_networkx_edges(
                G_g, pos_g, ax=ax_g,
                edgelist=esl_g,
                width=ew_gs,
                alpha=0.7,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=22
            )
 
            nx.draw_networkx_edge_labels(
                G_g, pos_g,
                edge_labels={(u, v): G_g[u][v]["weight"]
                             for u, v in esl_g},
                ax=ax_g,
                font_size=10
            )
 
        nx.draw_networkx_labels(
            G_g, pos_g, ax=ax_g,
            font_size=16,
            font_weight="bold"
        )
 
        ax_g.set_title(f"$\\bf{{{cond}}}$ — {sx}",
                       fontsize=16, weight="bold")
 
        ax_g.axis("off")
 
fig_grid.suptitle(
    "Cluster Transition Networks\n"
    "(node size - transitions · edge width - count (globally normalized))",
    fontsize=16,
    weight="bold",
    y=1.01
)
 
plt.tight_layout()
plt.savefig("TransitionNetworks_grid.pdf", bbox_inches="tight")
plt.show()

# %%
# =============================================================================
# SECTION 14 — BEHAVIORAL ENTROPY
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 14 — Behavioral entropy")
print("=" * 60)
 
cluster_counts_ent = (
    df_no_outliers
    .groupby(["experimental_id", "sex", "condition", "Cluster"])
    .size().reset_index(name="count")
)
state_entropy = (
    cluster_counts_ent
    .groupby(["experimental_id", "sex", "condition"])
    .apply(lambda x: shannon_entropy(x["count"]))
    .reset_index(name="entropy")
)
state_entropy["Group"] = state_entropy["condition"] + "-" + state_entropy["sex"]
 
#ent_groups = sorted(state_entropy["Group"].unique())
ent_groups = [
    "ctrl-M",
    "PNELS-M",
    "ctrl-F",
    "PNELS-F"
]
pal_ent    = {g: pal_trans.get(g, "#888888") for g in ent_groups}
 
H_ent, p_ent = kruskal(
    *[state_entropy.loc[state_entropy["Group"] == g, "entropy"] for g in ent_groups])
print(f"  Kruskal-Wallis H={H_ent:.3f}  p={p_ent:.4g}")
 
dunn_ent = sp.posthoc_dunn(
    state_entropy, val_col="entropy", group_col="Group", p_adjust="fdr_bh")
dunn_ent = dunn_ent.reindex(index=ent_groups, columns=ent_groups)
print("\n  Dunn post-hoc (FDR-BH):")
print(dunn_ent.round(3))
 
fig, ax = plt.subplots(figsize=(max(8, len(ent_groups) * 1.5), 6))
sns.barplot(data=state_entropy, x="Group", y="entropy", order=ent_groups,
            palette=pal_ent, estimator="mean", errorbar="se",
            edgecolor="black", alpha=0.85, ax=ax)
sns.stripplot(data=state_entropy, x="Group", y="entropy", order=ent_groups,
              color="black", jitter=0.25, size=5, alpha=0.6, ax=ax)
 
# Significance brackets
y_ent  = state_entropy["entropy"].max()
h_ent  = y_ent * 0.05
for i in range(len(ent_groups)):
    for j in range(i + 1, len(ent_groups)):
        p_val = dunn_ent.iloc[i, j]
        sym   = p_to_symbol(p_val)
        if sym != "ns":
            y_ann = y_ent + h_ent * (j - i)
            ax.plot([i, i, j, j],
                    [y_ann, y_ann + h_ent * 0.4, y_ann + h_ent * 0.4, y_ann],
                    lw=1.5, c="black")
            ax.text((i + j) / 2, y_ann + h_ent * 0.45, sym,
                    ha="center", va="bottom", fontsize=12, weight="bold")
 
ax.set_ylabel("Behavioral entropy", fontsize=18, weight="bold")
ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, weight="bold",
                   rotation=30, ha="right")
ax.tick_params(axis="y", labelsize=14)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_linewidth(2)
plt.tight_layout()
#plt.savefig("Behavioral_entropy.pdf", bbox_inches="tight")
plt.show()

# %%
# =============================================================================
# SECTION 14 — BEHAVIORAL ENTROPY (clean planned statistics version)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 14 — Behavioral entropy")
print("=" * 60)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.formula.api import ols
import statsmodels.api as sm


# ── Compute entropy per animal ───────────────────────────────────────────────

cluster_counts_ent = (
    df_no_outliers
    .groupby(["experimental_id", "sex", "condition", "Cluster"])
    .size()
    .reset_index(name="count")
)

state_entropy = (
    cluster_counts_ent
    .groupby(["experimental_id", "sex", "condition"])
    .apply(lambda x: shannon_entropy(x["count"]))
    .reset_index(name="entropy")
)

state_entropy["Group"] = state_entropy["condition"] + "-" + state_entropy["sex"]


# ── Define order (explicit, no alphabet chaos) ───────────────────────────────

ent_groups = ["ctrl-M", "PNELS-M", "ctrl-F", "PNELS-F"]

state_entropy["Group"] = pd.Categorical(
    state_entropy["Group"],
    categories=ent_groups,
    ordered=True
)

pal_ent = {g: pal_trans.get(g, "#888888") for g in ent_groups}


# ── Global effects (2-way ANOVA) ─────────────────────────────────────────────

model = ols("entropy ~ C(condition) * C(sex)", data=state_entropy).fit()
anova = sm.stats.anova_lm(model, typ=2)

print("\nTwo-way ANOVA:")
print(anova)


# ── Planned within-sex comparisons (ONLY meaningful ones) ────────────────────

def test_group(sex):
    g_ctrl = state_entropy[
        (state_entropy["sex"] == sex) &
        (state_entropy["condition"] == "ctrl")
    ]["entropy"]

    g_stress = state_entropy[
        (state_entropy["sex"] == sex) &
        (state_entropy["condition"] == "PNELS")
    ]["entropy"]

    stat, p = mannwhitneyu(g_ctrl, g_stress, alternative="two-sided")
    return p


p_f = test_group("F")
p_m = test_group("M")

print("\nWithin-sex comparisons:")
print(f"Females ctrl vs PNELS: p = {p_f:.4g}")
print(f"Males ctrl vs PNELS:   p = {p_m:.4g}")


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(max(8, len(ent_groups) * 1.5), 6))

sns.barplot(
    data=state_entropy,
    x="Group",
    y="entropy",
    order=ent_groups,
    palette=pal_ent,
    estimator="mean",
    errorbar="se",
    edgecolor="black",
    alpha=0.85,
    ax=ax
)

sns.stripplot(
    data=state_entropy,
    x="Group",
    y="entropy",
    order=ent_groups,
    color="black",
    jitter=0.25,
    size=5,
    alpha=0.6,
    ax=ax
)


# ── Significance annotation system ────────────────────────────────────────────

def p_to_symbol(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def add_bracket(ax, x1, x2, y, text, h):
    ax.plot([x1, x1, x2, x2],
            [y, y+h, y+h, y],
            lw=1.5, c="black")
    ax.text((x1+x2)/2, y+h*1.1, text,
            ha="center", va="bottom",
            fontsize=12, weight="bold")


# ── Global effects (line annotations instead of text blobs) ────────────────

y_max = state_entropy["entropy"].max()
h = y_max * 0.08

# how many bracket layers you will use
n_brackets = sum(p < 0.05 for _, _, p in comparisons)

# start global annotations ABOVE everything
base_y = y_max + h * (n_brackets + 2)

sex_p = anova.loc["C(sex)", "PR(>F)"]
cond_p = anova.loc["C(condition)", "PR(>F)"]

x_start = -0.3
x_end = len(ent_groups) - 0.7

offset = 0


def add_bracket(ax, x1, x2, y, text, h):
    ax.plot([x1, x1, x2, x2],
            [y, y+h, y+h, y],
            lw=1.5, c="black")
    ax.text((x1+x2)/2, y+h*1.1, text,
            ha="center", va="bottom",
            fontsize=12, weight="bold")


def add_global_line(ax, y, text):
    ax.plot([x_start, x_end], [y, y], lw=2, c="black")
    ax.text((x_start + x_end)/2, y + h*0.15,
            text, ha="center", va="bottom",
            fontsize=14, weight="bold")


offset = 1

# sex effect
if sex_p < 0.05:
    add_global_line(ax, y_max + h * offset, "# ")
    offset += 1

# condition effect
if cond_p < 0.05:
    add_global_line(ax, y_max + h * offset, "&")
    offset += 1


# ── Within-sex brackets ONLY ─────────────────────────────────────────────────

comparisons = [
    (0, 1, p_m),  # ctrl-M vs PNELS-M
    (2, 3, p_f)   # ctrl-F vs PNELS-F
]

for i, (x1, x2, p) in enumerate(comparisons):
    if p < 0.05:
        y = y_max + h * (offset + i)
        add_bracket(ax, x1, x2, y, p_to_symbol(p), h)


# ── Final styling ────────────────────────────────────────────────────────────

ax.set_ylabel("Behavioral entropy", fontsize=18, weight="bold")
ax.set_xlabel("")

ax.set_xticklabels(ax.get_xticklabels(),
                   fontsize=13, weight="bold",
                   rotation=30, ha="right")

ax.tick_params(axis="y", labelsize=14)

ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_linewidth(2)

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# SECTION 15 — CLUSTER–BIOMARKER CORRELATIONS  (NEW)
# =============================================================================
# This section correlates each cluster's occupancy (proportion of time an
# animal spends in that cluster) with external biomarkers such as cFos counts.
#
# Input: a CSV with columns [experimental_id, biomarker1, biomarker2, ...]
# If BIOMARKER_PATH is None the section is skipped gracefully.
# =============================================================================
 
print("\n" + "=" * 60)
print("SECTION 15 — Cluster–Biomarker Correlations")
print("=" * 60)
 
# ── 15a. Compute per-animal cluster occupancy proportions ────────────────────
occupancy = (
    df_no_outliers
    .groupby(["experimental_id", "Cluster"])
    .size()
    .reset_index(name="count")
)
total_per_animal = (
    df_no_outliers.groupby("experimental_id")
    .size().reset_index(name="total")
)
occupancy = occupancy.merge(total_per_animal, on="experimental_id")
occupancy["proportion"] = occupancy["count"] / occupancy["total"]
 
# Wide format: one row per animal, one column per cluster
occ_wide = occupancy.pivot(index="experimental_id",
                            columns="Cluster", values="proportion").fillna(0)
occ_wide.columns = [f"Cluster_{c}_prop" for c in occ_wide.columns]
occ_wide = occ_wide.reset_index()
 
if BIOMARKER_PATH is not None:
    df_bio = pd.read_csv(BIOMARKER_PATH)
    # Merge occupancy with biomarkers
    df_bio_merged = occ_wide.merge(df_bio, on="experimental_id", how="inner")
    print(f"  Animals with both occupancy and biomarker data: {len(df_bio_merged)}")
 
    biomarker_cols = [c for c in df_bio.columns if c != "experimental_id"]
    cluster_prop_cols = [c for c in occ_wide.columns if c != "experimental_id"]
 
    # Spearman correlation matrix: clusters × biomarkers
    rho_mat = pd.DataFrame(index=cluster_prop_cols, columns=biomarker_cols, dtype=float)
    pval_mat = pd.DataFrame(index=cluster_prop_cols, columns=biomarker_cols, dtype=float)
 
    for cp in cluster_prop_cols:
        for bm in biomarker_cols:
            valid = df_bio_merged[[cp, bm]].dropna()
            if len(valid) < 5:
                rho_mat.loc[cp, bm] = np.nan
                pval_mat.loc[cp, bm] = np.nan
            else:
                r, p = spearmanr(valid[cp], valid[bm])
                rho_mat.loc[cp, bm] = r
                pval_mat.loc[cp, bm] = p
 
    # FDR correction across all tests
    flat_p = pval_mat.values.flatten().astype(float)
    valid_mask = ~np.isnan(flat_p)
    flat_p_corr = np.full_like(flat_p, np.nan)
    if valid_mask.sum() > 0:
        _, p_adj, _, _ = multipletests(flat_p[valid_mask], method="fdr_bh")
        flat_p_corr[valid_mask] = p_adj
    pval_fdr_mat = pd.DataFrame(
        flat_p_corr.reshape(pval_mat.shape),
        index=cluster_prop_cols, columns=biomarker_cols)
 
    print("\n  Spearman ρ (cluster occupancy ~ biomarker):")
    print(rho_mat.round(3))
    print("\n  FDR-corrected p-values:")
    print(pval_fdr_mat.round(3))
 
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(max(6, len(biomarker_cols) * 1.5),
                                    max(4, len(cluster_prop_cols) * 0.8)))
    sns.heatmap(rho_mat.astype(float), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax,
                annot_kws={"size": 12, "weight": "bold"})
    # Overlay significance stars
    for i, cp in enumerate(cluster_prop_cols):
        for j, bm in enumerate(biomarker_cols):
            p_val = pval_fdr_mat.loc[cp, bm]
            sym   = p_to_symbol(p_val) if not np.isnan(p_val) else ""
            if sym not in ("ns", ""):
                ax.text(j + 0.5, i + 0.75, sym,
                        ha="center", va="center", fontsize=13,
                        color="black", weight="bold")
    ax.set_title("Cluster Occupancy × Biomarker Correlations (Spearman ρ)",
                 fontsize=14, weight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                       fontsize=12, weight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                       fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig("Biomarker_correlation_heatmap.pdf", bbox_inches="tight")
    plt.show()
else:
    print("  BIOMARKER_PATH not set — skipping biomarker correlation.")
    print("  Set BIOMARKER_PATH at the top of the script to activate this section.")
    # Still export the occupancy table so it is ready to join externally
    #occ_wide.to_csv("Cluster_occupancy_per_animal.csv", index=False)
    print("  Cluster occupancy table saved → Cluster_occupancy_per_animal.csv")
 

# %%
# =============================================================================
# SECTION 16 — MOST DISTINCT CLUSTER SELECTION  (NEW)
# =============================================================================
# Combines four orthogonal criteria into a single "distinctiveness score":
#
#   (a) Behavioral separation  — mean pairwise Dunn p-value rank across features
#   (b) Effect size            — mean η² from Kruskal-Wallis
#   (c) PERMANOVA significance — pseudo-F for condition and/or sex
#   (d) Biomarker linkage      — mean |ρ| to biomarkers (if available)
#
# Clusters are ranked and the top-k are highlighted.
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 16 — Most Distinct Cluster Selection")
print("=" * 60)
 
TOP_K = 3   # how many clusters to highlight
 
# ── (a) Behavioral separation score ──────────────────────────────────────────
# For each cluster: fraction of pairwise Dunn tests (vs other clusters) that
# are significant at p_FDR < 0.05
sep_scores = {}
for cid in cluster_ids:
    sig_count = total_count = 0
    for col, dm in dunn_results_all.items():
        for other in cluster_ids:
            if other == cid:
                continue
            total_count += 1
            try:
                if dm.loc[cid, other] < 0.05:
                    sig_count += 1
            except KeyError:
                pass
    sep_scores[cid] = sig_count / total_count if total_count > 0 else 0.0
 
# ── (b) Effect size (mean η² across all KW-significant features) ─────────────
eta2_by_feature = kw_df.set_index("feature")["eta2"].to_dict()
 
# Compute cluster-specific effect size as mean |delta mean| from grand mean
grand_mean = df_no_outliers[BEHAVIOR_COLS].mean()
effect_per_cluster = {}
for cid in cluster_ids:
    sub_mean = df_no_outliers[df_no_outliers["Cluster"] == cid][BEHAVIOR_COLS].mean()
    effect_per_cluster[cid] = (sub_mean - grand_mean).abs().mean()
 
# Normalise to [0, 1]
ef_vals  = np.array([effect_per_cluster[c] for c in cluster_ids])
ef_norm  = (ef_vals - ef_vals.min()) / (ef_vals.max() - ef_vals.min() + 1e-9)
ef_normd = dict(zip(cluster_ids, ef_norm))
 
# ── (c) PERMANOVA significance score ─────────────────────────────────────────
perm_f_scores = {cid: 0.0 for cid in cluster_ids}
if len(perm_df) > 0:
    for _, row in perm_df.iterrows():
        # Add pseudo-F if significant
        if row["p_fdr"] < 0.05:
            perm_f_scores[row["Cluster"]] = (
                perm_f_scores.get(row["Cluster"], 0) + row["pseudo-F"])
    pf_vals = np.array([perm_f_scores[c] for c in cluster_ids])
    pf_norm = (pf_vals - pf_vals.min()) / (pf_vals.max() - pf_vals.min() + 1e-9)
    pf_normd = dict(zip(cluster_ids, pf_norm))
else:
    pf_normd = {c: 0.0 for c in cluster_ids}
 
# ── (d) Biomarker linkage score ───────────────────────────────────────────────
bio_scores = {cid: 0.0 for cid in cluster_ids}
if BIOMARKER_PATH is not None:
    for cid in cluster_ids:
        col_name = f"Cluster_{cid}_prop"
        if col_name in rho_mat.index:
            bio_scores[cid] = rho_mat.loc[col_name].abs().mean()
    bv = np.array([bio_scores[c] for c in cluster_ids])
    bv_norm = (bv - bv.min()) / (bv.max() - bv.min() + 1e-9)
    bio_normd = dict(zip(cluster_ids, bv_norm))
else:
    bio_normd = {c: 0.0 for c in cluster_ids}
 
# ── Composite score ───────────────────────────────────────────────────────────
# Weights: separation 35%, effect size 30%, PERMANOVA 20%, biomarker 15%
W_SEP, W_EF, W_PERM, W_BIO = 0.35, 0.30, 0.20, 0.15
 
# Normalise sep_scores
sv = np.array([sep_scores[c] for c in cluster_ids])
sv_norm = (sv - sv.min()) / (sv.max() - sv.min() + 1e-9)
sv_normd = dict(zip(cluster_ids, sv_norm))
 
composite = {}
for cid in cluster_ids:
    composite[cid] = (W_SEP  * sv_normd[cid] +
                      W_EF   * ef_normd[cid] +
                      W_PERM * pf_normd[cid] +
                      W_BIO  * bio_normd[cid])
 
rank_df = pd.DataFrame({
    "Cluster":          cluster_ids,
    "Sep_score":        [sep_scores[c]         for c in cluster_ids],
    "EffectSize_norm":  [ef_normd[c]           for c in cluster_ids],
    "PERMANOVA_norm":   [pf_normd[c]           for c in cluster_ids],
    "Biomarker_norm":   [bio_normd[c]          for c in cluster_ids],
    "Composite_score":  [composite[c]          for c in cluster_ids],
}).sort_values("Composite_score", ascending=False).reset_index(drop=True)
 
rank_df["Rank"] = rank_df.index + 1
print("\n  Cluster Distinctiveness Ranking:")
print(rank_df.to_string(index=False))
 
top_clusters = rank_df.head(TOP_K)["Cluster"].tolist()
print(f"\n  → Top {TOP_K} most distinct clusters: {top_clusters}")
 
# ── Bar chart of composite scores ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(6, len(cluster_ids) * 1.0), 5))
bar_cols = [cluster_to_color.get(c, "#aaaaaa") for c in rank_df["Cluster"]]
bars = ax.bar(rank_df["Cluster"].astype(str), rank_df["Composite_score"],
              color=bar_cols, edgecolor="black", linewidth=1.2)
 
# Highlight top clusters with a gold border
for bar, cid in zip(bars, rank_df["Cluster"]):
    if cid in top_clusters:
        bar.set_edgecolor("gold")
        bar.set_linewidth(3)
 
ax.set_xlabel("Cluster", fontsize=16, weight="bold")
ax.set_ylabel("Composite Distinctiveness Score", fontsize=14, weight="bold")
ax.set_title(f"Cluster Distinctiveness\n(top {TOP_K} highlighted in gold border)",
             fontsize=14, weight="bold")
ax.tick_params(labelsize=13)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_linewidth(2)
plt.tight_layout()
#plt.savefig("Cluster_distinctiveness_ranking.pdf", bbox_inches="tight")
plt.show()
 
# ── Heatmap of component scores ───────────────────────────────────────────────
score_mat = rank_df.set_index("Cluster")[
    ["Sep_score", "EffectSize_norm", "PERMANOVA_norm",
     "Composite_score"]].astype(float) # "Biomarker_norm"
fig, ax = plt.subplots(figsize=(9, max(4, len(cluster_ids) * 0.7)))
sns.heatmap(score_mat, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, ax=ax,
            annot_kws={"size": 12, "weight": "bold"})
ax.set_title("Cluster Distinctiveness Component Scores", fontsize=14, weight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right",
                   fontsize=12, weight="bold")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12, weight="bold")
plt.tight_layout()
#plt.savefig("Cluster_distinctiveness_components.pdf", bbox_inches="tight")
plt.show()
 
# ── Export key tables ─────────────────────────────────────────────────────────
#rank_df.to_csv("Cluster_distinctiveness_ranking.csv", index=False)
#kw_df.to_csv("KW_results.csv", index=False)
#perm_df.to_csv("PERMANOVA_results.csv", index=False) if len(perm_df) > 0 else None
 
print("\n" + "=" * 60)
print("Pipeline complete.  Output files saved in working directory.")
print("=" * 60)

# %%
