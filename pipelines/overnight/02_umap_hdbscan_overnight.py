"""
02_umap_hdbscan_overnight.py
============================
Overnight (12-hour) behavioural clustering pipeline for DeepBox recordings.

Pipeline overview
-----------------
1.  Load & combine female + male DeepBox CSVs
2.  Filter IDs / genotypes, cap time at 12 h
3.  5-second interval binning (sum / mean / std per behaviour)
4.  Scale features → load pre-trained UMAP → transform
5.  HDBSCAN clustering
6.  Bootstrap stability validation (parallel, Jaccard threshold)
7.  Behavioural repertoire visualisation (heatmaps, radial plots)
8.  Sex × Genotype composition (chi-square, GLM, GEE w/ BW correction)
9.  Temporal cluster dynamics & significance (hourly, BW-adjusted)
10. Transition analysis (counts, probability matrices, networks, Markov)
11. Dwell time analysis
12. Entropy & behavioural complexity

Outputs (PDFs)
--------------
All figures are saved to the FIGURES_DIR defined in PARAMETERS.

Usage
-----
Run as a plain Python script or cell-by-cell in VS Code / JupyterLab.
Shared helper functions are imported from utils.py in the same folder.
"""

# ===========================================================================
# IMPORTS
# ===========================================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
import joblib

from collections import defaultdict
from itertools import combinations

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency, entropy as sp_entropy
from scipy.spatial.distance import pdist, euclidean

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

from skbio.stats.distance import permanova, DistanceMatrix
from joblib import Parallel, delayed
from pypalettes import load_cmap

import hdbscan

# Shared utilities (plot helpers, loaders, etc.)
from utils import (
    bin_behaviours,
    build_cluster_colour_map,
    plot_umap_embedding,
    plot_cluster_heatmap,
    plot_radial,
    compute_transitions,
    plot_transition_networks,
    compute_entropy,
    chi_square_cluster_composition,
    permanova_cluster_factor,
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# ===========================================================================
# PARAMETERS  ← edit these before running
# ===========================================================================

# --- Input files ---
FEMALE_CSV = (
    r"/Users/veronika/Nextcloud/PCA_analysis2025/DeepBox/chow/"
    r"females/atg7KO/master_combined_females_atg7KO_chow_DEEPBOX_FINAL.csv"
)
MALE_CSV = (
    r"/Users/veronika/Nextcloud/PCA_analysis2025/DeepBox/chow/"
    r"males/atg7KO/master_combined_males_atg7KO_chow_DEEPBOX_FINAL.csv"
)
BW_FEMALE_CSV = (
    r"/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/chow/females/"
    r"atg7KO/atg7KO_females_chow_SocialOF/bw_atg7KO_chow.csv"
)
BW_MALE_CSV = (
    r"/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/chow/males/"
    r"atg7KO/atg7KO_males_chow_SocialOF/bw_atg7KO_chow.csv"
)

# --- Pre-trained UMAP model ---
UMAP_MODEL_PATH = "umap_model_DB_chow.sav"

# --- Cached bootstrapping outputs (set LOAD_BOOTSTRAP=True to skip re-running) ---
BOOTSTRAP_CACHE    = "bootstrap_results_raw_atg7KD_DB_chow.joblib"
STAB_METRICS_CSV   = "cluster_stability_metrics_atg7KD_chow_DB.csv"
STABLE_DATA_PARQUET = "final_stable_data_atg7KD_chow_DB.parquet"
LOAD_BOOTSTRAP = True   # ← set to False to re-run bootstrapping from scratch

# --- Exclusions ---
EXCLUDED_IDS   = ["ID49", "ID61", "ID63"]
EXCLUDED_GENO  = "atg7OE"
TIME_CAP       = pd.Timedelta("0 days 12:00:00")

# --- Binning ---
INTERVAL_SEC = 5  # seconds per interval bin

# --- HDBSCAN ---
MIN_CLUSTER_SIZE = 600
MIN_SAMPLES      = 300

# --- Bootstrap ---
JACCARD_THRESHOLD = 0.75
N_BOOTSTRAPS      = 100
SAMPLE_FRAC       = 0.6
N_JOBS            = 6

# --- Visual ---
GROUP_COLORS = {
    "control-Male":   "#000000",
    "control-Female": "#9D9D9D",
    "atg7KO-Male":    "#444B29",
    "atg7KO-Female":  "#A44316",
}
GROUP_ORDER = ["control-Male", "control-Female", "atg7KO-Male", "atg7KO-Female"]

# --- Output directory ---
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def fig_path(filename):
    """Return full path for a figure file."""
    return os.path.join(FIGURES_DIR, filename)


# ===========================================================================
# GLOBAL STYLE
# ===========================================================================
sns.set_style("white")
sns.set_context("paper", rc={
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
})
plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "legend.title_fontsize": 12,
})


# ===========================================================================
# SECTION 1 – DATA LOADING & PREPROCESSING
# ===========================================================================
print("=" * 70)
print("SECTION 1: Loading & preprocessing data")
print("=" * 70)

BEHAVIOUR_COLS = [
    "nose2nose", "sidebyside", "sidereside", "nose2tail", "nose2body",
    "following", "climb-arena", "sniff-arena", "immobility",
    "stat-lookaround", "stat-active", "stat-passive",
    "moving", "sniffing", "speed", "missing",
]

df_females = pd.read_csv(FEMALE_CSV)
df_males   = pd.read_csv(MALE_CSV)

df_females["Sex"] = "Female"
df_males["Sex"]   = "Male"

for df in (df_females, df_males):
    df["Time"] = pd.to_timedelta(df["Time"])

df_combined = pd.concat([df_females, df_males], ignore_index=True)
df_combined["Time"] = pd.to_timedelta(df_combined["Time"])
df_combined.fillna(0, inplace=True)

# Keep only behaviour columns that exist in this dataset
behaviour_cols = [c for c in BEHAVIOUR_COLS if c in df_combined.columns]
df_combined[behaviour_cols] = df_combined[behaviour_cols].fillna(0)

# --- Filter: IDs, genotype, time cap ---
df_filtered = df_combined[
    (~df_combined["experimental_id"].isin(EXCLUDED_IDS)) &
    (df_combined["Geno"] != EXCLUDED_GENO) &
    (df_combined["Time"] <= TIME_CAP)
].copy()

# Drop any redundant speed columns
extra_speed = [c for c in df_filtered.columns if "speed" in c and c != "speed"]
if extra_speed:
    df_filtered.drop(columns=extra_speed, inplace=True)


# ===========================================================================
# SECTION 2 – TIME BINNING
# ===========================================================================
print("SECTION 2: Time binning")

df_filtered["Time_sec"]    = df_filtered["Time"].dt.total_seconds()
df_filtered["Interval_bin"] = (df_filtered["Time_sec"] // INTERVAL_SEC).astype(int)

behaviour_cols = [c for c in behaviour_cols if c in df_filtered.columns]
agg_dict = {col: ["sum", "mean", "std"] for col in behaviour_cols}

features = (
    df_filtered
    .groupby(["experimental_id", "Geno", "Sex", "Interval_bin"])
    .agg(agg_dict)
)
features.columns = ["_".join(col).strip() for col in features.columns]
features = features.reset_index()

features["Interval_start"] = pd.to_timedelta(
    features["Interval_bin"] * INTERVAL_SEC, unit="s"
)
features["Interval_end"] = pd.to_timedelta(
    (features["Interval_bin"] + 1) * INTERVAL_SEC, unit="s"
)
features["Interval_label"] = (
    features["Interval_start"].astype(str) + " - " +
    features["Interval_end"].astype(str)
)
features["Hour_bin"]     = features["Interval_bin"] // (3600 // INTERVAL_SEC)
features["Halfhour_bin"] = (
    features["Interval_bin"] // (1800 / INTERVAL_SEC)
).astype(int)

filtered_features = features.sort_values(
    ["experimental_id", "Interval_label"]
).reset_index(drop=True)


# ===========================================================================
# SECTION 3 – FEATURE SELECTION & SCALING
# ===========================================================================
print("SECTION 3: Feature selection & scaling")

EXCLUDE_META = [
    "experimental_id", "Geno", "Sex", "Interval_bin",
    "speed_mean", "speed_std", "speed_sum",
    "Interval_start", "Interval_end", "Interval_label", "Hour_bin",
]
feature_columns = [
    c for c in filtered_features.columns
    if c not in EXCLUDE_META and (
        c.endswith("_sum") or c.endswith("_mean") or c.endswith("_std")
    )
]
feature_columns_active = [c for c in feature_columns if "missing" not in c]
print(f"  Active features ({len(feature_columns_active)}): {feature_columns_active}")

filtered_features = filtered_features.dropna(subset=feature_columns_active)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(filtered_features[feature_columns_active])


# ===========================================================================
# SECTION 4 – UMAP (load pre-trained model)
# ===========================================================================
print("SECTION 4: Transforming with pre-trained UMAP model")

loaded_model = joblib.load(UMAP_MODEL_PATH)
embedding    = loaded_model.transform(X_scaled)


# ===========================================================================
# SECTION 5 – HDBSCAN CLUSTERING
# ===========================================================================
print("SECTION 5: HDBSCAN clustering")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
)
labels = clusterer.fit_predict(embedding)

features_clustered = filtered_features.copy()
features_clustered["UMAP1"]   = embedding[:, 0]
features_clustered["UMAP2"]   = embedding[:, 1]
features_clustered["Cluster"] = labels

features_clustered_clean = features_clustered[
    features_clustered["Cluster"] != -1
].copy()
features_clustered_clean["filtered_Cluster"] = features_clustered_clean["Cluster"]

# Quick overview: missing per cluster
missing_summary = (
    features_clustered
    .groupby("Cluster")["missing_mean"]
    .agg(["mean", "std", "count"])
)
print("Missing % per cluster (post-hoc):")
print(missing_summary)

# Cluster profile means (for later use)
cluster_means_with_missing = features_clustered.groupby("Cluster")[
    [col for col in feature_columns if col != "missing_mean"] + ["missing_mean"]
].mean()


# ===========================================================================
# SECTION 6 – BOOTSTRAP STABILITY VALIDATION
# ===========================================================================
print("SECTION 6: Bootstrap stability validation")


def _single_bootstrap(
    seed, orig_emb, orig_labels, N_full, sample_frac,
    min_cluster_size_full, min_samples_full
):
    """Run one bootstrap replicate and return per-cluster Jaccard scores."""
    rng = np.random.default_rng(seed)
    sample_size = max(1, int(sample_frac * N_full))
    idx = rng.choice(np.arange(N_full), size=sample_size, replace=True)

    emb_b   = orig_emb[idx]
    labs_ref = orig_labels[idx]

    mc_b = max(2, int(min_cluster_size_full * (len(emb_b) / N_full)))
    ms_b = max(1, int(0.5 * mc_b))

    clust_b  = hdbscan.HDBSCAN(min_cluster_size=mc_b, min_samples=ms_b)
    labels_b = clust_b.fit_predict(emb_b)

    # Build contingency matrix
    row_labels = np.unique(labs_ref)
    col_labels = np.unique(labels_b)
    row_idx = {lb: i for i, lb in enumerate(row_labels)}
    col_idx = {lb: j for j, lb in enumerate(col_labels)}
    C = np.zeros((len(row_labels), len(col_labels)), dtype=int)
    for r, b in zip(labs_ref, labels_b):
        if r == -1 or b == -1:
            continue
        C[row_idx[r], col_idx[b]] += 1

    # Hungarian matching
    m, n = C.shape
    dim  = max(m, n)
    Cpad = np.zeros((dim, dim), dtype=int)
    Cpad[:m, :n] = C
    ri, ci = linear_sum_assignment(-Cpad)
    mapping = {
        col_labels[c]: row_labels[r]
        for r, c in zip(ri, ci)
        if r < m and c < n and Cpad[r, c] > 0
    }
    mapped = np.array([mapping.get(lb, -9999) for lb in labels_b])

    # Per-cluster Jaccard
    jaccs = {}
    for orig_c in [u for u in np.unique(orig_labels) if u != -1]:
        om = labs_ref == orig_c
        if om.sum() == 0:
            jaccs[orig_c] = 0.0
            continue
        mm = mapped == orig_c
        inter = np.logical_and(om, mm).sum()
        union = np.logical_or(om, mm).sum()
        jaccs[orig_c] = (inter / union) if union > 0 else 0.0

    return jaccs, mapping, idx, labels_b


if LOAD_BOOTSTRAP:
    print("  Loading cached bootstrap results...")
    results      = joblib.load(BOOTSTRAP_CACHE)
    stab_metrics = pd.read_csv(STAB_METRICS_CSV)
    df_stable    = pd.read_parquet(STABLE_DATA_PARQUET)
else:
    print(f"  Running {N_BOOTSTRAPS} bootstrap replicates (n_jobs={N_JOBS})...")
    orig_emb    = features_clustered[["UMAP1", "UMAP2"]].values
    orig_labels = features_clustered["Cluster"].values
    N_full      = len(orig_labels)
    mc_full     = int(np.clip(0.01 * N_full, 30, 150))
    ms_full     = max(1, int(0.5 * mc_full))

    results = Parallel(n_jobs=N_JOBS)(
        delayed(_single_bootstrap)(
            int(42 + b), orig_emb, orig_labels, N_full,
            SAMPLE_FRAC, mc_full, ms_full
        )
        for b in range(N_BOOTSTRAPS)
    )

    # Aggregate Jaccard scores
    jacc_agg = defaultdict(list)
    for jaccs, *_ in results:
        for c, v in jaccs.items():
            jacc_agg[c].append(v)

    stab_summary = [
        {
            "Cluster": c,
            "size": int((orig_labels == c).sum()),
            "mean_jaccard": float(np.nanmean(vals)),
            "median_jaccard": float(np.nanmedian(vals)),
            "n_bootstraps_present": int(np.sum(np.array(vals) > 0)),
        }
        for c, vals in sorted(jacc_agg.items())
    ]
    stab_metrics = pd.DataFrame(stab_summary).sort_values("mean_jaccard")
    stab_metrics["keep"] = np.where(
        stab_metrics["mean_jaccard"] >= JACCARD_THRESHOLD, "keep", "drop"
    )
    print(stab_metrics)

    keep_clusters_full = stab_metrics.loc[
        stab_metrics["keep"] == "keep", "Cluster"
    ].tolist()
    df_stable = features_clustered[
        features_clustered["Cluster"].isin(keep_clusters_full)
    ].copy()

    # Save
    joblib.dump(results, BOOTSTRAP_CACHE)
    stab_metrics.to_csv(STAB_METRICS_CSV, index=False)
    df_stable.to_parquet(STABLE_DATA_PARQUET)
    print("  Bootstrap results saved.")

# Identify stable clusters
keep_clusters = stab_metrics.loc[
    stab_metrics["keep"] == "keep", "Cluster"
].tolist()
print(f"  Stable clusters: {keep_clusters}  (n={len(keep_clusters)})")

# Align variables expected by downstream code
features_clustered = df_stable
filtered           = df_stable[
    df_stable["Cluster"].isin(keep_clusters)
].copy()

if str(filtered["Cluster"].dtype).startswith("category"):
    filtered["Cluster"] = filtered["Cluster"].cat.remove_unused_categories()
print(f"  Rows retained after stability filter: {len(filtered)}")

# --- Jaccard histogram ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(stab_metrics["mean_jaccard"], bins=15, edgecolor="black")
ax.axvline(JACCARD_THRESHOLD, linestyle="--", color="red", label=f"threshold={JACCARD_THRESHOLD}")
ax.set_xlabel("Mean Jaccard stability")
ax.set_ylabel("Count")
ax.set_title("Cluster Bootstrap Stability")
ax.legend()
plt.tight_layout()
plt.savefig(fig_path("bootstrap_jaccard_histogram.pdf"), bbox_inches="tight")
plt.show()

# --- UMAP: keep vs drop ---
mask_keep = features_clustered["Cluster"].isin(keep_clusters)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(
    features_clustered.loc[~mask_keep, "UMAP1"],
    features_clustered.loc[~mask_keep, "UMAP2"],
    s=8, alpha=0.25, label="Drop / Unstable"
)
ax.scatter(
    features_clustered.loc[mask_keep, "UMAP1"],
    features_clustered.loc[mask_keep, "UMAP2"],
    s=8, alpha=0.7, label="Keep / Stable"
)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("UMAP – Stable vs Unstable Clusters")
ax.legend()
plt.tight_layout()
plt.savefig(fig_path("REPORT_UMAP_stable_vs_unstable.pdf"), bbox_inches="tight")
plt.show()


# ===========================================================================
# SECTION 7 – BEHAVIOURAL REPERTOIRE VISUALISATION
# ===========================================================================
print("SECTION 7: Behavioural repertoire visualisation")

stable_cluster_means = cluster_means_with_missing.loc[
    cluster_means_with_missing.index.isin(keep_clusters)
].copy()

sum_cols  = [c for c in stable_cluster_means.columns if c.endswith("_sum")]
mean_cols = [c for c in stable_cluster_means.columns if c.endswith("_mean")]
std_cols  = [c for c in stable_cluster_means.columns if c.endswith("_std")]

if "missing_mean" in stable_cluster_means.columns:
    if "missing_mean" not in mean_cols:
        mean_cols.append("missing_mean")
    if "missing_std" not in stable_cluster_means.columns:
        stable_cluster_means["missing_std"] = 0
    if "missing_std" not in std_cols:
        std_cols.append("missing_std")


def _plot_heatmap(df, cols, title, save_path=None):
    fig, ax = plt.subplots(figsize=(25, 8))
    sns.heatmap(
        df[cols], cmap="viridis", annot=True, fmt=".2f",
        annot_kws={"fontsize": 12}, ax=ax
    )
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Behavioural feature", fontsize=14)
    ax.set_ylabel("Cluster", fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


_plot_heatmap(stable_cluster_means, sum_cols,
              "Cluster Profiles – SUM",
              fig_path("cluster_profiles_sum.pdf"))
_plot_heatmap(stable_cluster_means, mean_cols,
              "Cluster Profiles – MEAN (incl. missing)",
              fig_path("cluster_profiles_mean.pdf"))
_plot_heatmap(stable_cluster_means, std_cols,
              "Cluster Profiles – STD",
              fig_path("cluster_profiles_std.pdf"))

# --- Proportional radial plots ---
total_per_cluster = stable_cluster_means[sum_cols].sum(axis=1)
prop_sum          = stable_cluster_means[sum_cols].div(total_per_cluster, axis=0)
prop_sum_no0      = prop_sum.loc[prop_sum.index != 0]

cmap_tab20 = plt.cm.tab20
cluster_to_color = {
    c: cmap_tab20(i) for i, c in enumerate(stable_cluster_means.index)
}


def _plot_radial(df_mean, df_std=None, title="", colors=None,
                 fill=True, show_std=True, save_path=None):
    categories = [
        c.replace("_sum", "").replace("_mean", "").replace("_std", "")
        for c in df_mean.columns
    ]
    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    for cluster in df_mean.index:
        vals   = df_mean.loc[cluster].values.tolist() + [df_mean.loc[cluster].values[0]]
        color  = colors[cluster]
        ax.plot(angles, vals, color=color, linewidth=2.5,
                label=f"Cluster {cluster}")
        if fill:
            ax.fill(angles, vals, color=color, alpha=0.18)
        if df_std is not None and show_std:
            errs = df_std.loc[cluster].values.tolist()
            errs += errs[:1]
            lower = np.maximum(np.array(vals) - np.array(errs), 0)
            upper = np.array(vals) + np.array(errs)
            ax.fill_between(angles, lower, upper, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, weight="bold")
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        deg = np.degrees(angle)
        label.set_rotation(deg)
        label.set_rotation_mode("anchor")
        label.set_horizontalalignment(
            "left" if deg <= 90 or deg >= 270 else "right"
        )
    ax.set_rmax(max(df_mean.max().max() * 1.15, 0.6))
    ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1.05),
              frameon=False, fontsize=10)
    plt.title(title, fontsize=14, weight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()


_plot_radial(
    prop_sum_no0, title="Behavioural Repertoire – PROPORTIONS (excl. cluster 0)",
    colors=cluster_to_color, fill=True, show_std=False,
    save_path=fig_path("REPORT_radial_proportions.pdf")
)


# ===========================================================================
# SECTION 8 – SEX × GENOTYPE COMPOSITION
# ===========================================================================
print("SECTION 8: Sex × Genotype composition")

df_stable = features_clustered[features_clustered["Cluster"].isin(keep_clusters)].copy()
df_stable["Sex_Geno"] = df_stable["Geno"] + "-" + df_stable["Sex"]

# --- Global chi-square ---
contingency = pd.crosstab(df_stable["Cluster"], df_stable["Sex_Geno"])
chi2_global, p_global, dof_global, expected = chi2_contingency(contingency)
residuals = (contingency - expected) / np.sqrt(expected)

n_total  = contingency.values.sum()
k        = min(contingency.shape)
cramers_v = np.sqrt(chi2_global / (n_total * (k - 1)))

print(f"\nGlobal Chi-Square: χ²={chi2_global:.2f}, df={dof_global}, "
      f"p={p_global:.4e}, Cramér's V={cramers_v:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
sns.heatmap(
    np.log2(contingency.div(contingency.sum(axis=1), axis=0).add(1e-6)),
    cmap="coolwarm", center=0, annot=True, ax=axes[0]
)
axes[0].set_title("Cluster Preference (Log2 Enrichment)")
sns.heatmap(
    residuals, annot=True, cmap="RdBu_r", center=0, ax=axes[1]
)
axes[1].set_title("Pearson Residuals (|>2| = significant)")
plt.tight_layout()
plt.savefig(fig_path("cluster_significance_heatmap_atg7KD_chow.pdf"),
            bbox_inches="tight")
plt.show()

# --- Stacked bar (raw proportions) ---
cluster_props = (
    contingency
    .div(contingency.sum(axis=1), axis=0)
    .reindex(columns=GROUP_ORDER, fill_value=0)
)
fig, ax = plt.subplots(figsize=(12, 6))
cluster_props.plot(
    kind="bar", stacked=True, ax=ax,
    color=[GROUP_COLORS[g] for g in GROUP_ORDER]
)
ax.set_ylabel("Proportion")
ax.set_xlabel("Cluster")
ax.set_title("Proportion of Sex × Genotype per Cluster (All Hours)")
ax.legend(title="Sex × Genotype", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(
    fig_path("REPORT_proportion_SEXGENOperCluster_ALLhours_atg7KD_chow.pdf"),
    bbox_inches="tight"
)
plt.show()

# --- Per-cluster GLM (genotype × sex composition) ---
prop_df = (
    df_stable
    .groupby(["Cluster", "Hour_bin", "Sex_Geno"])
    .size()
    .reset_index(name="count")
)
prop_df["total"] = (
    prop_df.groupby(["Cluster", "Hour_bin"])["count"].transform("sum")
)
prop_df["proportion"] = prop_df["count"] / prop_df["total"]
prop_df = prop_df.sort_values(["Cluster", "Hour_bin", "Sex_Geno"]).reset_index(drop=True)


def _get_symbol(p, sym="*"):
    if p < 0.001: return sym * 3
    if p < 0.01:  return sym * 2
    if p < 0.05:  return sym
    return ""


print("\n--- GLM: per-cluster genotype × sex effects ---")
print("=" * 80)
for cid in sorted(prop_df["Cluster"].unique()):
    df_c = prop_df[prop_df["Cluster"] == cid].copy()
    df_c["Geno"] = df_c["Sex_Geno"].str.split("-").str[0]
    df_c["Sex"]  = df_c["Sex_Geno"].str.split("-").str[1]
    try:
        model = smf.glm(
            "proportion ~ C(Geno, Treatment(reference='control')) "
            "* C(Sex, Treatment(reference='Female')) + C(Hour_bin)",
            data=df_c,
            family=sm.families.Binomial(),
            var_weights=df_c["total"],
        ).fit()
        wald  = model.wald_test_terms().table
        chi_g = wald.loc["C(Geno, Treatment(reference='control'))", "statistic"].item()
        p_g   = wald.loc["C(Geno, Treatment(reference='control'))", "pvalue"].item()
        chi_s = wald.loc["C(Sex, Treatment(reference='Female'))", "statistic"].item()
        p_s   = wald.loc["C(Sex, Treatment(reference='Female'))", "pvalue"].item()
        chi_i = wald.loc[
            "C(Geno, Treatment(reference='control')):"
            "C(Sex, Treatment(reference='Female'))", "statistic"
        ].item()
        p_i = wald.loc[
            "C(Geno, Treatment(reference='control')):"
            "C(Sex, Treatment(reference='Female'))", "pvalue"
        ].item()
        print(
            f"Cluster {cid}: Geno χ²={chi_g:.2f} p={p_g:.4e}, "
            f"Sex χ²={chi_s:.2f} p={p_s:.4e}, "
            f"Interaction χ²={chi_i:.2f} p={p_i:.4e}"
        )
    except Exception as e:
        print(f"Cluster {cid}: model failed – {e}")


# ===========================================================================
# SECTION 9 – BODY WEIGHT (BW) CORRECTION & ADJUSTED PROPORTIONS
# ===========================================================================
print("\nSECTION 9: BW-corrected GEE models")

df_bw_f = pd.read_csv(BW_FEMALE_CSV)
df_bw_m = pd.read_csv(BW_MALE_CSV)
df_bw   = pd.concat([df_bw_f, df_bw_m], ignore_index=True)

# Build per-animal proportion dataframe with BW
df_seq = df_stable.copy()
df_seq["Sex_Geno"] = df_seq["Geno"] + "-" + df_seq["Sex"]

prop_per_animal = (
    df_seq
    .groupby(["experimental_id", "Cluster", "Hour_bin", "Sex", "Geno"])
    .size()
    .reset_index(name="count")
)
total_per_ah = (
    df_seq
    .groupby(["experimental_id", "Hour_bin"])
    .size()
    .reset_index(name="total")
)
prop_per_animal = prop_per_animal.merge(
    total_per_ah, on=["experimental_id", "Hour_bin"]
)
prop_per_animal["proportion"] = prop_per_animal["count"] / prop_per_animal["total"]
prop_per_animal = prop_per_animal.merge(
    df_bw[["experimental_id", "BW"]], on="experimental_id", how="left"
)
prop_per_animal["Sex_Geno"] = prop_per_animal["Geno"] + "-" + prop_per_animal["Sex"]

# --- Main-effects GEE (all clusters pooled) ---
gee_main = smf.gee(
    "count ~ C(Sex_Geno, Treatment(reference='control-Female')) + Hour_bin + BW",
    groups="experimental_id",
    data=prop_per_animal,
    family=sm.families.Poisson(),
    offset=np.log(prop_per_animal["total"]),
    cov_struct=Exchangeable(),
).fit(cov_type="robust")
print(gee_main.summary())

# Wald test: genotype effect within males
wald_male = gee_main.wald_test(
    "C(Sex_Geno, Treatment(reference='control-Female'))[T.atg7KO-Male] = "
    "C(Sex_Geno, Treatment(reference='control-Female'))[T.control-Male]"
)
print(f"\nWald test (atg7KO-Male vs control-Male): "
      f"χ²={wald_male.statistic[0][0]:.4f}, p={wald_male.pvalue:.4f}")

# --- Per-cluster BW-adjusted GEE factorial ---
print("\n--- GEE Factorial (BW-adjusted) per cluster ---")
gee_results = []
for cid in sorted(prop_per_animal["Cluster"].unique()):
    df_c = prop_per_animal[prop_per_animal["Cluster"] == cid].copy()
    df_c["count"] = df_c["count"] + 0.1  # Laplace smoothing
    try:
        model_gee = smf.gee(
            "count ~ C(Geno, Treatment('control')) * C(Sex, Treatment('Female')) "
            "+ Hour_bin + BW",
            groups="experimental_id",
            data=df_c,
            family=sm.families.Poisson(),
            offset=np.log(df_c["total"].clip(lower=1)),
            cov_struct=Exchangeable(),
        ).fit(cov_type="robust")

        wald  = model_gee.wald_test_terms().table
        p_g   = wald.loc["C(Geno, Treatment('control'))", "pvalue"].item()
        p_s   = wald.loc["C(Sex, Treatment('Female'))", "pvalue"].item()
        p_i   = wald.loc[
            "C(Geno, Treatment('control')):C(Sex, Treatment('Female'))", "pvalue"
        ].item()
        gee_results.append({
            "Cluster": cid,
            "Geno (*)": _get_symbol(p_g),
            "Sex (#)":  _get_symbol(p_s, "#"),
            "GxS (x)":  _get_symbol(p_i, "x"),
            "P_Geno": p_g, "P_Sex": p_s, "P_Inter": p_i,
        })
    except Exception as e:
        print(f"  Cluster {cid} failed: {e}")

gee_sig_df = pd.DataFrame(gee_results)
print(gee_sig_df[["Cluster", "Geno (*)", "Sex (#)", "GxS (x)"]])

# --- BW-adjusted hourly line plots with significance stars ---
adj_records = []
sig_records = []
all_hours   = sorted(prop_per_animal["Hour_bin"].unique())

for cid in sorted(prop_per_animal["Cluster"].unique()):
    df_c = prop_per_animal[prop_per_animal["Cluster"] == cid].copy()
    df_c["count"] = df_c["count"] + 0.1

    mean_bw   = df_c["BW"].mean()
    mean_hour = df_c["Hour_bin"].mean()

    for hr in all_hours:
        hr_data = df_c[df_c["Hour_bin"] == hr]
        if hr_data["count"].sum() < 0.5:
            continue
        try:
            res = smf.gee(
                "count ~ C(Sex_Geno, Treatment(reference='control-Female')) + BW",
                groups="experimental_id",
                data=hr_data,
                family=sm.families.Poisson(),
                offset=np.log(hr_data["total"]),
                cov_struct=Exchangeable(),
            ).fit(cov_type="robust")

            for grp in GROUP_ORDER:
                if grp == "control-Female":
                    continue
                term = (
                    f"C(Sex_Geno, Treatment(reference='control-Female'))[T.{grp}]"
                )
                # Adjusted occupancy: coef + BW effect at mean BW
                coef = res.params.get(term, 0.0)
                bw_c = res.params.get("BW", 0.0)
                adj_prop = np.exp(coef + bw_c * mean_bw) / (
                    1 + np.exp(coef + bw_c * mean_bw)
                )
                adj_records.append({
                    "Cluster": cid, "Hour": hr,
                    "Sex_Geno": grp, "Adj_Prop": adj_prop,
                })

            # Wald significance
            def _p_star(p):
                if p < 0.001: return "***"
                if p < 0.01:  return "**"
                if p < 0.05:  return "*"
                return None

            h_f = "C(Sex_Geno, Treatment(reference='control-Female'))[T.atg7KO-Female] = 0"
            h_m = (
                "C(Sex_Geno, Treatment(reference='control-Female'))[T.atg7KO-Male] = "
                "C(Sex_Geno, Treatment(reference='control-Female'))[T.control-Male]"
            )
            sig_records.append({
                "Cluster": cid, "Hour": hr,
                "Female_Star": _p_star(res.wald_test(h_f).pvalue),
                "Male_Star":   _p_star(res.wald_test(h_m).pvalue),
            })
        except Exception:
            continue

adj_df    = pd.DataFrame(adj_records)
bw_sig_df = pd.DataFrame(sig_records)

# Faceted line plot with BW-adjusted occupancy
if not adj_df.empty:
    g = sns.relplot(
        data=adj_df, x="Hour", y="Adj_Prop", hue="Sex_Geno",
        col="Cluster", col_wrap=4, kind="line", marker="o",
        palette=GROUP_COLORS, height=3.5, aspect=1.2,
        linewidth=2.5, markersize=8,
        facet_kws={"sharey": False}, legend=False,
    )
    for ax in g.axes.flatten():
        title_text = ax.get_title().split("= ")[-1].strip()
        c_sig = bw_sig_df[bw_sig_df["Cluster"].astype(str) == title_text]
        y_min, y_max = ax.get_ylim()
        y_rng = y_max - y_min
        ax.set_ylim(y_min, y_max + y_rng * 0.3)
        for hr in all_hours:
            h_row = c_sig[c_sig["Hour"] == hr]
            if h_row.empty:
                continue
            f_star = h_row["Female_Star"].values[0]
            m_star = h_row["Male_Star"].values[0]
            if f_star:
                ax.text(hr, y_max + y_rng * 0.18, f_star,
                        color=GROUP_COLORS["atg7KO-Female"],
                        ha="center", fontsize=11, fontweight="bold")
            if m_star:
                ax.text(hr, y_max + y_rng * 0.06, m_star,
                        color=GROUP_COLORS["atg7KO-Male"],
                        ha="center", fontsize=11, fontweight="bold")
    g.set_axis_labels("Hour", "BW-Adjusted Occupancy", fontsize=12, fontweight="bold")
    g.set_titles("Cluster {col_name}", fontweight="bold", size=14)
    for ax in g.axes.flatten():
        ax.set_xticks(all_hours)
    plt.tight_layout()
    plt.savefig(
        fig_path("GEE_All_Clusters_Hourly_Occupancy_with_Star_FINAL.pdf"),
        bbox_inches="tight"
    )
    plt.show()


# ===========================================================================
# SECTION 10 – TRANSITION ANALYSIS
# ===========================================================================
print("\nSECTION 10: Transition analysis")

df_seq = df_stable.copy()
df_seq["Sex_Geno"]    = df_seq["Geno"] + "-" + df_seq["Sex"]
df_seq["Cluster"]     = df_seq["Cluster"].astype("Int64")
df_seq["Prev_Cluster"] = (
    df_seq.groupby("experimental_id")["Cluster"].shift(1).astype("Int64")
)
df_seq = df_seq.dropna(subset=["Prev_Cluster"])

# --- Hourly transition counts (no self-transitions) ---
def _count_transitions(df):
    trans = (
        df[df["Cluster"] != df["Prev_Cluster"]]
        .groupby(["experimental_id", "Geno", "Sex", "Hour_bin"])
        .size()
        .reset_index(name="n_transitions")
    )
    trans["Sex_Geno"] = trans["Geno"] + "-" + trans["Sex"]
    return trans


hourly_trans_counts = _count_transitions(df_seq)

# Merge BW for transition model
hourly_trans_counts = hourly_trans_counts.merge(
    df_bw[["experimental_id", "BW"]], on="experimental_id", how="left"
)

# GEE Poisson (transitions ~ Sex_Geno * Hour + BW)
model_hourly = smf.gee(
    "n_transitions ~ C(Sex_Geno, Treatment('control-Female')) * Hour_bin + BW",
    groups="experimental_id",
    data=hourly_trans_counts,
    family=sm.families.Poisson(),
    cov_struct=Exchangeable(),
).fit(cov_type="robust")

w_table  = model_hourly.wald_test_terms().table
chi_grp  = w_table.loc["C(Sex_Geno, Treatment('control-Female'))", "statistic"].item()
p_grp    = w_table.loc["C(Sex_Geno, Treatment('control-Female'))", "pvalue"].item()
df_grp   = int(w_table.loc["C(Sex_Geno, Treatment('control-Female'))", "df_constraint"].item())
chi_intr = w_table.loc["C(Sex_Geno, Treatment('control-Female')):Hour_bin", "statistic"].item()
p_intr   = w_table.loc["C(Sex_Geno, Treatment('control-Female')):Hour_bin", "pvalue"].item()
df_intr  = int(w_table.loc["C(Sex_Geno, Treatment('control-Female')):Hour_bin", "df_constraint"].item())

print(f"\nTransition GEE results (BW-adjusted):")
print(f"  Group effect:      χ²({df_grp}) = {chi_grp:.2f}, p = {p_grp:.4e}")
print(f"  Group × Time int.: χ²({df_intr}) = {chi_intr:.2f}, p = {p_intr:.4e}")

# Factorial breakdown: Geno * Sex + Hour
trans_df = hourly_trans_counts.copy()
model_fact = smf.glm(
    "n_transitions ~ C(Geno, Treatment('control')) * C(Sex, Treatment('Female')) + C(Hour_bin)",
    data=trans_df,
    family=sm.families.Poisson(),
).fit()

for label, key in [
    ("Genotype", "C(Geno, Treatment('control'))"),
    ("Sex",      "C(Sex, Treatment('Female'))"),
    ("Interaction", "C(Geno, Treatment('control')):C(Sex, Treatment('Female'))"),
]:
    row = model_fact.wald_test_terms().table.loc[key]
    print(f"  {label}: χ²({int(row['df_constraint'])}) = "
          f"{row['statistic'].item():.2f}, p = {row['pvalue'].item():.4e}")

# Line plot
fig, ax = plt.subplots(figsize=(14, 8))
sns.lineplot(
    data=hourly_trans_counts, x="Hour_bin", y="n_transitions",
    hue="Sex_Geno", palette=GROUP_COLORS, marker="o",
    markersize=10, linewidth=3.8, errorbar="se",
    hue_order=GROUP_ORDER, ax=ax,
)
ax.set_xlabel("Hour", fontsize=18, fontweight="bold")
ax.set_ylabel("Number of transitions", fontsize=18, fontweight="bold")
ax.set_title("Hourly Transitions (BW-adjusted)", fontsize=22, fontweight="bold")
ax.set_xticks(range(1, 13))
sns.despine()
plt.tight_layout()
plt.savefig(fig_path("TRANSITIONS_Final_Poisson_NoSelf.pdf"), bbox_inches="tight")
plt.show()

# --- Transition probability matrices ---
transition_counts = (
    df_seq
    .groupby(["Geno", "Sex", "Prev_Cluster", "Cluster"])
    .size()
    .reset_index(name="count")
    .rename(columns={"Prev_Cluster": "from_cluster", "Cluster": "to_cluster"})
)
transition_counts["Sex_Geno"] = (
    transition_counts["Geno"] + "-" + transition_counts["Sex"]
)

all_cluster_nodes = sorted(df_seq["Cluster"].dropna().unique())
cmap_t20 = load_cmap("Tableau_20", cmap_type="discrete")
cluster_to_color_net = {
    c: cmap_t20(i) for i, c in enumerate(all_cluster_nodes)
}


def _edge_width(w):
    if w <= 3:    return 0.5
    if w <= 30:   return 1.2
    if w <= 200:  return 3.5
    if w <= 500:  return 7.0
    if w <= 1300: return 14.0
    return 22.0


# 2×2 overall transition networks
all_node_incidences = []
for group in GROUP_ORDER:
    sub = transition_counts[transition_counts["Sex_Geno"] == group]
    G_t = nx.DiGraph()
    for _, row in sub.iterrows():
        G_t.add_edge(row["from_cluster"], row["to_cluster"], weight=row["count"])
    for node in G_t.nodes():
        in_w  = sum(d["weight"] for _, _, d in G_t.in_edges(node, data=True))
        out_w = sum(d["weight"] for _, _, d in G_t.out_edges(node, data=True))
        all_node_incidences.append(in_w + out_w)

gnode_min = min(all_node_incidences)
gnode_max = max(all_node_incidences)
MIN_NODE, MAX_NODE = 400, 2800

fig, axes = plt.subplots(2, 2, figsize=(20, 18))
for ax, group in zip(axes.flat, GROUP_ORDER):
    sub = transition_counts[transition_counts["Sex_Geno"] == group]
    G = nx.DiGraph()
    for _, row in sub.iterrows():
        G.add_edge(row["from_cluster"], row["to_cluster"], weight=row["count"])

    node_inc = {
        n: (
            sum(d["weight"] for _, _, d in G.in_edges(n, data=True)) +
            sum(d["weight"] for _, _, d in G.out_edges(n, data=True))
        )
        for n in G.nodes()
    }
    inc_arr = np.array(list(node_inc.values()))
    if gnode_max == gnode_min:
        node_sizes = [MIN_NODE] * len(inc_arr)
    else:
        node_sizes = MIN_NODE + (MAX_NODE - MIN_NODE) * (
            (inc_arr - gnode_min) / (gnode_max - gnode_min)
        )

    edge_ws  = [_edge_width(d["weight"]) for _, _, d in G.edges(data=True)]
    pos      = nx.kamada_kawai_layout(G, scale=2)
    for k_ in pos:
        pos[k_] += np.random.normal(0, 0.7, 2)

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes,
        node_color=[cluster_to_color_net[n] for n in G.nodes()],
        alpha=0.9, ax=ax
    )
    edges_nsl = [(u, v) for u, v in G.edges() if u != v]
    widths_nsl = [_edge_width(G[u][v]["weight"]) for u, v in edges_nsl]
    nx.draw_networkx_edges(
        G, pos, edgelist=edges_nsl, width=widths_nsl,
        alpha=0.7, arrows=True, arrowstyle="-|>", arrowsize=5, ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=18, font_weight="bold", ax=ax)
    ax.set_title(group, fontsize=20)
    ax.axis("off")

plt.suptitle("Cluster Transition Networks – Overall", fontsize=24, y=1.01)
plt.tight_layout()
plt.savefig(
    fig_path("REPORT_cluster_transition_network_OVERALL_atg7KD_chow.pdf"),
    bbox_inches="tight"
)
plt.show()

# --- Hourly transition networks (one PDF per Sex_Geno) ---
hourly_trans = (
    df_seq
    .groupby(["Sex_Geno", "Hour_bin", "Prev_Cluster", "Cluster"])
    .size()
    .reset_index(name="count")
    .rename(columns={"Prev_Cluster": "from_cluster", "Cluster": "to_cluster"})
)

all_inc_h = []
for (_, _), sub in hourly_trans.groupby(["Sex_Geno", "Hour_bin"]):
    G_t = nx.DiGraph()
    for _, r in sub.iterrows():
        G_t.add_edge(r["from_cluster"], r["to_cluster"], weight=r["count"])
    for n in G_t.nodes():
        all_inc_h.append(
            sum(d["weight"] for _, _, d in G_t.in_edges(n, data=True)) +
            sum(d["weight"] for _, _, d in G_t.out_edges(n, data=True))
        )

gnode_max_h = max(all_inc_h) if all_inc_h else 1

# Fixed layout from global graph
G_global = nx.DiGraph()
for _, row in hourly_trans.iterrows():
    G_global.add_edge(row["from_cluster"], row["to_cluster"])
fixed_pos = nx.kamada_kawai_layout(G_global, scale=4.0)

hours_h = sorted(hourly_trans["Hour_bin"].unique())
n_cols  = 3
n_rows  = int(np.ceil(len(hours_h) / n_cols))

for sg in GROUP_ORDER:
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.8 * n_rows),
                             squeeze=False)
    fig.suptitle(f"Cluster Transitions — {sg}", fontsize=22, y=0.98)

    for ax, hour in zip(axes.flat, hours_h):
        sub = hourly_trans[
            (hourly_trans["Sex_Geno"] == sg) &
            (hourly_trans["Hour_bin"] == hour)
        ]
        ax.set_title(f"Hour {hour}", fontsize=14)
        if sub.empty:
            ax.axis("off")
            continue
        G = nx.DiGraph()
        for _, r in sub.iterrows():
            G.add_edge(r["from_cluster"], r["to_cluster"], weight=r["count"])

        node_inc_h = {
            n: (
                sum(d["weight"] for _, _, d in G.in_edges(n, data=True)) +
                sum(d["weight"] for _, _, d in G.out_edges(n, data=True))
            )
            for n in G.nodes()
        }
        inc_v    = np.array(list(node_inc_h.values()))
        nsizes_h = 400 + (1200 - 400) * ((inc_v / gnode_max_h) ** 1.5)

        nx.draw_networkx_nodes(
            G, fixed_pos, nodelist=list(G.nodes()),
            node_size=nsizes_h,
            node_color=[cluster_to_color_net[n] for n in G.nodes()],
            alpha=0.9, ax=ax
        )
        edges_h  = [(u, v) for u, v in G.edges() if u != v]
        widths_h = [_edge_width(G[u][v]["weight"]) for u, v in edges_h]
        nx.draw_networkx_edges(
            G, fixed_pos, edgelist=edges_h, width=widths_h,
            edge_color="black", alpha=1.0,
            arrows=True, arrowstyle="-|>", arrowsize=20, ax=ax
        )
        nx.draw_networkx_labels(G, fixed_pos, font_size=14,
                                font_weight="bold", ax=ax)
        ax.axis("off")

    for ax in axes.flat[len(hours_h):]:
        ax.axis("off")
    plt.tight_layout()
    fname = (
        f"REPORT_cluster_transition_network_"
        f"{sg.replace('-', '_').replace('atg7KO', 'atg7KD')}_chow.pdf"
    )
    plt.savefig(fig_path(fname), bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ===========================================================================
# SECTION 11 – DWELL TIME ANALYSIS
# ===========================================================================
print("\nSECTION 11: Dwell time analysis")


def _dwell_times(df_sub):
    df_sub = df_sub.sort_values("Interval_bin")
    seq = df_sub["Cluster"].values
    current, length, dwells = seq[0], 1, []
    for cl in seq[1:]:
        if cl == current:
            length += 1
        else:
            dwells.append((current, length))
            current, length = cl, 1
    dwells.append((current, length))
    return dwells


dwell_records = []
for (eid, sex, geno), group_df in df_stable[df_stable["Cluster"] != -1].groupby(
    ["experimental_id", "Sex", "Geno"]
):
    for cl, length in _dwell_times(group_df):
        dwell_records.append({
            "experimental_id": eid,
            "Sex": sex, "Geno": geno,
            "Cluster": cl,
            "dwell_time": length,
        })

df_dwell = pd.DataFrame(dwell_records)
df_dwell["log_dwell"] = np.log1p(df_dwell["dwell_time"])
df_dwell["Sex_Geno"]  = df_dwell["Geno"] + "-" + df_dwell["Sex"]

dwell_stats = []
for cid in sorted(df_dwell["Cluster"].unique()):
    dc = df_dwell[df_dwell["Cluster"] == cid]
    try:
        m = smf.mixedlm(
            "log_dwell ~ C(Sex_Geno, Treatment('control-Female'))",
            data=dc,
            groups=dc["experimental_id"],
        ).fit()
        for term in m.params.index:
            if term in ("Intercept", "Group Var"):
                continue
            grp = term.split("[T.")[-1].rstrip("]")
            dwell_stats.append({
                "Cluster": cid,
                "Group": grp,
                "Coef": m.params[term],
                "P": m.pvalues[term],
            })
    except Exception as e:
        print(f"  Dwell Cluster {cid}: {e}")

dwell_stats_df = pd.DataFrame(dwell_stats)
dwell_stats_df["Star"] = dwell_stats_df["P"].apply(_get_symbol)

# Boxplot
fig, ax = plt.subplots(figsize=(14, 6))
order = GROUP_ORDER
sns.boxplot(
    data=df_dwell, x="Cluster", y="log_dwell", hue="Sex_Geno",
    palette=GROUP_COLORS, hue_order=order,
    flierprops=dict(marker="o", markersize=3), ax=ax
)
ax.set_ylabel("Log dwell time (frames)", fontsize=14)
ax.set_xlabel("Cluster", fontsize=14)
ax.set_title("Dwell Time per Cluster (Mixed-effects model)",
             fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(fig_path("REPORT_dwellTimes_perCluster_atg7KD_chow.pdf"),
            bbox_inches="tight")
plt.show()


# ===========================================================================
# SECTION 12 – ENTROPY & BEHAVIOURAL COMPLEXITY
# ===========================================================================
print("\nSECTION 12: Entropy & complexity")

df_ent = df_stable.copy()
df_ent["Sex_Geno"] = df_ent["Geno"] + "-" + df_ent["Sex"]

# Shannon entropy per animal per hour
entropy_df = (
    df_ent[df_ent["Cluster"] != -1]
    .groupby(["experimental_id", "Sex_Geno", "Hour_bin"])["Cluster"]
    .value_counts(normalize=True)
    .groupby(level=[0, 1, 2])
    .apply(lambda x: sp_entropy(x.values))
    .reset_index(name="entropy")
)

# Transition entropy (complexity score) per animal per hour
complexity_records = []
for (eid, sg, hr), sub in (
    df_ent[df_ent["Cluster"] != -1]
    .groupby(["experimental_id", "Sex_Geno", "Hour_bin"])
):
    seq = sub.sort_values("Interval_bin")["Cluster"].values
    if len(seq) < 2:
        continue
    clust_ids = sorted(set(seq))
    n_c = len(clust_ids)
    c2i = {c: i for i, c in enumerate(clust_ids)}
    counts = np.zeros((n_c, n_c))
    for a, b in zip(seq[:-1], seq[1:]):
        counts[c2i[a], c2i[b]] += 1
    row_s = counts.sum(axis=1, keepdims=True)
    T = np.divide(counts, row_s, out=np.zeros_like(counts), where=row_s != 0)
    stat_dist = counts.sum(axis=1) / counts.sum()
    h_rate = -np.nansum(
        stat_dist[:, None] * np.where(T > 0, T * np.log2(T + 1e-12), 0)
    )
    complexity_records.append({
        "experimental_id": eid,
        "Sex_Geno": sg,
        "Hour_bin": hr,
        "complexity_score": h_rate,
    })

complexity_df = pd.DataFrame(complexity_records)

# GEE for complexity
if not complexity_df.empty:
    comp_gee = smf.gee(
        "complexity_score ~ C(Sex_Geno, Treatment('control-Female')) * Hour_bin",
        groups="experimental_id",
        data=complexity_df,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        cov_struct=Exchangeable(),
    ).fit(cov_type="robust")
    print(comp_gee.summary())

# Bar plot (overall complexity)
fig, ax = plt.subplots(figsize=(12, 6))
comp_overall = (
    complexity_df
    .groupby(["experimental_id", "Sex_Geno"])["complexity_score"]
    .mean()
    .reset_index()
)
order = ["control-Female", "atg7KO-Female", "control-Male", "atg7KO-Male"]
sns.barplot(
    data=comp_overall, x="Sex_Geno", y="complexity_score",
    order=order, palette=GROUP_COLORS,
    errorbar="se", edgecolor="black", ax=ax,
)
sns.stripplot(
    data=comp_overall, x="Sex_Geno", y="complexity_score",
    order=order, color="navy", jitter=0.15, size=5, ax=ax,
)
ax.set_ylabel("Mean complexity score (bits/transition)", fontsize=14)
ax.set_xlabel("")
ax.set_title("Behavioural Complexity (Transition Entropy)",
             fontsize=16, fontweight="bold")
sns.despine()
plt.tight_layout()
plt.savefig(fig_path("Complexity_Barplot_Final_ENTROPY_atg7KD_chow.pdf"),
            bbox_inches="tight")
plt.show()


# ===========================================================================
# SECTION 13 – MARKOV CHAIN MODELLING
# ===========================================================================
print("\nSECTION 13: Markov chain modelling")

df_mc = df_stable.copy()
df_mc["Sex_Geno"] = df_mc["Geno"] + "-" + df_mc["Sex"]
df_mc = df_mc.sort_values(["experimental_id", "Hour_bin"])

if pd.api.types.is_categorical_dtype(df_mc["Cluster"]):
    mc_clusters = df_mc["Cluster"].cat.categories
else:
    mc_clusters = pd.Index(sorted(df_mc["Cluster"].dropna().unique()))
n_mc = len(mc_clusters)

animal_matrices = {}
for animal, sub in df_mc.groupby("experimental_id"):
    states = sub["Cluster"].values
    counts = np.zeros((n_mc, n_mc))
    for s1, s2 in zip(states[:-1], states[1:]):
        i = mc_clusters.get_loc(s1)
        j = mc_clusters.get_loc(s2)
        counts[i, j] += 1
    row_s = counts.sum(axis=1, keepdims=True)
    prob_mat = np.divide(counts, row_s, out=np.zeros_like(counts), where=row_s != 0)
    animal_matrices[animal] = prob_mat

# Tidy dataframe
mc_records = [
    {
        "experimental_id": animal,
        "Sex_Geno": df_mc.loc[df_mc["experimental_id"] == animal, "Sex_Geno"].iloc[0],
        "from_cluster": mc_clusters[i],
        "to_cluster":   mc_clusters[j],
        "prob": animal_matrices[animal][i, j],
    }
    for animal in animal_matrices
    for i in range(n_mc)
    for j in range(n_mc)
]
trans_mc_df = pd.DataFrame(mc_records)

# Mixed-effects model
mc_model = smf.mixedlm(
    "prob ~ Sex_Geno", data=trans_mc_df, groups=trans_mc_df["experimental_id"]
).fit()
print(mc_model.summary())

# Per-Sex_Geno average heatmaps
vmax_global = trans_mc_df["prob"].max()
custom_colors = {
    "control-Male":   "#000000",
    "control-Female": "#686767",
    "atg7KO-Male":    "#444B29",
    "atg7KO-Female":  "#A44316",
}

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
for ax, sg in zip(axes.flat, GROUP_ORDER):
    sub = trans_mc_df[trans_mc_df["Sex_Geno"] == sg]
    mat = (
        sub.groupby(["from_cluster", "to_cluster"])["prob"]
        .mean()
        .reset_index()
        .pivot(index="from_cluster", columns="to_cluster", values="prob")
    )
    color = custom_colors[sg]
    cmap_sg = mcolors.LinearSegmentedColormap.from_list(
        f"cmap_{sg}", ["white", color]
    )
    sns.heatmap(
        mat, annot=True, fmt=".2f", cmap=cmap_sg,
        vmin=0, vmax=vmax_global,
        cbar_kws={"label": "Probability"},
        annot_kws={"fontsize": 14, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Transition Probabilities: {sg}", fontsize=16, weight="bold")
    ax.set_xlabel("To Cluster", fontsize=13)
    ax.set_ylabel("From Cluster", fontsize=13)

plt.tight_layout()
plt.savefig(
    fig_path("transition_probability_matrices_atg7KD_chow_2x2.pdf"),
    bbox_inches="tight"
)
plt.show()


# ===========================================================================
# SECTION 14 – CLUSTER VALIDATION METRICS
# ===========================================================================
print("\nSECTION 14: Cluster validation (silhouette + PERMANOVA)")

feature_cols_val = [
    c for c in feature_columns_active if c in df_stable.columns
]
n_samp  = min(len(df_stable), 1000)
df_sub  = df_stable.sample(n=n_samp, random_state=42)
sil_score = silhouette_score(
    df_sub[feature_cols_val], df_sub["Cluster"], metric="euclidean"
)
print(f"Global Silhouette Score: {sil_score:.4f}")

n_per_cluster = 100
clusters_val  = sorted(df_stable["Cluster"].unique())
perm_results  = []
print(f"Running Pairwise PERMANOVA for {len(clusters_val)} clusters...")

for c1, c2 in combinations(clusters_val, 2):
    c1d = df_stable[df_stable["Cluster"] == c1]
    c2d = df_stable[df_stable["Cluster"] == c2]
    pair_sub = pd.concat([
        c1d.sample(n=min(len(c1d), n_per_cluster), random_state=42),
        c2d.sample(n=min(len(c2d), n_per_cluster), random_state=42),
    ]).reset_index(drop=True)
    ids = [str(i) for i in range(len(pair_sub))]
    try:
        dm  = DistanceMatrix(
            pdist(pair_sub[feature_cols_val], metric="braycurtis"), ids=ids
        )
        res = permanova(
            dm, grouping=pair_sub["Cluster"].astype(str).tolist(),
            permutations=999
        )
        perm_results.append({
            "Cluster_A": c1, "Cluster_B": c2,
            "Pseudo_F": res["test statistic"],
            "P_Value":  res["p-value"],
        })
    except Exception as e:
        print(f"  PERMANOVA {c1} vs {c2}: {e}")

perm_df = pd.DataFrame(perm_results)
if not perm_df.empty:
    perm_df["P_Adj"]     = (perm_df["P_Value"] * len(perm_df)).clip(upper=1.0)
    perm_df["Significant"] = perm_df["P_Adj"] < 0.05
    print("\n--- Pairwise PERMANOVA (Bonferroni-adjusted) ---")
    print(perm_df[["Cluster_A", "Cluster_B", "Pseudo_F", "P_Adj", "Significant"]])


print("\n" + "=" * 70)
print("Pipeline complete. Figures saved to:", FIGURES_DIR)
print("=" * 70)
