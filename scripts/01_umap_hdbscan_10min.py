"""
Unsupervised Behavioural Clustering — 10-Minute Open-Field Recordings
======================================================================
Pipeline for dimensionality reduction and unsupervised clustering of
DeepOF behavioural output from short (≤10 min) rodent open-field tests.

Pipeline overview
-----------------
1.  Load male and female DeepOF CSVs; combine into a single DataFrame.
2.  Bin frames into 2-second intervals; compute mean behaviour per bin.
3.  Filter animals and time range; drop excluded IDs and conditions.
4.  Multiple imputation of missing values (IterativeImputer).
5.  Standard scaling of behavioural features.
6.  UMAP dimensionality reduction (2D embedding).
7.  HDBSCAN density-based clustering.
8.  Cluster validation (Silhouette, Davies–Bouldin, Calinski–Harabasz, DBCV).
9.  Behavioural profiling per cluster (polar plot, heatmap).
10. Cluster composition analysis (Chi-square, Cramér's V, PERMANOVA).
11. Kruskal–Wallis + Dunn post-hoc tests across clusters.
12. Cluster proportion modelling (OLS regression, GAM splines).
13. Cluster transition analysis (transition matrices, network graphs).
14. Behavioural entropy per animal.
15. Semi-Markov modelling of dwell times.
16. Temporal dynamics (polar plot of cluster representation over time).

Usage
-----
    python 01_umap_hdbscan_10min.py

Edit the PARAMETERS block below to match your data paths and preferences.

Dependencies
------------
    numpy, pandas, scikit-learn, umap-learn, hdbscan, matplotlib,
    seaborn, scipy, statsmodels, scikit-posthocs, scikit-bio,
    networkx, pypalettes, statannotations

Author  : <your name>
Date    : 2025
License : MIT
"""

# ── Thread control (must come before numpy import) ────────────────────────────
import os
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ── Standard library ──────────────────────────────────────────────────────────
import random
import math

# ── Scientific stack ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer          # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import umap.umap_ as umap
import hdbscan
from hdbscan.validity import validity_index

from scipy.stats import kruskal, chi2_contingency, entropy
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines

import scikit_posthocs as sp
from skbio.stats.distance import permanova, DistanceMatrix
import networkx as nx
from pypalettes import load_cmap
from statannotations.Annotator import Annotator


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS  ← edit here
# ══════════════════════════════════════════════════════════════════════════════

# Input data paths
FILE_FEMALES = "/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/chow/females/atg7KO/atg7KO_females_chow_SocialOF/master_combined_females_atg7KO_chow_SocialOF_FINAL.csv"
FILE_MALES   = "/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/chow/males/atg7KO/master_combined_males_atg7KO_chow_SocialOF_FINAL.csv"

# Body weight files (for regression models)
FILE_BW_FEMALES = "/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/chow/females/atg7KO/atg7KO_females_chow_SocialOF/bw_atg7KO_chow.csv"
FILE_BW_MALES   = "/Users/veronika/ownCloud/PCA_analysis2025/SocialOF/chow/males/atg7KO/bw_atg7KO_chow_males.csv"

# Filtering
EXCLUDE_IDS    = ['ID63', 'ID214']       # animal IDs to exclude
EXCLUDE_GENOS  = ['atg7OE']             # genotypes to exclude
LAST_INTERVAL  = "0 days 00:09:58 - 0 days 00:10:00"  # last time bin to include

# Binning
INTERVAL_SEC = 2   # bin width in seconds

# Behavioural features
BEHAVIOUR_COLS = [
    'B_W_nose2nose', 'B_W_sidebyside', 'B_W_sidereside', 'B_W_nose2tail',
    'B_W_nose2body', 'B_W_following', 'B_climb-arena', 'B_sniff-arena',
    'B_immobility', 'B_stat-lookaround', 'B_stat-active', 'B_stat-passive',
    'B_moving', 'B_sniffing', 'B_speed'
]

# Human-readable label mapping for plotting
LABEL_MAP = {
    "B_W_following":    "following",
    "B_stat-active":    "stat-active",
    "B_W_nose2body":    "nose2body",
    "B_W_sidebyside":   "sidebyside",
    "B_sniff-arena":    "sniff-arena",
    "B_immobility":     "immobility",
    "B_sniffing":       "sniffing",
    "B_climb-arena":    "climb-arena",
    "B_W_nose2nose":    "nose2nose",
    "B_W_sidereside":   "sidereside",
    "B_W_nose2tail":    "nose2tail",
    "B_stat-passive":   "stat-passive",
    "B_moving":         "moving",
    "B_stat-lookaround":"stat-lookaround",
    "B_speed":          "speed",
}

# UMAP parameters
UMAP_PARAMS = dict(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric='euclidean',
    verbose=True,
    random_state=42,
)

# HDBSCAN parameters
HDBSCAN_PARAMS = dict(
    min_cluster_size=500,
    min_samples=90,
)

# Colour palette per experimental group
GROUP_COLOURS = {
    'Female_control': '#9D9D9D',
    'Female_atg7KO':  '#A44316',
    'Male_control':   '#000000',
    'Male_atg7KO':    '#444B29',
}

# Group order for plotting
GROUP_ORDER = ['Female_control', 'Female_atg7KO', 'Male_control', 'Male_atg7KO']
GROUP_LABELS = ['control Female', 'atg7KO Female', 'control Male', 'atg7KO Male']

# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def load_and_combine(file_females: str, file_males: str) -> pd.DataFrame:
    """Load male and female DeepOF CSVs and concatenate into one DataFrame."""
    df_f = pd.read_csv(file_females)
    df_m = pd.read_csv(file_males)
    df_f['Sex'] = 'Female'
    df_m['Sex'] = 'Male'
    df = pd.concat([df_f, df_m], ignore_index=True)
    df['Time'] = pd.to_timedelta(df['Time'])
    print(f"Loaded {len(df):,} rows — {df['experimental_id'].nunique()} animals.")
    return df


def bin_behaviours(df: pd.DataFrame, interval_sec: int, behaviour_cols: list) -> pd.DataFrame:
    """
    Assign each frame to a fixed-width time bin, then compute mean
    behavioural scores per (animal × bin).

    Parameters
    ----------
    df            : Raw combined DataFrame with a 'Time' timedelta column.
    interval_sec  : Bin width in seconds.
    behaviour_cols: Column names to aggregate.

    Returns
    -------
    Aggregated DataFrame with 'Interval_bin' and 'Interval_label' columns added.
    """
    df = df.copy()
    df['Interval_bin'] = (df['Time'].dt.total_seconds() // interval_sec).astype(int)
    df['Interval_start'] = pd.to_timedelta(df['Interval_bin'] * interval_sec, unit='s')
    df['Interval_end']   = pd.to_timedelta((df['Interval_bin'] + 1) * interval_sec, unit='s')
    df['Interval_label'] = df['Interval_start'].astype(str) + ' - ' + df['Interval_end'].astype(str)

    agg = (
        df.groupby(['Interval_bin', 'Interval_label', 'Sex', 'experimental_id', 'Geno'])
        [behaviour_cols]
        .mean()
        .reset_index()
    )
    print(f"Binned into {agg['Interval_bin'].nunique()} × {interval_sec}-second intervals.")
    return agg


def filter_and_impute(
    agg: pd.DataFrame,
    exclude_ids: list,
    exclude_genos: list,
    last_label: str,
    behaviour_cols: list,
) -> pd.DataFrame:
    """
    Filter out excluded animals / genotypes / late time bins, sort deterministically,
    then apply multiple imputation (IterativeImputer) to behavioural features.

    Returns the imputed DataFrame.
    """
    filtered = (
        agg[
            (~agg['experimental_id'].isin(exclude_ids)) &
            (~agg['Geno'].isin(exclude_genos)) &
            (agg['Interval_label'] <= last_label)
        ]
        .copy()
        .sort_values(['experimental_id', 'Interval_label'])
        .reset_index(drop=True)
    )

    # Columns used for imputation (exclude metadata and speed)
    impute_cols = [c for c in filtered.columns
                   if c not in ['Interval_label', 'experimental_id',
                                 'Interval_bin', 'B_speed', 'Geno', 'Sex']]

    imputer = IterativeImputer(random_state=42)
    df_imputed = filtered.copy()
    df_imputed[impute_cols] = imputer.fit_transform(filtered[impute_cols])

    n_missing_before = filtered[impute_cols].isnull().sum().sum()
    print(f"Imputed {n_missing_before} missing values across {len(impute_cols)} features.")
    return df_imputed, impute_cols


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DIMENSIONALITY REDUCTION & CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────

def run_umap(df_imputed: pd.DataFrame, feature_cols: list, umap_params: dict) -> np.ndarray:
    """Scale features and compute UMAP 2D embedding."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_imputed[feature_cols])

    reducer = umap.UMAP(**umap_params)
    embedding = reducer.fit_transform(X_scaled)
    print(f"UMAP embedding shape: {embedding.shape}")
    return embedding


def run_hdbscan(embedding: np.ndarray, hdbscan_params: dict) -> tuple[np.ndarray, hdbscan.HDBSCAN]:
    """Cluster the UMAP embedding with HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    labels = clusterer.fit_predict(embedding)
    unique, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        tag = "noise" if lbl == -1 else f"cluster {lbl}"
        print(f"  {tag}: {cnt:,} frames ({cnt/len(labels):.1%})")
    return labels, clusterer


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CLUSTER VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    clusterer: hdbscan.HDBSCAN,
) -> dict:
    """
    Compute standard internal clustering validation metrics.

    Returns a dict with Silhouette, Davies–Bouldin, Calinski–Harabasz,
    DBCV, noise proportion, and mean membership probability.
    """
    mask = labels != -1
    X_c, y_c = embedding[mask], labels[mask]

    metrics = {}

    # DBCV (HDBSCAN-native)
    metrics['DBCV'] = validity_index(np.asarray(embedding, dtype=np.float64), clusterer.labels_)

    if len(np.unique(y_c)) > 1:
        metrics['Silhouette']         = silhouette_score(X_c, y_c)
        metrics['Davies_Bouldin']     = davies_bouldin_score(X_c, y_c)
        metrics['Calinski_Harabasz']  = calinski_harabasz_score(X_c, y_c)

    metrics['noise_proportion'] = np.mean(labels == -1)
    if hasattr(clusterer, 'probabilities_'):
        metrics['mean_membership_prob'] = np.mean(clusterer.probabilities_)

    print("\n── Cluster validation ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — BEHAVIOURAL PROFILING
# ──────────────────────────────────────────────────────────────────────────────

def build_cluster_colour_map(cluster_labels: list) -> dict:
    """Assign a distinct colour to each cluster using the Tableau_10 palette."""
    cmap = load_cmap("Tableau_10", cmap_type='discrete')
    return {c: cmap(i) for i, c in enumerate(cluster_labels)}


def plot_umap_embedding(
    embedding: np.ndarray,
    df_imputed: pd.DataFrame,
    cluster_colour_map: dict,
    save_path: str = None,
) -> None:
    """Scatter plot of the UMAP embedding coloured by cluster and shaped by genotype."""
    mask = df_imputed['Cluster'] != -1
    emb  = embedding[mask]
    meta = df_imputed[mask].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.set_style("ticks")

    colour_str_map = {str(int(k)): v for k, v in cluster_colour_map.items()}

    sns.scatterplot(
        x=emb[:, 0], y=emb[:, 1],
        hue=meta['Cluster'].astype(str),
        style=meta['Geno'],
        palette=colour_str_map,
        alpha=0.7, ax=ax,
    )

    ax.set_xlabel('UMAP 1', fontsize=25, weight='bold')
    ax.set_ylabel('UMAP 2', fontsize=25, weight='bold')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis='both', labelsize=22)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=20,
              title='', frameon=False, markerscale=2.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_polar_behavioural_profile(
    df_no_outliers: pd.DataFrame,
    behaviour_cols_renamed: list,
    cluster_colour_map: dict,
    save_path: str = None,
) -> None:
    """Radar / polar chart of mean behavioural scores per cluster."""
    cluster_summary = df_no_outliers.groupby('Cluster')[behaviour_cols_renamed].mean()

    N      = len(behaviour_cols_renamed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    for cluster in cluster_summary.index:
        values = cluster_summary.loc[cluster].tolist() + [cluster_summary.loc[cluster].tolist()[0]]
        ax.plot(angles, values, label=f'Cluster {cluster}',
                color=cluster_colour_map[cluster], linewidth=2.5)
        ax.fill(angles, values, color=cluster_colour_map[cluster], alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(behaviour_cols_renamed, color='black', size=23, weight='bold')

    for label, angle in zip(ax.get_xticklabels(), angles):
        deg = np.degrees(angle)
        label.set_horizontalalignment('left' if (deg >= 270 or deg <= 90) else 'right')
        label.set_rotation(deg)
        label.set_rotation_mode('anchor')

    ax.set_rmax(1.0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_cluster_heatmap(
    df_no_outliers: pd.DataFrame,
    behaviour_cols_renamed: list,
    cluster_labels_dict: dict = None,
    save_path: str = None,
) -> None:
    """
    Heatmap of mean behavioural feature values per cluster.

    Parameters
    ----------
    cluster_labels_dict : optional dict mapping cluster int → descriptive string label.
    """
    cluster_summary = df_no_outliers.groupby('Cluster')[behaviour_cols_renamed].mean().round(2)

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.heatmap(cluster_summary, annot=True, cmap='viridis', ax=ax,
                annot_kws={'size': 12})

    y_labels = (
        [cluster_labels_dict[i] for i in cluster_summary.index]
        if cluster_labels_dict else [str(i) for i in cluster_summary.index]
    )

    ax.set_xticklabels(behaviour_cols_renamed, rotation=40, ha='right',
                        color='black', size=16, weight='bold')
    ax.set_yticklabels(y_labels, rotation=0, color='black', size=16, weight='bold')
    ax.set_title('Cluster Behavioural Profiles', size=18, weight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CLUSTER COMPOSITION STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def chi_square_cluster_composition(df_no_outliers: pd.DataFrame, factor: str) -> None:
    """
    Chi-square test + Cramér's V for overall cluster × factor association,
    then per-cluster tests of in-cluster vs. out-of-cluster proportions.
    """
    print(f"\n══ Chi-square: Cluster × {factor} ══════════════════")

    # Overall
    ct = pd.crosstab(df_no_outliers['Cluster'], df_no_outliers[factor])
    chi2, p, dof, _ = chi2_contingency(ct)
    n   = ct.values.sum()
    v   = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    print(f"  Overall  χ²={chi2:.2f}  p={p:.4g}  Cramér's V={v:.3f}")

    # Per-cluster
    for cluster in sorted(df_no_outliers['Cluster'].unique()):
        mask = df_no_outliers['Cluster'] == cluster
        ct_sub = pd.crosstab(mask, df_no_outliers[factor])
        if ct_sub.shape[0] > 1 and ct_sub.shape[1] > 1:
            chi2_c, p_c, _, _ = chi2_contingency(ct_sub)
            sig = "✓ significant" if p_c < 0.05 else "  not significant"
            print(f"  Cluster {cluster}  p={p_c:.4g}  {sig}")


def permanova_cluster_factor(
    df_no_outliers: pd.DataFrame,
    behaviour_cols: list,
    factor: str,
    n_permutations: int = 999,
) -> pd.DataFrame:
    """
    PERMANOVA testing whether behavioural centroid differs by `factor`
    within each cluster. Returns a results DataFrame with FDR correction.
    """
    results = []
    for cid in sorted(df_no_outliers['Cluster'].unique()):
        sub = df_no_outliers[df_no_outliers['Cluster'] == cid].copy()
        sub.index = sub.index.astype(str)
        if sub[factor].nunique() < 2:
            continue
        X   = sub[behaviour_cols].values
        dm  = DistanceMatrix(squareform(pdist(X, 'euclidean')), ids=list(sub.index))
        res = permanova(dm, sub.loc[list(sub.index), factor], permutations=n_permutations)
        results.append({
            'Cluster': cid, 'Factor': factor,
            'pseudo-F': res['test statistic'],
            'p-value': res['p-value'], 'n': len(sub),
        })

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        _, p_fdr, _, _ = multipletests(df_res['p-value'], method='fdr_bh')
        df_res['p-value_FDR'] = p_fdr
        df_res['significant'] = p_fdr < 0.05
    print(f"\n── PERMANOVA ({factor}) ──────────────────────────────")
    print(df_res.to_string(index=False))
    return df_res


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — KRUSKAL–WALLIS + POST-HOC DUNN
# ──────────────────────────────────────────────────────────────────────────────

def kruskal_wallis_features(
    df_no_outliers: pd.DataFrame,
    behaviour_cols: list,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Kruskal–Wallis H-test + FDR correction across clusters for each
    behavioural feature. Also computes η² effect size.
    """
    results = []
    for col in behaviour_cols:
        groups = [
            g[col].dropna().values
            for _, g in df_no_outliers.groupby('Cluster')
            if _ != -1
        ]
        if len(groups) < 2:
            continue
        H, p   = kruskal(*groups)
        n, k   = sum(len(g) for g in groups), len(groups)
        eta2   = (H - k + 1) / (n - k)
        results.append({'feature': col, 'H': H, 'p_raw': p, 'eta_squared': eta2})

    kw_df = pd.DataFrame(results)
    reject, p_fdr, _, _ = multipletests(kw_df['p_raw'], alpha=alpha, method='fdr_bh')
    kw_df['p_fdr']           = p_fdr
    kw_df['significant_fdr'] = reject
    kw_df['eta_class']       = pd.cut(
        kw_df['eta_squared'],
        bins=[-np.inf, 0.01, 0.06, 0.14, np.inf],
        labels=["negligible", "small", "medium", "large"],
    )
    print("\n── Kruskal–Wallis results (FDR-corrected) ──────────")
    for _, row in kw_df.iterrows():
        tag = "SIGNIFICANT" if row['significant_fdr'] else "n.s."
        print(f"  {row['feature']}: H={row['H']:.2f}  p_FDR={row['p_fdr']:.4g}"
              f"  η²={row['eta_squared']:.3f} ({tag})")
    return kw_df


def dunn_posthoc(
    df_no_outliers: pd.DataFrame,
    behaviour_cols: list,
    kw_df: pd.DataFrame,
    p_adjust: str = 'fdr_bh',
) -> dict:
    """Run Dunn's post-hoc test for features significant after KW + FDR."""
    sig_features = kw_df.loc[kw_df['significant_fdr'], 'feature']
    dunn_results = {}
    for col in sig_features:
        sub = df_no_outliers[df_no_outliers['Cluster'] != -1]
        dunn_results[col] = sp.posthoc_dunn(
            sub, val_col=col, group_col='Cluster', p_adjust=p_adjust
        )
    return dunn_results


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — CLUSTER PROPORTION MODELLING (OLS + GAM)
# ──────────────────────────────────────────────────────────────────────────────

def compute_cluster_proportions(
    df_no_outliers: pd.DataFrame,
    df_bw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-animal cluster proportions and merge with body-weight data.
    """
    counts = (
        df_no_outliers
        .groupby(['experimental_id', 'Cluster'])
        .size()
        .reset_index(name='count')
    )
    totals = (
        df_no_outliers
        .groupby('experimental_id')
        .size()
        .reset_index(name='total')
    )
    props = counts.merge(totals, on='experimental_id')
    props['cluster_prop'] = props['count'] / props['total']

    df_bw = df_bw.copy()
    df_bw['experimental_id'] = df_bw['experimental_id'].astype(str)
    props['experimental_id'] = props['experimental_id'].astype(str)

    return props.merge(df_bw, on='experimental_id', how='left')


def ols_cluster_models(df_merged: pd.DataFrame) -> None:
    """Fit OLS (cluster_prop ~ Geno + BW + Sex) for each cluster."""
    print("\n── OLS regression: cluster_prop ~ Geno + BW + Sex ──")
    for cid in sorted(df_merged['Cluster'].unique()):
        sub = df_merged[df_merged['Cluster'] == cid]
        model = ols('cluster_prop ~ Geno + BW + Sex', data=sub).fit()
        print(f"\nCluster {cid}:")
        print(model.summary().tables[1])


def gam_cluster_models(df_merged: pd.DataFrame) -> None:
    """Fit GAM with BSpline on BW for each cluster."""
    print("\n── GAM: cluster_prop ~ Geno + Sex + s(BW) ──────────")
    for cid in sorted(df_merged['Cluster'].unique()):
        sub = df_merged[df_merged['Cluster'] == cid]
        exog = pd.get_dummies(sub[['Geno', 'Sex']], drop_first=True)
        bs   = BSplines(sub[['BW']], df=[6], degree=[3])
        model = GLMGam(sub['cluster_prop'], exog=exog, smoother=bs).fit()
        print(f"\nCluster {cid}:")
        print(model.summary().tables[1])


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — CLUSTER TRANSITION ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def compute_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long-format DataFrame of consecutive cluster transitions
    with metadata (experimental_id, Geno, Sex).
    """
    df = df.sort_values(['experimental_id', 'Interval_bin'])
    records = []
    for exp_id, sub in df.groupby('experimental_id'):
        geno, sex = sub['Geno'].iloc[0], sub['Sex'].iloc[0]
        pairs = list(zip(sub['Cluster'], sub['Cluster'].shift(-1).dropna()))
        records.extend(
            [(exp_id, geno, sex, int(a), int(b)) for a, b in pairs[:-1]]
        )
    return pd.DataFrame(records, columns=['experimental_id', 'Geno', 'Sex',
                                           'from_cluster', 'to_cluster'])


def plot_transition_heatmaps(
    transition_counts: pd.DataFrame,
    group_col: str = 'Sex_Geno',
    save_path: str = None,
) -> None:
    """4-panel heatmap of cluster transition counts for each group."""
    group_order = ['Female-control', 'Female-atg7KO', 'Male-control', 'Male-atg7KO']
    overall_max = transition_counts['count'].max()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)

    for ax, group in zip(axes.flat, group_order):
        sub = transition_counts[transition_counts[group_col] == group]
        matrix = sub.pivot(index='from_cluster', columns='to_cluster',
                            values='count').fillna(0)
        sns.heatmap(matrix, annot=True, fmt='g', cmap='viridis',
                    ax=ax, vmin=0, vmax=overall_max, cbar=False)
        ax.set_title(group, fontsize=18, weight='bold', pad=15)
        ax.set_ylabel('From Cluster', size=18)
        ax.set_xlabel('To Cluster', size=18)
        ax.tick_params(left=False, bottom=False)
        plt.setp(ax.get_xticklabels(), fontsize=16, weight='bold')
        plt.setp(ax.get_yticklabels(), fontsize=16, weight='bold')

    # Shared colourbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    sm = mpl.cm.ScalarMappable(
        cmap='viridis',
        norm=mpl.colors.Normalize(vmin=0, vmax=overall_max)
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax).set_ylabel('Transition Count', fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_transition_networks(
    transition_counts: pd.DataFrame,
    cluster_colour_map: dict,
    group_col: str = 'Sex_Geno',
    save_path: str = None,
) -> None:
    """2×2 directed network graph of cluster transitions per group."""
    group_order = ['Female-control', 'Female-atg7KO', 'Male-control', 'Male-atg7KO']
    titles = [
        '$\\bf{Female\\ Control}$', '$\\bf{Female\\ atg7KO}$',
        '$\\bf{Male\\ Control}$',   '$\\bf{Male\\ atg7KO}$',
    ]

    all_weights = transition_counts['count'].values
    g_min, g_max = all_weights.min(), all_weights.max()
    min_w, max_w = 1, 90

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for ax, group, title in zip(axes.flat, group_order, titles):
        sub = transition_counts[transition_counts[group_col] == group]
        G = nx.DiGraph()
        for _, row in sub.iterrows():
            G.add_edge(row['from_cluster'], row['to_cluster'], weight=row['count'])

        node_colors = [cluster_colour_map[n] for n in G.nodes()]
        edge_weights = np.array([d['weight'] for _, _, d in G.edges(data=True)])
        edge_widths  = (
            min_w + (max_w - min_w) * ((edge_weights - g_min) / (g_max - g_min)) ** 1.5
            if len(edge_weights) and g_max != g_min
            else np.full(len(edge_weights), min_w)
        )

        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500,
                                alpha=0.9, ax=ax)
        edges_nsl = [(u, v) for u, v in G.edges() if u != v]
        ew_nsl = [edge_widths[list(G.edges()).index(e)] for e in edges_nsl]
        nx.draw_networkx_edges(G, pos, edgelist=edges_nsl, width=ew_nsl,
                                alpha=0.7, arrows=True,
                                arrowstyle='-|>', arrowsize=22, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=20, font_weight='bold', ax=ax)
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)},
            font_size=12, ax=ax,
        )
        ax.set_title(title, fontsize=22)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — BEHAVIOURAL ENTROPY
# ──────────────────────────────────────────────────────────────────────────────

def compute_entropy(df_no_outliers: pd.DataFrame) -> pd.DataFrame:
    """Compute Shannon entropy of cluster-visit distribution per animal."""
    counts = (
        df_no_outliers
        .groupby(['experimental_id', 'Sex', 'Geno', 'Cluster'])
        .size()
        .reset_index(name='count')
    )
    ent = (
        counts.groupby(['experimental_id', 'Sex', 'Geno'])
        .apply(lambda x: entropy(x['count']))
        .reset_index(name='entropy')
    )
    ent['Sex_Geno'] = ent['Sex'] + '_' + ent['Geno']
    return ent


def plot_entropy(
    state_entropy: pd.DataFrame,
    group_order: list,
    group_labels: list,
    palette: dict,
    dunn_results: pd.DataFrame = None,
    save_path: str = None,
) -> None:
    """Bar + strip plot of per-animal behavioural entropy with optional annotations."""

    def p_to_symbol(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        if p < 0.1:   return '#'
        return 'ns'

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=state_entropy, x='Sex_Geno', y='entropy',
                order=group_order, palette=palette,
                estimator='mean', errorbar='se',
                edgecolor='black', alpha=0.85, ax=ax)
    sns.stripplot(data=state_entropy, x='Sex_Geno', y='entropy',
                  order=group_order, color='black',
                  jitter=0.25, size=5, alpha=0.7, ax=ax)

    ax.set_xticklabels(group_labels, fontsize=18, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, fontweight='bold')
    ax.set_ylabel('Behavioural entropy', fontsize=22, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('Behavioural state entropy per animal', fontsize=22, fontweight='bold')
    sns.despine()

    if dunn_results is not None:
        y_max = state_entropy['entropy'].max()
        h     = y_max * 0.05
        for i in range(len(group_order)):
            for j in range(i + 1, len(group_order)):
                sym = p_to_symbol(dunn_results.iloc[i, j])
                if sym != 'ns':
                    y = y_max + h * (j - i)
                    ax.plot([i, i, j, j], [y, y + h, y + h, y], lw=1.5, c='black')
                    ax.text((i + j) / 2, y + h, sym, ha='center', va='bottom',
                             fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 — TEMPORAL DYNAMICS
# ──────────────────────────────────────────────────────────────────────────────

def plot_temporal_cluster_dynamics(
    df_no_outliers: pd.DataFrame,
    cluster_colour_map: dict,
    bin_seconds: int = 30,
    save_path: str = None,
) -> None:
    """
    Polar plot showing the proportion of each cluster across 30-second
    time bins over the full recording duration.
    """
    df_counts = (
        df_no_outliers
        .groupby(['Interval_bin', 'Cluster'])
        .size()
        .reset_index(name='count')
    )
    df_total = (
        df_no_outliers
        .groupby('Interval_bin')
        .size()
        .reset_index(name='total')
    )
    df_merged = df_counts.merge(df_total, on='Interval_bin')
    df_merged['cluster_prop'] = df_merged['count'] / df_merged['total']

    # Aggregate to 30-second bins
    n_bins_per_30s = bin_seconds // INTERVAL_SEC
    df_merged['bin_30s'] = (df_merged['Interval_bin'] // n_bins_per_30s) * bin_seconds
    df_agg = df_merged.groupby(['bin_30s', 'Cluster'], as_index=False)['cluster_prop'].mean()

    intervals  = sorted(df_agg['bin_30s'].unique())
    clusters   = sorted(df_agg['Cluster'].unique())
    n_intervals = len(intervals)
    theta      = 2 * np.pi * np.arange(n_intervals) / n_intervals
    max_time_s = intervals[-1] + bin_seconds

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 12))

    for cluster in clusters:
        y = np.zeros(n_intervals)
        sub = df_agg[df_agg['Cluster'] == cluster].sort_values('bin_30s')
        for _, row in sub.iterrows():
            y[intervals.index(row['bin_30s'])] = row['cluster_prop']
        tc = np.append(theta, theta[0])
        yc = np.append(y, y[0])
        ax.plot(tc, yc, color=cluster_colour_map[cluster], linewidth=3,
                label=f'Cluster {cluster}', alpha=0.9)
        ax.fill(tc, yc, color=cluster_colour_map[cluster], alpha=0.15)

    major_min  = np.arange(0, min(10, max_time_s // 60 + 1), 2)
    tick_locs  = 2 * np.pi * (major_min * 60 / max_time_s)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f"{m} min" for m in major_min],
                        fontsize=25, color='black', weight='bold')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.tick_params(axis='x', pad=10)
    ax.grid(True)
    ax.tick_params(axis='y', labelsize=18, labelcolor='grey')
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title('Cluster Representation Across Time (10 min)',
                  va='bottom', fontsize=25, weight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=18, frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    df_raw  = load_and_combine(FILE_FEMALES, FILE_MALES)
    agg     = bin_behaviours(df_raw, INTERVAL_SEC, BEHAVIOUR_COLS)
    df_imp, impute_cols = filter_and_impute(
        agg, EXCLUDE_IDS, EXCLUDE_GENOS, LAST_INTERVAL, BEHAVIOUR_COLS
    )

    # ── 2. UMAP + HDBSCAN ────────────────────────────────────────────────────
    embedding        = run_umap(df_imp, impute_cols, UMAP_PARAMS)
    labels, clusterer = run_hdbscan(embedding, HDBSCAN_PARAMS)
    df_imp['Cluster'] = labels

    # ── 3. Validation ────────────────────────────────────────────────────────
    validate_clusters(embedding, labels, clusterer)

    # ── 4. Subset non-outlier frames & build colour map ──────────────────────
    df_no_out = df_imp[df_imp['Cluster'] != -1].reset_index(drop=True)
    cluster_ids = sorted(df_no_out['Cluster'].unique())
    cmap = build_cluster_colour_map(cluster_ids)

    # Renamed behaviour columns for plotting
    renamed_cols = [LABEL_MAP.get(c, c) for c in impute_cols
                    if c in LABEL_MAP or c in impute_cols]
    df_no_out_r  = df_no_out.rename(columns=LABEL_MAP)
    renamed_cols = [LABEL_MAP.get(c, c) for c in impute_cols]

    # ── 5. Visualise embedding & profiles ────────────────────────────────────
    plot_umap_embedding(embedding, df_imp, cmap,
                        save_path="figures/UMAP_embedding.pdf")
    plot_polar_behavioural_profile(df_no_out_r, renamed_cols, cmap,
                                   save_path="figures/polar_profile.pdf")
    plot_cluster_heatmap(df_no_out_r, renamed_cols,
                         save_path="figures/cluster_heatmap.pdf")

    # ── 6. Composition statistics ────────────────────────────────────────────
    chi_square_cluster_composition(df_no_out, 'Geno')
    chi_square_cluster_composition(df_no_out, 'Sex')
    permanova_cluster_factor(df_no_out, impute_cols, 'Geno')
    permanova_cluster_factor(df_no_out, impute_cols, 'Sex')

    # ── 7. Feature statistics ────────────────────────────────────────────────
    kw_df  = kruskal_wallis_features(df_no_out, impute_cols)
    dunn_d = dunn_posthoc(df_no_out, impute_cols, kw_df)

    # ── 8. Proportion modelling ───────────────────────────────────────────────
    df_bw_f = pd.read_csv(FILE_BW_FEMALES)
    df_bw_m = pd.read_csv(FILE_BW_MALES)
    df_bw_f['Sex'] = 'Female'
    df_bw_m['Sex'] = 'Male'
    df_bw = pd.concat([df_bw_f, df_bw_m], ignore_index=True)

    df_merged = compute_cluster_proportions(df_no_out, df_bw)
    ols_cluster_models(df_merged)
    gam_cluster_models(df_merged)

    # ── 9. Transition analysis ────────────────────────────────────────────────
    trans_df = compute_transitions(df_no_out)
    trans_df['Sex_Geno'] = trans_df['Sex'] + '-' + trans_df['Geno']
    trans_counts = (
        trans_df.groupby(['Sex_Geno', 'from_cluster', 'to_cluster'])
        .size()
        .reset_index(name='count')
    )
    plot_transition_heatmaps(trans_counts,
                              save_path="figures/transition_heatmaps.pdf")
    plot_transition_networks(trans_counts, cmap,
                              save_path="figures/transition_networks.pdf")

    # ── 10. Entropy ───────────────────────────────────────────────────────────
    ent_df = compute_entropy(df_no_out)
    groups_ent = [ent_df.loc[ent_df['Sex_Geno'] == g, 'entropy'] for g in GROUP_ORDER]
    H, p_global = kruskal(*groups_ent)
    print(f"\nEntropy — Kruskal–Wallis H={H:.3f}, p={p_global:.4g}")
    dunn_ent = sp.posthoc_dunn(ent_df, val_col='entropy',
                                group_col='Sex_Geno', p_adjust='fdr_bh')
    dunn_ent = dunn_ent.loc[GROUP_ORDER, GROUP_ORDER]
    plot_entropy(ent_df, GROUP_ORDER, GROUP_LABELS, GROUP_COLOURS, dunn_ent,
                 save_path="figures/entropy.pdf")

    # ── 11. Temporal dynamics ─────────────────────────────────────────────────
    plot_temporal_cluster_dynamics(df_no_out, cmap,
                                   save_path="figures/temporal_dynamics.pdf")

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
