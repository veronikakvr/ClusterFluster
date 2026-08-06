"""
utils.py

Utility functions for behavioral analysis pipeline:
  - Behavior binning & aggregation
  - Cluster color mapping & visualization (UMAP, Heatmaps, Radial Plots)
  - Markov transition matrix computation & network visualizations
  - Information theory (Entropy) calculations
  - Statistical testing (Chi-square composition & PERMANOVA distance matrix testing)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from scipy.stats import chi2_contingency, entropy
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import permanova, DistanceMatrix


# ==============================================================================
# 1. DATA PROCESSING & AGGREGATION
# ==============================================================================

def bin_behaviours(
    df: pd.DataFrame,
    behavior_cols: list,
    interval_length: int = 5,
    exclude_cols: list = None
) -> pd.DataFrame:
    """
    Computes interval time bins and aggregates behavioral columns by sum, mean, and std.
    """
    df_work = df.copy()
    
    # Ensure Time_sec exists
    if 'Time_sec' not in df_work.columns and 'Time' in df_work.columns:
        if pd.api.types.is_timedelta64_dtype(df_work['Time']):
            df_work['Time_sec'] = df_work['Time'].dt.total_seconds()
        else:
            df_work['Time_sec'] = pd.to_timedelta(df_work['Time']).dt.total_seconds()
            
    # Calculate numeric interval bin
    df_work['Interval_bin'] = (df_work['Time_sec'] // interval_length).astype(int)

    # Filter available behavior columns
    avail_cols = [c for c in behavior_cols if c in df_work.columns]
    agg_dict = {col: ['sum', 'mean', 'std'] for col in avail_cols}

    # Group by experimental keys and interval
    groupby_keys = ['experimental_id', 'Geno', 'Sex', 'Interval_bin']
    groupby_keys = [k for k in groupby_keys if k in df_work.columns]

    features = df_work.groupby(groupby_keys).agg(agg_dict)
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features = features.reset_index()

    # Calculate interval labels and hourly bins
    features['Interval_start'] = pd.to_timedelta(features['Interval_bin'] * interval_length, unit='s')
    features['Interval_end'] = pd.to_timedelta((features['Interval_bin'] + 1) * interval_length, unit='s')
    features['Interval_label'] = features['Interval_start'].astype(str) + ' - ' + features['Interval_end'].astype(str)
    
    sec_per_hour = 3600
    features['Hour_bin'] = features['Interval_bin'] // (sec_per_hour // interval_length)
    features['Halfhour_bin'] = (features['Interval_bin'] // (1800 / interval_length)).astype(int)

    return features.sort_values(['experimental_id', 'Interval_bin']).reset_index(drop=True)


# ==============================================================================
# 2. COLOR MAPPING & VISUALIZATIONS
# ==============================================================================

def build_cluster_colour_map(clusters: list, palette_name: str = "tab20") -> dict:
    """
    Builds a consistent cluster-to-color mapping dictionary across all visualizations.
    """
    unique_clusters = sorted([c for c in set(clusters) if c != -1])
    cmap = plt.get_cmap(palette_name)
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(unique_clusters)}
    if -1 in clusters:
        color_map[-1] = (0.7, 0.7, 0.7, 1.0)  # Gray for noise/unclustered points
    return color_map


def plot_umap_embedding(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    umap1_col: str = "UMAP1",
    umap2_col: str = "UMAP2",
    color_map: dict = None,
    title: str = "UMAP Embedding",
    ax: plt.Axes = None
):
    """
    Plots a 2D UMAP scatter plot colored by cluster ID.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if color_map is None:
        color_map = build_cluster_colour_map(df[cluster_col].unique())

    sns.scatterplot(
        data=df,
        x=umap1_col,
        y=umap2_col,
        hue=cluster_col,
        palette=color_map,
        alpha=0.6,
        s=20,
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster")
    plt.tight_layout()
    return ax


def plot_cluster_heatmap(
    df: pd.DataFrame,
    cols: list,
    title: str = "Cluster Behavioral Profiles",
    save_path: str = None
):
    """
    Plots a cluster vs feature heatmap.
    """
    plt.figure(figsize=(22, 7))
    sns.heatmap(
        df[cols],
        cmap='viridis',
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 10, "fontname": "Arial"}
    )
    plt.title(title, fontsize=16, fontname='Arial', fontweight='bold')
    plt.xlabel("Behavioral Feature", fontsize=12, fontname='Arial', fontweight='bold')
    plt.ylabel("Cluster", fontsize=12, fontname='Arial', fontweight='bold')
    plt.xticks(rotation=90, fontsize=10, fontname='Arial')
    plt.yticks(fontsize=10, fontname='Arial')

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_radial(
    df_mean: pd.DataFrame,
    df_std: pd.DataFrame = None,
    title: str = "Behavioral Repertoire",
    color_map: dict = None,
    rmax: float = None
):
    """
    Plots a polar radial spider chart of cluster behavioral profiles with optional error shading.
    """
    categories = [
        c.replace('_sum', '').replace('_mean', '').replace('_std', '') 
        for c in df_mean.columns
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close loop

    if color_map is None:
        color_map = build_cluster_colour_map(df_mean.index.tolist())

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for cluster in df_mean.index:
        values = df_mean.loc[cluster].tolist()
        values += values[:1]

        if df_std is not None and cluster in df_std.index:
            errors = df_std.loc[cluster].tolist()
            errors += errors[:1]
        else:
            errors = np.zeros_like(values)

        color = color_map.get(cluster, (0.2, 0.4, 0.8, 1.0))

        ax.plot(angles, values, label=f'Cluster {cluster}', color=color, linewidth=2)

        if df_std is not None:
            val_arr = np.array(values)
            err_arr = np.array(errors)
            ax.fill_between(
                angles,
                np.maximum(0, val_arr - err_arr),
                val_arr + err_arr,
                color=color,
                alpha=0.15
            )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, weight='bold')

    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        angle_deg = np.degrees(angle)
        if angle_deg >= 270 or angle_deg <= 90:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    if rmax is None:
        rmax = max(df_mean.max().max() + 0.05, 0.5)
    ax.set_rmax(rmax)

    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05), fontsize=10, frameon=False)
    plt.title(title, fontsize=14, weight='bold', pad=25)
    plt.tight_layout()
    plt.show()
    plt.close()


# ==============================================================================
# 3. TRANSITION & NETWORK ANALYSIS
# ==============================================================================

def compute_transitions(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    id_col: str = "experimental_id",
    time_col: str = "Interval_bin"
) -> pd.DataFrame:
    """
    Calculates first-order Markov transition matrices (counts and row-normalized probabilities) 
    between sequential behavioral clusters.
    """
    df_sorted = df.sort_values([id_col, time_col]).copy()
    df_sorted['next_cluster'] = df_sorted.groupby(id_col)[cluster_col].shift(-1)
    
    # Filter out valid transitions
    valid = df_sorted.dropna(subset=['next_cluster'])
    valid = valid[(valid[cluster_col] != -1) & (valid['next_cluster'] != -1)]
    
    counts = pd.crosstab(
        valid[cluster_col].astype(int), 
        valid['next_cluster'].astype(int), 
        dropna=False
    )
    
    # Row-normalize to get transition probabilities
    prob_matrix = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return prob_matrix


def plot_transition_networks(
    prob_matrix: pd.DataFrame,
    threshold: float = 0.05,
    color_map: dict = None,
    title: str = "Cluster Transition Network"
):
    """
    Renders a directed NetworkX graph of cluster transitions above a probability threshold.
    """
    G = nx.DiGraph()
    clusters = prob_matrix.index.tolist()

    for c in clusters:
        G.add_node(c)

    for i in clusters:
        for j in clusters:
            weight = prob_matrix.loc[i, j]
            if weight >= threshold:
                G.add_edge(i, j, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 8))

    if color_map is None:
        color_map = build_cluster_colour_map(clusters)

    node_colors = [color_map.get(node, 'lightgray') for node in G.nodes()]

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", font_color="white")

    edges = G.edges(data=True)
    weights = [d['weight'] * 4 for u, v, d in edges]
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, width=weights, arrowstyle='->', arrowsize=15, edge_color='gray'
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    plt.close()


# ==============================================================================
# 4. STATISTICAL & INFORMATION THEORY UTILITIES
# ==============================================================================

def compute_entropy(labels: pd.Series, base: float = 2.0) -> float:
    """
    Computes Shannon entropy of cluster label distributions.
    """
    counts = labels.value_counts()
    probs = counts / len(labels)
    return entropy(probs, base=base)


def chi_square_cluster_composition(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    group_col: str = "Geno"
) -> pd.DataFrame:
    """
    Performs Chi-square independence tests on cluster compositions across experimental groups.
    """
    contingency = pd.crosstab(df[cluster_col], df[group_col])
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    
    results = pd.DataFrame([{
        'Chi2_Statistic': chi2,
        'p_value': p_val,
        'Degrees_of_Freedom': dof
    }])
    return results, contingency


def permanova_cluster_factor(
    df: pd.DataFrame,
    feature_cols: list,
    group_col: str = "Geno",
    metric: str = "euclidean",
    permutations: int = 999
):
    """
    Performs PERMANOVA non-parametric multivariate analysis of variance on behavioral feature distances.
    """
    clean_df = df.dropna(subset=feature_cols).copy()
    dist_array = pdist(clean_df[feature_cols].values, metric=metric)
    dist_matrix = DistanceMatrix(squareform(dist_array), ids=clean_df.index.astype(str))
    
    permanova_results = permanova(
        distance_matrix=dist_matrix,
        grouping=clean_df[group_col].astype(str),
        permutations=permutations
    )
    return permanova_results