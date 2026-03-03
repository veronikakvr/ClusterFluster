# ClusterFluster

Unsupervised behavioural state discovery from rodent open-field recordings using UMAP dimensionality reduction and HDBSCAN density-based clustering of [DeepOF](https://github.com/mlfpm/deepof) output.

---

## Overview

| Script | Purpose |
|--------|---------|
| `scripts/01_umap_hdbscan_10min.py` | Full pipeline for short (≤10 min) recordings |
| `scripts/02_umap_hdbscan_overnight.py` | Extended pipeline for overnight recordings (includes bootstrapping) *(coming soon)* |

---

## Pipeline — 10-minute recordings

```
Raw DeepOF CSVs (male + female)
        ↓
  2-second binning
        ↓
  Filtering + multiple imputation (IterativeImputer)
        ↓
  Standard scaling
        ↓
  UMAP (2D embedding)
        ↓
  HDBSCAN clustering
        ↓
  ┌─────────────────────────────────────────────────┐
  │  Cluster validation        (Silhouette, DB, DBCV) │
  │  Behavioural profiling     (polar plot, heatmap)  │
  │  Composition statistics    (χ², Cramér's V,       │
  │                             PERMANOVA)            │
  │  Feature statistics        (Kruskal–Wallis, Dunn) │
  │  Proportion modelling      (OLS, GAM splines)     │
  │  Transition analysis       (heatmaps, networks)   │
  │  Behavioural entropy                              │
  │  Temporal dynamics         (polar time plot)      │
  └─────────────────────────────────────────────────┘
```

---

## Repository Structure

```
ClusterFluster/
├── scripts/
│   ├── 01_umap_hdbscan_10min.py
│   └── 02_umap_hdbscan_overnight.py   # coming soon
├── data/
│   └── example/                       # place example DeepOF CSVs here
├── figures/                           # output figures saved here (gitignored)
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Requirements

| Package | Version tested |
|---------|---------------|
| Python | ≥ 3.10 |
| numpy | ≥ 1.24 |
| pandas | ≥ 2.0 |
| scikit-learn | ≥ 1.3 |
| umap-learn | ≥ 0.5 |
| hdbscan | ≥ 0.8 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.13 |
| scipy | ≥ 1.11 |
| statsmodels | ≥ 0.14 |
| scikit-posthocs | ≥ 0.8 |
| scikit-bio | ≥ 0.5 |
| networkx | ≥ 3.1 |
| pypalettes | ≥ 0.1 |
| statannotations | ≥ 0.6 |

**Install with conda (recommended):**

```bash
conda env create -f environment.yml
conda activate clusterfluster
```

**Or with pip:**

```bash
pip install -r requirements.txt
```

---

## Usage

1. Open `scripts/01_umap_hdbscan_10min.py`
2. Edit the **PARAMETERS** block at the top of the file:

```python
# Input data
FILE_FEMALES = "/path/to/females.csv"
FILE_MALES   = "/path/to/males.csv"
FILE_BW_FEMALES = "/path/to/bw_females.csv"
FILE_BW_MALES   = "/path/to/bw_males.csv"

# Filtering
EXCLUDE_IDS   = ['ID63', 'ID214']
EXCLUDE_GENOS = ['atg7OE']
LAST_INTERVAL = "0 days 00:09:58 - 0 days 00:10:00"

# UMAP
UMAP_PARAMS = dict(n_neighbors=30, min_dist=0.1, ...)

# HDBSCAN
HDBSCAN_PARAMS = dict(min_cluster_size=500, min_samples=90)
```

3. Run:

```bash
python scripts/01_umap_hdbscan_10min.py
```

All figures are saved as vector PDFs to the `figures/` directory.

---

## Methods

### Preprocessing
Behavioural scores from male and female cohorts are concatenated and binned into 2-second intervals. Low-confidence or missing values are handled via multiple imputation (MICE; `IterativeImputer` with `random_state=42`). All features are then standardised (zero mean, unit variance) before embedding.

### Dimensionality reduction
UMAP (`n_neighbors=30`, `min_dist=0.1`, Euclidean metric) projects the high-dimensional behavioural feature space into a 2D embedding. Thread count is fixed to 1 before import to ensure reproducibility across machines.

### Clustering
HDBSCAN identifies density-based clusters in the UMAP embedding. Frames not assigned to any cluster (label = −1) are treated as noise and excluded from downstream analyses.

### Validation
Cluster quality is assessed with four complementary metrics: DBCV (HDBSCAN-native validity index), Silhouette score, Davies–Bouldin index, and Calinski–Harabasz index.

### Statistical analyses
- **Composition**: χ² + Cramér's V (global); PERMANOVA per cluster for genotype and sex effects.
- **Feature differences**: Kruskal–Wallis H-test across clusters with Benjamini–Hochberg FDR correction; Dunn's post-hoc for significant features.
- **Proportion modelling**: OLS regression (`cluster_prop ~ Geno + BW + Sex`) and GAM with B-spline on body weight.
- **Transition analysis**: Directed transition count/probability matrices; network graph visualisation; per-animal ANOVA + Tukey HSD.
- **Entropy**: Shannon entropy of each animal's cluster-visit distribution.
- **Temporal dynamics**: Cluster representation plotted over 30-second bins on a polar axis.

---

## Output figures

| File | Description |
|------|------------|
| `UMAP_embedding.pdf` | 2D scatter plot of UMAP embedding |
| `polar_profile.pdf` | Radar chart of mean behavioural scores per cluster |
| `cluster_heatmap.pdf` | Heatmap of cluster behavioural profiles |
| `transition_heatmaps.pdf` | 4-panel transition count heatmaps |
| `transition_networks.pdf` | 4-panel directed transition network graphs |
| `entropy.pdf` | Per-animal entropy by sex × genotype group |
| `temporal_dynamics.pdf` | Polar plot of cluster proportions over time |

---

## Citation

If you use this pipeline in your research, please cite this repository and the relevant DeepOF paper:

- Bordes *et al.* (2023) DeepOF. *NatComm*. https://doi.org/10.1038/s41467-023-40040-3


---

## License

MIT © \<your name\>
