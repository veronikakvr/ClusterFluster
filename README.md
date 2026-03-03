# ClusterFluster

Unsupervised behavioural state discovery from rodent recordings using UMAP dimensionality reduction and HDBSCAN density-based clustering of [DeepOF](https://github.com/mlfpm/deepof) output.

---

## Overview

| Script | Recording type | Key features |
|--------|---------------|--------------|
| `scripts/01_umap_hdbscan_10min.py` | Short (≤10 min) | UMAP training, HDBSCAN, OLS/GAM proportion models |
| `scripts/02_umap_hdbscan_overnight.py` | Overnight (12 h) | Pre-trained UMAP, bootstrap stability, GEE + BW correction, Markov chains |
| `scripts/utils.py` | — | Shared helper functions imported by both scripts |

---

## Pipelines

### 10-minute recordings (`01_umap_hdbscan_10min.py`)

```
Raw DeepOF CSVs (male + female)
        ↓
  2-second binning
        ↓
  Filtering + multiple imputation (IterativeImputer)
        ↓
  Standard scaling
        ↓
  UMAP (2D embedding, trained here)
        ↓
  HDBSCAN clustering
        ↓
  ┌─────────────────────────────────────────────────┐
  │  Cluster validation   (Silhouette, DB, DBCV)    │
  │  Behavioural profiling (polar plot, heatmap)    │
  │  Composition stats    (χ², Cramér's V,          │
  │                         PERMANOVA)              │
  │  Feature stats        (Kruskal–Wallis, Dunn)    │
  │  Proportion modelling (OLS, GAM splines)        │
  │  Transition analysis  (heatmaps, networks)      │
  │  Behavioural entropy                            │
  │  Temporal dynamics    (polar time plot)         │
  └─────────────────────────────────────────────────┘
```

### Overnight recordings (`02_umap_hdbscan_overnight.py`)

```
Raw DeepOF CSVs (male + female, 12 h)
        ↓
  5-second binning
        ↓
  RobustScaler
        ↓
  Load pre-trained UMAP model → transform
        ↓
  HDBSCAN clustering
        ↓
  Bootstrap stability validation (n=100, Jaccard ≥ 0.75)
        ↓
  ┌─────────────────────────────────────────────────┐
  │  Behavioural repertoire (heatmaps, radial plots)│
  │  Composition stats    (χ², Cramér's V, GLM)     │
  │  GEE models + BW correction (hourly occupancy)  │
  │  Transition analysis  (overall + hourly nets)   │
  │  Dwell time analysis  (mixed-effects model)     │
  │  Shannon entropy + transition complexity        │
  │  Markov chain modelling                         │
  │  Silhouette + pairwise PERMANOVA validation     │
  └─────────────────────────────────────────────────┘
```

---

## Repository structure

```
ClusterFluster/
├── scripts/
│   ├── utils.py                       ← shared functions
│   ├── 01_umap_hdbscan_10min.py       ← imports from utils
│   └── 02_umap_hdbscan_overnight.py   ← imports from utils
├── data/
│   └── example/                       ← # pending to be added
├── figures/                           ← output PDFs saved here (gitignored)
├── environment.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

**With conda (recommended):**

```bash
conda env create -f environment.yml
conda activate clusterfluster
```

**With pip:**

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Edit the PARAMETERS block

Both scripts have a clearly labelled `PARAMETERS` section at the top. This is the only part you need to edit:

```python
# --- Input files ---
FEMALE_CSV = "/path/to/females.csv"
MALE_CSV   = "/path/to/males.csv"
BW_FEMALE_CSV = "/path/to/bw_females.csv"
BW_MALE_CSV   = "/path/to/bw_males.csv"

# --- Exclusions ---
EXCLUDED_IDS  = ["ID49", "ID61"]
EXCLUDED_GENO = "atg7OE"

# --- HDBSCAN ---
MIN_CLUSTER_SIZE = 600
MIN_SAMPLES      = 300
```

For the overnight script, also set:

```python
UMAP_MODEL_PATH = "umap_model_DB_chow.sav"   # path to your pre-trained UMAP
LOAD_BOOTSTRAP  = True    # True = load cached results; False = re-run from scratch
```

### 2. Run

```bash
python scripts/01_umap_hdbscan_10min.py
# or
python scripts/02_umap_hdbscan_overnight.py
```

All figures are saved as vector PDFs to the `figures/` directory.

---

## Methods

### Preprocessing
Behavioural scores from male and female cohorts are concatenated and binned into fixed-width time intervals (2 s for 10-min recordings; 5 s for overnight). Features are aggregated as sum, mean, and standard deviation per interval. Missing values are handled via multiple imputation (MICE; 10-min) or median fill with RobustScaler (overnight).

### Dimensionality reduction
UMAP projects the high-dimensional behavioural feature space into a 2D embedding. For 10-min recordings the model is trained on the dataset. For overnight recordings a pre-trained UMAP is loaded and used only for inference, ensuring the embedding space is identical across datasets.

### Clustering
HDBSCAN identifies density-based clusters in the UMAP embedding. Frames assigned label −1 (noise) are excluded from downstream analyses.

### Bootstrap stability (overnight only)
100 bootstrap replicates are run on 60% subsamples of the embedding. Cluster correspondence is resolved with the Hungarian algorithm and per-cluster Jaccard stability is computed. Only clusters with mean Jaccard ≥ 0.75 are retained.

### Statistical analyses
- **Composition**: χ² + Cramér's V (global); GLM and GEE per cluster for genotype × sex effects.
- **Body weight correction**: GEE Poisson models include body weight as a continuous covariate; Wald tests compare groups at mean BW.
- **Transition analysis**: Directed transition count and probability matrices; network graph visualisation (Kamada–Kawai layout); Poisson GLM for group and time effects.
- **Dwell time**: Run-length encoding of cluster sequences; log-dwell modelled with a linear mixed-effects model (animal as random intercept).
- **Entropy**: Shannon entropy of each animal's cluster-visit distribution per hour.
- **Complexity**: Transition entropy rate (stationary distribution × per-row conditional entropy).
- **Markov chains**: Per-animal transition probability matrices; group differences tested with a linear mixed-effects model.
- **Validation**: Silhouette score; pairwise PERMANOVA (Bray–Curtis distance, 999 permutations, Bonferroni correction).

---

## Output figures

### 10-minute pipeline

| File | Description |
|------|-------------|
| `UMAP_embedding.pdf` | 2D UMAP scatter coloured by cluster |
| `polar_profile.pdf` | Radar chart of mean behavioural scores per cluster |
| `cluster_heatmap.pdf` | Heatmap of cluster behavioural profiles |
| `transition_heatmaps.pdf` | 4-panel transition count heatmaps |
| `transition_networks.pdf` | 4-panel directed transition networks |
| `entropy.pdf` | Per-animal entropy by sex × genotype |
| `temporal_dynamics.pdf` | Cluster proportions over time (polar) |

### Overnight pipeline

| File | Description |
|------|-------------|
| `bootstrap_jaccard_histogram.pdf` | Cluster stability distribution |
| `REPORT_UMAP_stable_vs_unstable.pdf` | UMAP coloured by stability |
| `cluster_profiles_*.pdf` | Heatmaps of sum / mean / std profiles |
| `REPORT_radial_proportions.pdf` | Radial plot of proportional repertoire |
| `GEE_All_Clusters_Hourly_Occupancy_with_Star_FINAL.pdf` | BW-adjusted hourly occupancy + significance stars |
| `REPORT_cluster_transition_network_OVERALL_*.pdf` | Overall transition networks |
| `REPORT_cluster_transition_network_*_chow.pdf` | Per-Sex_Geno hourly transition networks |
| `REPORT_dwellTimes_perCluster_*.pdf` | Dwell time boxplots |
| `Complexity_Barplot_Final_ENTROPY_*.pdf` | Behavioural complexity bar plots |
| `transition_probability_matrices_*.pdf` | Per-group Markov probability heatmaps |

---

## Dependencies

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
| joblib | ≥ 1.3 |
| statannotations | ≥ 0.6 |

---

## Citation

If you use this pipeline in your research, please cite this repository and the relevant tools:

- Bordes *et al.* (2023) DeepOF. *NatComm*. https://doi.org/10.1038/s41467-023-40040-3


---

## License

MIT © \<your name\>
