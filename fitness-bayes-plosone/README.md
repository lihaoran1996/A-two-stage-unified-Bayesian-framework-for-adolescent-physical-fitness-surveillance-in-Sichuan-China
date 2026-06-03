# fitness-bayes-plosone

This repository contains code for the manuscript **"A two-stage unified Bayesian framework for adolescent physical fitness surveillance in Sichuan, China: latent factor measurement followed by spatiotemporal modeling"**.

The repository is organized for manuscript revision and reproducibility. It contains two model-construction templates and three figure-reproduction scripts.

## Repository structure

```text
fitness-bayes-plosone/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── config.py
├── code/
│   ├── 01_stage1_model_template.py
│   ├── 02_stage2_model_template.py
│   ├── 03_plot_stage1_figures.py
│   ├── 04_plot_stage2_diagnostics.py
│   └── 05_plot_stage2_maps.py
├── data/
│   ├── Fig_2_factor_loadings_table.csv
│   ├── Fig_S1_ppc_density_table.csv
│   ├── Fig_S2_factor_scores_cell_mean_table.csv
│   ├── Fig_S2_factor_correlation_matrix.csv
│   └── health_data_with_factors.csv
├── traces/
│   ├── trace_step2_力量.nc
│   ├── trace_step2_耐力.nc
│   ├── trace_step2_柔韧性.nc
│   └── trace_step2_速度.nc
├── shp/
│   └── sichuan_all/
│       ├── sichuan_all.shp
│       ├── sichuan_all.shx
│       ├── sichuan_all.dbf
│       ├── sichuan_all.prj
│       └── sichuan_all.cpg
└── outputs/
    ├── figures_stage1/
    └── figures_stage2/
```

## Code files

- `code/01_stage1_model_template.py`: PyMC template for the Stage 1 Bayesian CFA model. This file documents the model construction and required input arrays. It is not intended to run without access to the restricted individual-level data.
- `code/02_stage2_model_template.py`: PyMC template for the Stage 2 Bayesian spatiotemporal hierarchical model. This file documents the model construction and required input arrays. It is not intended to run without access to the restricted analytical data.
- `code/03_plot_stage1_figures.py`: Reproduces Stage 1 figures using CSV tables in `data/`.
- `code/04_plot_stage2_diagnostics.py`: Reproduces Stage 2 posterior coefficient, spatial-effect, and PPC figures using posterior traces in `traces/`.
- `code/05_plot_stage2_maps.py`: Reproduces the raw factor mean and adjusted spatial random-effect maps using `data/health_data_with_factors.csv`, posterior traces, and the Sichuan shapefile.

## Required input files

Place the following files in `data/`:

```text
Fig_2_factor_loadings_table.csv
Fig_S1_ppc_density_table.csv
Fig_S2_factor_scores_cell_mean_table.csv
Fig_S2_factor_correlation_matrix.csv
health_data_with_factors.csv
```

Place the following posterior trace files in `traces/`:

```text
trace_step2_力量.nc
trace_step2_耐力.nc
trace_step2_柔韧性.nc
trace_step2_速度.nc
```

Place the shapefile components in `shp/sichuan_all/`:

```text
sichuan_all.shp
sichuan_all.shx
sichuan_all.dbf
sichuan_all.prj
sichuan_all.cpg
```

The `.shp` file must be accompanied by its associated `.shx`, `.dbf`, `.prj`, and other sidecar files.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate fitness-bayes
```

or using pip:

```bash
pip install -r requirements.txt
```

## Reproducing figures

Run the scripts from the repository root.

### Stage 1 figures

```bash
python code/03_plot_stage1_figures.py
```

Outputs are saved to:

```text
outputs/figures_stage1/
```

### Stage 2 diagnostic figures

```bash
python code/04_plot_stage2_diagnostics.py
```

Outputs are saved to:

```text
outputs/figures_stage2/
```

### Stage 2 spatial maps

```bash
python code/05_plot_stage2_maps.py
```

Outputs are saved to:

```text
outputs/figures_stage2/Fig_raw_mean_vs_spatial_random_effects.png
```

## Data availability note

The model templates require restricted individual-level or analytical data that are not included in this repository. The figure-reproduction scripts require precomputed tables, posterior traces, and shapefile files placed in the paths listed above.

Before making this repository public, ensure that all data files, posterior traces, factor-score tables, and shapefile files are permitted for public release by the relevant data providers and copyright/license holders.

## License

Code may be released under the MIT License. Data and map files should only be released if their licenses and data-sharing permissions allow public redistribution.
