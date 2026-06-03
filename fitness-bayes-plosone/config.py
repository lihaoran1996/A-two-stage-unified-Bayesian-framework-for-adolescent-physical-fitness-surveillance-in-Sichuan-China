from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Input directories
DATA_DIR = PROJECT_ROOT / "data"
TRACE_DIR = PROJECT_ROOT / "traces"
SHP_DIR = PROJECT_ROOT / "shp"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_STAGE1_DIR = OUTPUT_DIR / "figures_stage1"
FIGURE_STAGE2_DIR = OUTPUT_DIR / "figures_stage2"

for directory in [DATA_DIR, TRACE_DIR, SHP_DIR, FIGURE_STAGE1_DIR, FIGURE_STAGE2_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Stage 1 public figure tables
FIG2_TABLE = DATA_DIR / "Fig_2_factor_loadings_table.csv"
FIGS1_TABLE = DATA_DIR / "Fig_S1_ppc_density_table.csv"
FIGS2_FACTOR_SCORE_TABLE = DATA_DIR / "Fig_S2_factor_scores_cell_mean_table.csv"
FIGS2_CORR_TABLE = DATA_DIR / "Fig_S2_factor_correlation_matrix.csv"

# Stage 2 posterior traces
TRACE_FILE_MAP = {
    "力量": TRACE_DIR / "trace_step2_力量.nc",
    "耐力": TRACE_DIR / "trace_step2_耐力.nc",
    "柔韧性": TRACE_DIR / "trace_step2_柔韧性.nc",
    "速度": TRACE_DIR / "trace_step2_速度.nc",
}

# Spatial and factor-level data used for map reproduction
SHP_PATH = SHP_DIR / "sichuan_all" / "sichuan_all.shp"
HEALTH_DATA_PATH = DATA_DIR / "health_data_with_factors.csv"
