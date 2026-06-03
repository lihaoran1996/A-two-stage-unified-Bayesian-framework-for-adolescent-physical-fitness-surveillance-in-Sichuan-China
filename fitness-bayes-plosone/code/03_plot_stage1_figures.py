import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 0. Path settings
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FIG2_TABLE,
    FIGS1_TABLE,
    FIGS2_FACTOR_SCORE_TABLE,
    FIGS2_CORR_TABLE,
    FIGURE_STAGE1_DIR,
)

FIGURE_STAGE1_DIR.mkdir(parents=True, exist_ok=True)

fig2_table = FIG2_TABLE
figs1_table = FIGS1_TABLE
figs2_factor_score_table = FIGS2_FACTOR_SCORE_TABLE
figs2_corr_table = FIGS2_CORR_TABLE
FIG_DIR = FIGURE_STAGE1_DIR


# =========================================================
# 1. Fig 2: Factor Loadings Heatmap
# =========================================================

def plot_factor_loadings_heatmap(table_path, save_path=None):
    df = pd.read_csv(table_path)

    item_order = (
        df[["item_order", "item"]]
        .drop_duplicates()
        .sort_values("item_order")["item"]
        .tolist()
    )

    factor_order = (
        df[["factor_order", "factor"]]
        .drop_duplicates()
        .sort_values("factor_order")["factor"]
        .tolist()
    )

    df_wide = df.pivot(
        index="item",
        columns="factor",
        values="loading_mean"
    )

    df_wide = df_wide.loc[item_order, factor_order]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df_wide,
        annot=True,
        cmap="vlag",
        center=0,
        fmt=".2f",
        linewidths=0.8,
        cbar_kws={"label": "Factor Loading"}
    )

    plt.title("Step 1 Result: Factor Loadings Matrix (Λ)", fontsize=14)
    plt.ylabel("Observed Test Items", fontsize=12)
    plt.xlabel("Latent Factors", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return df_wide


# =========================================================
# 2. Fig S1: Posterior Predictive Check
# =========================================================

def plot_ppc_density(table_path, save_path=None):
    df = pd.read_csv(table_path)

    df_obs = df[df["curve_type"] == "observed"]
    df_pp = df[df["curve_type"] == "posterior_predictive"]

    plt.figure(figsize=(8, 5))

    for sample_id, g in df_pp.groupby("sample_id"):
        plt.plot(
            g["x"],
            g["density"],
            linewidth=0.6,
            alpha=0.18
        )

    plt.plot(
        df_obs["x"],
        df_obs["density"],
        linewidth=2.2,
        label="Observed"
    )

    plt.title("Posterior Predictive Check: Overall Distribution")
    plt.xlabel("Observed / Predicted Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# =========================================================
# 3. Fig S2: 原始版本 pairplot / KDE
#    只基于公开的 630 个 cell-level factor mean 表生成
# =========================================================

def plot_factor_correlations_pairplot(
    factor_score_table_path,
    factor_names=None,
    save_path=None,
    corr_table_path=None
):
    df = pd.read_csv(factor_score_table_path)

    if factor_names is None:
        factor_names = [col for col in df.columns if col != "cell_id"]

    # 原始版本：pairplot + KDE
    g = sns.pairplot(
        df[factor_names],
        kind="kde",
        diag_kind="kde",
        corner=True
    )

    g.fig.suptitle("Posterior Correlations between Latent Factors", y=1.02)
    g.fig.tight_layout()

    if save_path is not None:
        g.fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    corr = df[factor_names].corr()

    print("\n=== Latent Factor Correlation Matrix ===")
    print(corr.round(3))

    # 可选：和已导出的 correlation matrix 进行一致性检查
    if corr_table_path is not None and os.path.exists(corr_table_path):
        corr_saved = pd.read_csv(corr_table_path, index_col=0)
        max_diff = (corr.values - corr_saved.values).max()
        print(f"\nMax difference from saved correlation matrix: {abs(max_diff):.10f}")

    return df, corr


# =========================================================
# 4. 执行绘图
# =========================================================

plot_factor_loadings_heatmap(
    table_path=fig2_table,
    save_path=FIG_DIR / "Fig_2_Factor_Loadings_Heatmap.png"
)

plot_ppc_density(
    table_path=figs1_table,
    save_path=FIG_DIR / "Fig_S1_PPC_Distribution.png"
)

df_factors, corr = plot_factor_correlations_pairplot(
    factor_score_table_path=figs2_factor_score_table,
    factor_names=["Strength", "Speed", "Endurance", "Flexibility"],
    save_path=FIG_DIR / "Fig_S2_Factor_Correlations.png",
    corr_table_path=figs2_corr_table
)

