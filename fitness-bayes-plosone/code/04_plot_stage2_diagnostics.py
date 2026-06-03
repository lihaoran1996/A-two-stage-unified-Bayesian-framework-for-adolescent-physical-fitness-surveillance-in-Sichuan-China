from pathlib import Path
import os
import sys

import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Path settings and posterior trace import
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TRACE_FILE_MAP, FIGURE_STAGE2_DIR, HEALTH_DATA_PATH

SAVE_DIR = FIGURE_STAGE2_DIR
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_step2_traces(trace_file_map=TRACE_FILE_MAP):
    """Load existing Stage 2 posterior traces from the traces/ directory."""
    results = {}
    trace_paths = {}
    for factor_cn, path in trace_file_map.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find {path}. Please place the corresponding .nc file in the traces/ directory."
            )
        results[factor_cn] = az.from_netcdf(path)
        trace_paths[factor_cn] = str(path)
    return results, trace_paths


results, trace_paths = load_step2_traces()
factor_names_loaded = list(results.keys())

# =============================================================================
# 3. Name maps
# =============================================================================

factor_name_map = {
    "力量": "Strength",
    "速度": "Speed",
    "耐力": "Endurance",
    "柔韧性": "Flexibility",
    "strength": "Strength",
    "speed": "Speed",
    "endurance": "Endurance",
    "flexibility": "Flexibility",
}

cov_name_map = {
    "人均GDP": "GDP per capita",
    "城镇化率(%)": "Urbanization rate",
    "人口密度(人/平方公里)": "Population density",
    "文化体育传媒支出（万元）": "Culture/Sports/Media expenditure",
    "平均海拔（m）": "Mean elevation",
    "年平均温度(℃)": "Annual mean temperature",
    "全年降水量( millimeters)": "Annual precipitation",
    "年平均相对湿度(%)": "Annual mean relative humidity",
}

city_name_map = {
    "自贡市教育局": "Zigong",
    "攀枝花市教育局": "Panzhihua",
    "泸州市教育局": "Luzhou",
    "德阳市教育局": "Deyang",
    "绵阳市教育和体育局": "Mianyang",
    "广元市教育局": "Guangyuan",
    "遂宁市教育局": "Suining",
    "内江市教育局": "Neijiang",
    "乐山市教育局": "Leshan",
    "南充市教育局": "Nanchong",
    "眉山市教育局": "Meishan",
    "宜宾市教育局": "Yibin",
    "广安市教育局": "Guang'an",
    "达州市教育局": "Dazhou",
    "雅安市教育局": "Ya'an",
    "巴中市教育局": "Bazhong",
    "资阳市教育局": "Ziyang",
    "阿坝藏族羌族自治州教育局": "Aba Prefecture",
    "甘孜藏族自治州教育局": "Garze Prefecture",
    "凉山彝族自治州教育局": "Liangshan Prefecture",
    "成都市教育局": "Chengdu",
}


# =============================================================================
# 4. Helper functions
# =============================================================================

def map_names(names, name_map):
    """
    Translate names using a mapping dictionary.
    Unmatched names are kept unchanged.
    """
    return [name_map.get(str(x), str(x)) for x in names]


def build_city_order_from_cov_panel(
    cov_panel_path,
    city_name_map,
    city_col="市州",
    city_id_col="city_id"
):
    """
    Build the city label order according to the city_id used in the model.

    This is critical because posterior spatial effects S_c are indexed by city_id.
    """
    cov_panel = pd.read_csv(cov_panel_path)

    required_cols = {city_col, city_id_col}
    missing_cols = required_cols - set(cov_panel.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in cov_panel: {missing_cols}")

    city_order_df = (
        cov_panel[[city_id_col, city_col]]
        .drop_duplicates()
        .sort_values(city_id_col)
        .reset_index(drop=True)
    )

    # Check whether city_id is consecutive: 0, 1, ..., N-1
    expected_ids = list(range(len(city_order_df)))
    observed_ids = city_order_df[city_id_col].astype(int).tolist()

    if observed_ids != expected_ids:
        raise ValueError(
            "city_id is not consecutive or not sorted as expected.\n"
            f"Observed city_id: {observed_ids}\n"
            f"Expected city_id: {expected_ids}"
        )

    city_order_df["city_en"] = city_order_df[city_col].map(city_name_map)

    if city_order_df["city_en"].isna().any():
        missing = city_order_df.loc[city_order_df["city_en"].isna(), city_col].tolist()
        raise ValueError(
            "Some city names are not found in city_name_map: "
            + ", ".join(missing)
        )

    default_cities = city_order_df["city_en"].tolist()

    return default_cities, city_order_df


def translate_covariate_names(names, cov_name_map):
    """
    Translate covariate names to English.
    """
    return [cov_name_map.get(str(x), str(x)) for x in names]


def get_coord_values(trace, var_name, expected_length=None, default_names=None):
    """
    Try to obtain coordinate labels from the trace.
    If unavailable or generic integer coordinates are found, use default_names.
    """
    da = trace.posterior[var_name]

    for dim in da.dims:
        if dim in ["chain", "draw"]:
            continue

        if dim in trace.posterior.coords:
            values = list(trace.posterior.coords[dim].values)

            if expected_length is None or len(values) == expected_length:
                # If coordinates are generic integers, use default names.
                if default_names is not None:
                    if all(isinstance(v, (int, np.integer)) for v in values):
                        return default_names
                return [str(v) for v in values]

    if default_names is not None:
        return default_names

    if expected_length is not None:
        return [f"{var_name}_{i+1}" for i in range(expected_length)]

    n = da.shape[-1]
    return [f"{var_name}_{i+1}" for i in range(n)]


def posterior_mean_hdi(trace, var_name, hdi_prob=0.94):
    """
    Return posterior mean and HDI for a posterior variable.
    """
    da = trace.posterior[var_name]
    mean = da.mean(dim=("chain", "draw")).values
    hdi = az.hdi(da, hdi_prob=hdi_prob)[var_name].values
    return mean, hdi


def save_current_fig(path, dpi=300):
    """
    Save current matplotlib figure.
    """
    plt.gcf().savefig(path, dpi=dpi, bbox_inches="tight")



# =============================================================================
# 5. Build DEFAULT_CITIES from data/health_data_with_factors.csv
# =============================================================================

DEFAULT_CITIES, city_order_df = build_city_order_from_cov_panel(
    cov_panel_path=HEALTH_DATA_PATH,
    city_name_map=city_name_map,
    city_col="市州",
    city_id_col="city_id",
)

DEFAULT_COV_NAMES = [
    "Population density",
    "GDP per capita",
    "Annual precipitation",
    "Urbanization rate",
    "Mean elevation",
    "Annual mean temperature",
    "Annual mean relative humidity",
    "Culture/Sports/Media expenditure",
]




def diagnose_step2_results_from_traces(
    results_dict,
    factor_order=None,
    save_dir=SAVE_DIR,
    default_cov_names=DEFAULT_COV_NAMES,
    default_cities=DEFAULT_CITIES,
    factor_name_map=factor_name_map,
    cov_name_map=cov_name_map,
):
    """
    Generate Step 2 figures from posterior traces.

    Outputs:
    1. Covariate-effect forest plots
    2. Spatial random-effect bar plots
    3. Posterior predictive check plots

    This function does not print diagnostic text or export summary tables.
    """

    if factor_order is None:
        factor_order = list(results_dict.keys())

    os.makedirs(save_dir, exist_ok=True)

    for factor in factor_order:
        if factor not in results_dict:
            continue

        trace = results_dict[factor]
        factor_en = factor_name_map.get(factor, factor)

        # =====================================================
        # 1. Covariate effects: beta forest plot
        # =====================================================
        if "beta" in trace.posterior.data_vars:
            beta = trace.posterior["beta"]

            non_sample_dims = [
                d for d in beta.dims
                if d not in ["chain", "draw"]
            ]

            if len(non_sample_dims) != 1:
                raise ValueError(
                    f"Expected beta to have one non-sampling dimension, "
                    f"got {non_sample_dims}."
                )

            beta_dim = non_sample_dims[0]
            n_cov = beta.sizes[beta_dim]

            cov_names = get_coord_values(
                trace,
                var_name="beta",
                expected_length=n_cov,
                default_names=default_cov_names,
            )

            cov_names = translate_covariate_names(cov_names, cov_name_map)

            if len(cov_names) != n_cov:
                cov_names = default_cov_names[:n_cov]

            beta_mean, beta_hdi = posterior_mean_hdi(
                trace,
                "beta",
                hdi_prob=0.94,
            )

            df_beta = pd.DataFrame({
                "covariate": cov_names,
                "mean": beta_mean,
                "hdi_3": beta_hdi[:, 0],
                "hdi_97": beta_hdi[:, 1],
            })

            df_beta_plot = df_beta.iloc[::-1].reset_index(drop=True)

            plt.figure(figsize=(9, max(4, len(df_beta_plot) * 0.55)))
            plt.errorbar(
                x=df_beta_plot["mean"],
                y=np.arange(len(df_beta_plot)),
                xerr=[
                    df_beta_plot["mean"] - df_beta_plot["hdi_3"],
                    df_beta_plot["hdi_97"] - df_beta_plot["mean"],
                ],
                fmt="o",
                capsize=3,
            )

            plt.axvline(0, color="gray", linestyle="--", linewidth=1)
            plt.yticks(np.arange(len(df_beta_plot)), df_beta_plot["covariate"])
            plt.xlabel("Posterior estimate")
            plt.ylabel("Covariates")
            plt.title(f"{factor_en}: Covariate Effects")
            plt.tight_layout()

            forest_path = os.path.join(
                save_dir,
                f"Fig_{factor_en}_Forest_Beta.png"
            )
            save_current_fig(forest_path)
            plt.show()

        # =====================================================
        # 2. Spatial random effects: S_c / S_spatial / S
        # =====================================================
        spatial_var = None
        for candidate in ["S_c", "S_spatial", "S"]:
            if candidate in trace.posterior.data_vars:
                spatial_var = candidate
                break

        if spatial_var is not None:
            S_mean, S_hdi = posterior_mean_hdi(
                trace,
                spatial_var,
                hdi_prob=0.94,
            )

            n_cities = len(S_mean)

            if n_cities != len(default_cities):
                raise ValueError(
                    f"Spatial effect length mismatch: S_c has {n_cities} cities, "
                    f"but default_cities has {len(default_cities)} labels. "
                    "Please check cov_panel.csv and city_id."
                )

            df_spatial = pd.DataFrame({
                "city_id": np.arange(n_cities),
                "city": default_cities,
                "mean": S_mean,
                "hdi_3": S_hdi[:, 0],
                "hdi_97": S_hdi[:, 1],
            })

            df_spatial_plot = df_spatial.sort_values(
                "mean",
                ascending=True,
            )

            plt.figure(figsize=(12, 6))

            colors = [
                "red" if x > 0 else "blue"
                for x in df_spatial_plot["mean"]
            ]

            plt.bar(
                df_spatial_plot["city"],
                df_spatial_plot["mean"],
                color=colors,
                alpha=0.7,
            )

            plt.axhline(0, color="black", linewidth=0.8)
            plt.xticks(rotation=90, fontsize=9)
            plt.title(f"{factor_en}: Spatial Random Effects by City")
            plt.xlabel("City")
            plt.ylabel("Effect")
            plt.tight_layout()

            spatial_path = os.path.join(
                save_dir,
                f"Fig_{factor_en}_Spatial_Effects.png"
            )
            save_current_fig(spatial_path)
            plt.show()

        # =====================================================
        # 3. Posterior predictive check
        # =====================================================
        has_ppc = (
            hasattr(trace, "posterior_predictive")
            and trace.posterior_predictive is not None
        )

        if has_ppc:
            try:
                az.plot_ppc(trace, num_pp_samples=100, mean=False)
                plt.title(f"{factor_en}: Posterior Predictive Check")
                plt.xlabel("Outcome")
                plt.ylabel("Density")

                ppc_path = os.path.join(
                    save_dir,
                    f"Fig_{factor_en}_PPC.png"
                )
                save_current_fig(ppc_path)
                plt.show()

            except Exception:
                pass

diagnose_step2_results_from_traces(
    results_dict=results,
    factor_order=["力量", "速度", "耐力", "柔韧性"],
    save_dir=SAVE_DIR,
)