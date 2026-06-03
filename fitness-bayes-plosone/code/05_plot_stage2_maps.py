# =========================================================
# Sichuan prefecture-level map plotting
# Based on health_data_with_factors.csv + Step2 nc traces
# Raw factor mean vs. spatial random effect
# =========================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# =========================================================
# 0. Basic configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SHP_PATH, HEALTH_DATA_PATH, TRACE_FILE_MAP, FIGURE_STAGE2_DIR

TRACE_PATHS = TRACE_FILE_MAP
FIGURE_STAGE2_DIR.mkdir(parents=True, exist_ok=True)

FACTOR_PLOT_ORDER = ["力量", "速度", "耐力", "柔韧性"]

FACTOR_EN_MAP = {
    "力量": "Strength",
    "速度": "Speed",
    "耐力": "Endurance",
    "柔韧性": "Flexibility",
}

FACTOR_COL_MAP = {
    "力量": "F_strength",
    "速度": "F_speed",
    "耐力": "F_endurance",
    "柔韧性": "F_flexibility",
}

COMMON_CMAP = "RdYlBu_r"


# =========================================================
# 1. City information table for shapefile
# =========================================================

city_info = pd.DataFrame(
    [
        ["chengdu",    "成都市",             510100, 20],
        ["zigong",     "自贡市",             510300, 6],
        ["panzhihua",  "攀枝花市",           510400, 5],
        ["luzhou",     "泸州市",             510500, 7],
        ["deyang",     "德阳市",             510600, 6],
        ["mianyang",   "绵阳市",             510700, 9],
        ["guangyuan",  "广元市",             510800, 7],
        ["suining",    "遂宁市",             510900, 5],
        ["neijiang",   "内江市",             511000, 5],
        ["leshan",     "乐山市",             511100, 11],
        ["nanchong",   "南充市",             511300, 9],
        ["meishan",    "眉山市",             511400, 6],
        ["yibin",      "宜宾市",             511500, 10],
        ["guangan",    "广安市",             511600, 6],
        ["dazhou",     "达州市",             511700, 7],
        ["yaan",       "雅安市",             511800, 8],
        ["bazhong",    "巴中市",             511900, 5],
        ["ziyang",     "资阳市",             512000, 3],
        ["aba",        "阿坝藏族羌族自治州", 513200, 13],
        ["ganzi",      "甘孜藏族自治州",     513300, 18],
        ["liangshan",  "凉山彝族自治州",     513400, 17],
    ],
    columns=["city_key", "city_name_ch", "adcode", "childrenNu"],
)


# =========================================================
# 2. Utility functions
# =========================================================

def read_health_data(path=HEALTH_DATA_PATH):
    try:
        df = pd.read_csv(path, encoding="gbk")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gb18030")

    required_cols = [
        "市州", "city_id",
        "F_strength", "F_speed", "F_endurance", "F_flexibility"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if len(missing) > 0:
        raise KeyError(
            f"health_data_with_factors.csv 缺少必要列: {missing}\n"
            f"当前列名: {df.columns.tolist()}"
        )

    return df


def clean_city_name(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    suffixes = [
        "教育和体育局",
        "教育体育局",
        "教育局",
    ]

    for s in suffixes:
        x = x.replace(s, "")

    alias_map = {
        "阿坝州": "阿坝藏族羌族自治州",
        "阿坝": "阿坝藏族羌族自治州",
        "甘孜州": "甘孜藏族自治州",
        "甘孜": "甘孜藏族自治州",
        "凉山州": "凉山彝族自治州",
        "凉山": "凉山彝族自治州",
    }

    x = alias_map.get(x, x)

    return x.strip()


def safe_two_slope_norm(values, center=0.0):
    values = pd.Series(values).dropna()

    if len(values) == 0:
        return None

    vmin = values.min()
    vmax = values.max()

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None

    if vmin < center < vmax:
        return colors.TwoSlopeNorm(
            vmin=vmin,
            vcenter=center,
            vmax=vmax
        )

    return None


# =========================================================
# 3. Build city-level GeoDataFrame from shapefile
# =========================================================

def build_city_geodataframe(shp_path=SHP_PATH, city_info=city_info):
    gdf_county = gpd.read_file(shp_path, encoding="utf-8")

    if "layer" not in gdf_county.columns:
        raise KeyError(
            "shapefile 中必须包含 'layer' 字段，例如 chengdu_all。\n"
            f"当前字段: {gdf_county.columns.tolist()}"
        )

    gdf_county = gdf_county.copy()

    gdf_county["city_key"] = (
        gdf_county["layer"]
        .astype(str)
        .str.replace("_all", "", regex=False)
        .str.lower()
        .str.strip()
    )

    gdf_county = gdf_county.merge(
        city_info,
        on="city_key",
        how="left",
        validate="many_to_one",
    )

    unmatched = (
        gdf_county.loc[gdf_county["city_name_ch"].isna(), ["layer", "city_key"]]
        .drop_duplicates()
    )

    if len(unmatched) > 0:
        raise ValueError(
            "存在未匹配的 shapefile 城市，请检查 city_key：\n"
            f"{unmatched}"
        )

    if gdf_county.crs is None:
        gdf_county = gdf_county.set_crs(epsg=4326)

    try:
        gdf_county["geometry"] = gdf_county.geometry.make_valid()
    except Exception:
        gdf_county["geometry"] = gdf_county.geometry.buffer(0)

    gdf_city = gdf_county.dissolve(
        by="city_key",
        as_index=False,
        aggfunc={
            "adcode": "first",
            "city_name_ch": "first",
            "childrenNu": "first",
        },
    )

    gdf_city["center"] = None
    gdf_city["centroid"] = None
    gdf_city["level"] = "city"
    gdf_city["parent"] = None
    gdf_city["acroutes"] = None

    gdf_city = gdf_city.sort_values("adcode").reset_index(drop=True)
    gdf_city["subFeature"] = gdf_city.index

    gdf_city = gdf_city[
        [
            "adcode",
            "city_name_ch",
            "center",
            "centroid",
            "childrenNu",
            "level",
            "parent",
            "subFeature",
            "acroutes",
            "geometry",
        ]
    ].rename(columns={"city_name_ch": "name"})

    gdf_city["city_clean"] = gdf_city["name"].apply(clean_city_name)

    return gdf_city


# =========================================================
# 4. Prepare health factor data
# =========================================================

def prepare_health_factor_data(health_df):
    df = health_df.copy()

    df["city_clean"] = df["市州"].apply(clean_city_name)

    city_lookup = (
        df[["city_id", "市州", "city_clean"]]
        .drop_duplicates()
        .sort_values("city_id")
        .reset_index(drop=True)
    )

    check = city_lookup.groupby("city_id")["city_clean"].nunique()
    bad_ids = check[check > 1]

    if len(bad_ids) > 0:
        raise ValueError(
            "存在一个 city_id 对应多个城市名称的情况，请检查数据：\n"
            f"{bad_ids}"
        )

    if city_lookup["city_id"].nunique() != 21:
        raise ValueError(
            f"当前 city_id 数量为 {city_lookup['city_id'].nunique()}，不是 21。"
        )

    return df, city_lookup


# =========================================================
# 5. Load Step-2 posterior traces
# =========================================================

def load_step2_results(trace_paths):
    results = {}

    for factor, path in trace_paths.items():
        idata_full = az.from_netcdf(path)
        results[factor] = {
            "full(spatial+time)": idata_full
        }

    return results


# =========================================================
# 6. Extract raw means and spatial effects
# =========================================================

def extract_city_data_from_health(
    idata,
    factor_name,
    health_df,
    city_lookup,
    spatial_var="S_c",
):
    if factor_name not in FACTOR_COL_MAP:
        raise KeyError(f"未知因子名称: {factor_name}")

    raw_source_col = FACTOR_COL_MAP[factor_name]
    raw_col = f"Raw_{factor_name}"
    eff_col = f"S_{factor_name}"

    raw_means = (
        health_df
        .groupby(["city_id", "city_clean"], as_index=False)[raw_source_col]
        .mean()
        .rename(columns={raw_source_col: raw_col})
    )

    if spatial_var not in idata.posterior:
        raise KeyError(
            f"idata.posterior 中找不到空间随机效应变量 '{spatial_var}'。\n"
            f"当前变量包括: {list(idata.posterior.data_vars)}"
        )

    S_da = idata.posterior[spatial_var]
    S_post = S_da.mean(dim=("chain", "draw")).values
    S_post = np.asarray(S_post).reshape(-1)

    if len(S_post) != len(city_lookup):
        raise ValueError(
            "空间随机效应 S_c 的长度与 city_lookup 城市数量不一致。\n"
            f"len(S_post) = {len(S_post)}, len(city_lookup) = {len(city_lookup)}。"
        )

    coord_candidates = ["city_id", "city", "cities", "city_idx", "city_name_ch"]

    coord_values = None

    for coord_name in coord_candidates:
        if coord_name in S_da.coords:
            vals = S_da.coords[coord_name].values
            if len(vals) == len(S_post):
                coord_values = vals
                break

    if coord_values is not None:
        if np.issubdtype(np.asarray(coord_values).dtype, np.number):
            df_spatial = pd.DataFrame({
                "city_id": coord_values.astype(int),
                eff_col: S_post,
            })

            df_spatial = df_spatial.merge(
                city_lookup[["city_id", "city_clean"]],
                on="city_id",
                how="left",
                validate="one_to_one",
            )

        else:
            df_spatial = pd.DataFrame({
                "city_clean": pd.Series(coord_values).apply(clean_city_name),
                eff_col: S_post,
            })

            df_spatial = df_spatial.merge(
                city_lookup[["city_id", "city_clean"]],
                on="city_clean",
                how="left",
                validate="one_to_one",
            )

    else:
        df_spatial = city_lookup[["city_id", "city_clean"]].copy()
        df_spatial[eff_col] = S_post

    if df_spatial["city_clean"].isna().any():
        raise ValueError(
            "部分 S_c 城市无法匹配，请检查 nc 文件中的城市坐标。"
        )

    df_final = raw_means.merge(
        df_spatial[["city_id", "city_clean", eff_col]],
        on=["city_id", "city_clean"],
        how="left",
        validate="one_to_one",
    )

    return df_final


# =========================================================
# 7. Plot function
# =========================================================

def plot_raw_and_spatial_effects(
    gdf_city,
    health_df,
    city_lookup,
    results_step2,
    factor_plot_order=FACTOR_PLOT_ORDER,
    factor_en_map=FACTOR_EN_MAP,
    cmap=COMMON_CMAP,
):
    gdf_plot = gdf_city.copy()

    fig, axes = plt.subplots(
        nrows=len(factor_plot_order),
        ncols=2,
        figsize=(16, 6 * len(factor_plot_order)),
        layout="constrained",
    )

    if len(factor_plot_order) == 1:
        axes = np.asarray(axes).reshape(1, 2)

    for i, factor in enumerate(factor_plot_order):
        factor_en = factor_en_map.get(factor, factor)

        idata_curr = results_step2[factor]["full(spatial+time)"]

        df_factor = extract_city_data_from_health(
            idata=idata_curr,
            factor_name=factor,
            health_df=health_df,
            city_lookup=city_lookup,
            spatial_var="S_c",
        )

        gdf_merged = gdf_plot.merge(
            df_factor,
            on="city_clean",
            how="left",
            validate="one_to_one",
        )

        raw_col = f"Raw_{factor}"
        eff_col = f"S_{factor}"

        if gdf_merged[raw_col].isna().any():
            raise ValueError(f"{factor_en} 的 Raw Mean 存在城市匹配缺失。")

        if gdf_merged[eff_col].isna().any():
            raise ValueError(f"{factor_en} 的 Spatial Effect 存在城市匹配缺失。")

        # -----------------------------
        # A. Raw mean map
        # -----------------------------
        raw_values = gdf_merged[raw_col].dropna()
        raw_center = raw_values.mean() if len(raw_values) > 0 else 0.0
        norm_raw = safe_two_slope_norm(raw_values, center=raw_center)

        ax_raw = axes[i, 0]

        gdf_merged.plot(
            column=raw_col,
            cmap=cmap,
            norm=norm_raw,
            linewidth=0.8,
            edgecolor="0.9",
            legend=True,
            legend_kwds={
                "shrink": 0.6,
                "label": "Observed factor score",
            },
            ax=ax_raw,
        )

        ax_raw.set_title(
            f"Factor [{factor_en}]: Raw Mean\n"
            f"(Red > Provincial Mean, Blue < Provincial Mean)",
            fontsize=14,
        )
        ax_raw.axis("off")

        # -----------------------------
        # B. Spatial random effect map
        # -----------------------------
        eff_values = gdf_merged[eff_col].dropna()

        if len(eff_values) > 0:
            eff_max_abs = np.nanmax(np.abs(eff_values))
        else:
            eff_max_abs = np.nan

        if np.isfinite(eff_max_abs) and eff_max_abs > 0:
            norm_eff = colors.TwoSlopeNorm(
                vmin=-eff_max_abs,
                vcenter=0,
                vmax=eff_max_abs,
            )
        else:
            norm_eff = None

        ax_eff = axes[i, 1]

        gdf_merged.plot(
            column=eff_col,
            cmap=cmap,
            norm=norm_eff,
            linewidth=0.8,
            edgecolor="0.9",
            legend=True,
            legend_kwds={
                "shrink": 0.6,
                "label": "Spatial random effect",
            },
            ax=ax_eff,
        )

        ax_eff.set_title(
            f"Factor [{factor_en}]: Spatial Random Effect\n"
            f"(Red = Positive Spatial Effect, Blue = Negative Spatial Effect)",
            fontsize=14,
        )
        ax_eff.axis("off")

    plt.show()

    return fig, axes




# =========================================================
# 8. Run all
# =========================================================

gdf_city = build_city_geodataframe(SHP_PATH)

health_df_raw = read_health_data(HEALTH_DATA_PATH)

health_df, city_lookup = prepare_health_factor_data(health_df_raw)

results_step2 = load_step2_results(TRACE_PATHS)

fig, axes = plot_raw_and_spatial_effects(
    gdf_city=gdf_city,
    health_df=health_df,
    city_lookup=city_lookup,
    results_step2=results_step2,
    factor_plot_order=FACTOR_PLOT_ORDER,
)

fig.savefig(FIGURE_STAGE2_DIR / "Fig_raw_mean_vs_spatial_random_effects.png", dpi=300, bbox_inches="tight")
