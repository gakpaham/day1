"""Reproducible statistical EDA and figures for BTC/ETH 3-minute bars."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "reports" / ".mplconfig-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import FIGURES_DIR, RANDOM_STATE, TARGET_NAME
from data_loader import load_all_market_data
from preprocessing import build_panel
from visualization import save_interactive_figure, save_static_figure

PALETTE = {"BTCUSDT": "#0284C7", "ETHUSDT": "#10B981"}
EDA_FEATURES = [
    "current_log_return",
    "range_log",
    "log_volume",
    "log_trades",
    "buy_ratio",
    "flow_imbalance",
]


def _reported_p_value(value: float) -> float | str:
    """Avoid presenting floating-point underflow as an exact zero p-value."""
    return "<1e-300" if value == 0 else float(value)


def _epsilon_squared(statistic: float, groups: list[np.ndarray]) -> float:
    group_count = len(groups)
    sample_count = sum(len(group) for group in groups)
    return max(0.0, float((statistic - group_count + 1) / (sample_count - group_count)))


def run_eda() -> dict:
    """Compute auditable EDA statistics and save 2D/3D figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_all_market_data()
    panel = build_panel(frames)
    rng = np.random.default_rng(RANDOM_STATE)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    profile = {}
    returns = {}
    seasonality = {}
    association_results = {}
    mutual_information = {}
    robust_anomalies = {}
    vif_results = {}

    for symbol, frame in frames.items():
        return_values = np.log(frame["close"] / frame["close"].shift(1)).dropna()
        returns[symbol] = return_values
        skew = float(stats.skew(return_values))
        excess_kurtosis = float(stats.kurtosis(return_values))
        profile[symbol] = {
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
            "missing_percent": float(frame.isna().mean().mean() * 100),
            "duplicate_rows": int(frame.duplicated().sum()),
            "zero_variance_columns": [
                column for column in frame.columns if frame[column].nunique(dropna=False) <= 1
            ],
            "first_timestamp_utc": frame["dt"].min().isoformat(),
            "last_timestamp_utc": frame["dt"].max().isoformat(),
            "return_summary": {
                "n": int(return_values.size),
                "mean": float(return_values.mean()),
                "std": float(return_values.std(ddof=1)),
                "skew": skew,
                "excess_kurtosis": excess_kurtosis,
                "q01": float(return_values.quantile(0.01)),
                "median": float(return_values.median()),
                "q99": float(return_values.quantile(0.99)),
            },
        }

        temporal = pd.DataFrame(
            {
                "dt": frame["dt"].iloc[1:].to_numpy(),
                "abs_return": return_values.abs().to_numpy(),
            }
        )
        temporal["date"] = temporal["dt"].dt.floor("D")
        temporal["hour"] = temporal["dt"].dt.hour
        temporal["weekday"] = temporal["dt"].dt.day_of_week
        daily_hour = temporal.groupby(
            ["date", "hour"], observed=True
        )["abs_return"].mean().unstack("hour")
        hour_groups = [daily_hour[hour].dropna().to_numpy() for hour in daily_hour]
        hour_stat, hour_p = stats.kruskal(*hour_groups)
        hourly_median = daily_hour.median(axis=0)

        daily = temporal.groupby("date", observed=True).agg(
            abs_return=("abs_return", "mean"),
            weekday=("weekday", "first"),
        )
        weekday_groups = [
            daily.loc[daily["weekday"] == day, "abs_return"].to_numpy()
            for day in range(7)
        ]
        weekday_stat, weekday_p = stats.kruskal(*weekday_groups)
        daily_weekday_medians = daily.groupby("weekday")["abs_return"].median()

        heat = (
            temporal.groupby(["weekday", "hour"], observed=True)["abs_return"]
            .mean()
            .unstack("hour")
            * 100
        )
        seasonality[symbol] = heat
        seasonality_summary = {
            "hour_kruskal_h": float(hour_stat),
            "hour_p_value": _reported_p_value(float(hour_p)),
            "hour_epsilon_squared": _epsilon_squared(float(hour_stat), hour_groups),
            "highest_median_hour_utc": int(hourly_median.idxmax()),
            "highest_median_daily_abs_return": float(hourly_median.max()),
            "lowest_median_hour_utc": int(hourly_median.idxmin()),
            "lowest_median_daily_abs_return": float(hourly_median.min()),
            "weekday_kruskal_h": float(weekday_stat),
            "weekday_p_value": _reported_p_value(float(weekday_p)),
            "weekday_epsilon_squared": _epsilon_squared(
                float(weekday_stat), weekday_groups
            ),
            "daily_abs_return_median_by_weekday": {
                str(int(day)): float(value)
                for day, value in daily_weekday_medians.items()
            },
        }
        profile[symbol]["seasonality"] = seasonality_summary

        model_frame = panel.loc[panel["asset"] == symbol].copy()
        model_frame["flow_imbalance"] = (
            (
                2 * frame["taker_buy_base"].to_numpy()
                - frame["volume"].to_numpy()
            )
            / frame["volume"].replace(0, np.nan).to_numpy()
        )
        valid = (
            model_frame[EDA_FEATURES + [TARGET_NAME]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        sample = valid.iloc[
            rng.choice(len(valid), min(80_000, len(valid)), replace=False)
        ]
        mi_sample = sample.iloc[: min(30_000, len(sample))]
        mi_values = mutual_info_regression(
            mi_sample[EDA_FEATURES],
            mi_sample[TARGET_NAME],
            random_state=RANDOM_STATE,
        )
        mutual_information[symbol] = {
            feature: float(value)
            for feature, value in zip(EDA_FEATURES, mi_values)
        }
        association_results[symbol] = {}
        for feature in EDA_FEATURES:
            pearson_r, pearson_p = stats.pearsonr(
                sample[feature].to_numpy(), sample[TARGET_NAME].to_numpy()
            )
            spearman_r, spearman_p = stats.spearmanr(
                sample[feature].to_numpy(), sample[TARGET_NAME].to_numpy()
            )
            association_results[symbol][feature] = {
                "pearson_r": float(pearson_r),
                "pearson_p_value": _reported_p_value(float(pearson_p)),
                "spearman_rho": float(spearman_r),
                "spearman_p_value": _reported_p_value(float(spearman_p)),
                "mutual_information_nats": mutual_information[symbol][feature],
                "sample_size": int(len(sample)),
            }

        vif_frame = (
            model_frame[EDA_FEATURES]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        vif_sample = vif_frame.iloc[
            rng.choice(len(vif_frame), min(12_000, len(vif_frame)), replace=False)
        ]
        matrix = vif_sample.to_numpy(dtype=float)
        vif_full = {
            feature: float(variance_inflation_factor(matrix, index))
            for index, feature in enumerate(EDA_FEATURES)
        }
        reduced_features = [feature for feature in EDA_FEATURES if feature != "flow_imbalance"]
        reduced_matrix = vif_sample[reduced_features].to_numpy(dtype=float)
        vif_reduced = {
            feature: float(variance_inflation_factor(reduced_matrix, index))
            for index, feature in enumerate(reduced_features)
        }
        vif_results[symbol] = {
            "all_features": vif_full,
            "without_flow_imbalance": vif_reduced,
        }

        for column, values in {
            "bar_return": return_values.to_numpy(),
            "log_volume": np.log1p(frame["volume"].to_numpy()),
        }.items():
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            robust_z = (
                0.67448975 * (values - median) / mad
                if mad
                else np.zeros_like(values)
            )
            count = int(np.sum(np.abs(robust_z) > 6))
            robust_anomalies[f"{symbol}:{column}"] = {
                "threshold": 6,
                "count": count,
                "share": float(count / len(values)),
                "n": int(len(values)),
            }

    btc_returns = returns["BTCUSDT"].to_numpy()
    eth_returns = returns["ETHUSDT"].to_numpy()
    mann_whitney = stats.mannwhitneyu(
        btc_returns, eth_returns, alternative="two-sided", method="asymptotic"
    )
    rank_biserial = (
        2 * mann_whitney.statistic / (len(btc_returns) * len(eth_returns)) - 1
    )

    merged = (
        panel[["dt", "asset", "bar_return"]]
        .pivot(index="dt", columns="asset", values="bar_return")
        .dropna()
        .sort_index()
    )
    pearson = stats.pearsonr(merged["BTCUSDT"], merged["ETHUSDT"])
    spearman = stats.spearmanr(merged["BTCUSDT"], merged["ETHUSDT"])
    btc_leads_eth = stats.pearsonr(
        merged["BTCUSDT"].iloc[:-1], merged["ETHUSDT"].iloc[1:]
    )
    eth_leads_btc = stats.pearsonr(
        merged["ETHUSDT"].iloc[:-1], merged["BTCUSDT"].iloc[1:]
    )

    summary = {
        "target_definition": (
            "Exploratory only: log(close[t+1] / close[t]); features are available "
            "at the close of bar t."
        ),
        "inference_caveat": (
            "Mann-Whitney, Kruskal-Wallis, Pearson and Spearman p-values assume "
            "independent observations; serial dependence makes them descriptive."
        ),
        "dataset_profile": profile,
        "btc_vs_eth_return_distribution": {
            "mann_whitney_u": float(mann_whitney.statistic),
            "p_value": _reported_p_value(float(mann_whitney.pvalue)),
            "rank_biserial_effect": float(rank_biserial),
        },
        "mutual_information_nats": mutual_information,
        "feature_associations": association_results,
        "cross_asset_returns": {
            "aligned_n": int(len(merged)),
            "pearson_r": float(pearson.statistic),
            "pearson_p_value": _reported_p_value(float(pearson.pvalue)),
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value": _reported_p_value(float(spearman.pvalue)),
            "btc_t_to_eth_t_plus_1_r": float(btc_leads_eth.statistic),
            "btc_t_to_eth_t_plus_1_p_value": _reported_p_value(
                float(btc_leads_eth.pvalue)
            ),
            "eth_t_to_btc_t_plus_1_r": float(eth_leads_btc.statistic),
            "eth_t_to_btc_t_plus_1_p_value": _reported_p_value(
                float(eth_leads_btc.pvalue)
            ),
        },
        "vif": vif_results,
        "robust_anomalies": robust_anomalies,
    }

    # Return distributions: clipping is limited to display, not analysis.
    fig, axis = plt.subplots(figsize=(11, 6))
    for symbol, values in returns.items():
        array = values.to_numpy()
        lower, upper = np.quantile(array, [0.005, 0.995])
        sns.histplot(
            array[(array >= lower) & (array <= upper)],
            bins=120,
            stat="density",
            element="step",
            fill=False,
            linewidth=1.5,
            color=PALETTE[symbol],
            label=f"{symbol} (central 99% shown)",
            ax=axis,
        )
    axis.set_title("ETH has wider 3-minute return dispersion than BTC", fontsize=15, pad=14)
    axis.set_xlabel("Close-to-close log return per 3-minute bar")
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    axis.text(
        0.01,
        -0.20,
        "Key visual takeaway: Central distributions are shown without tail compression; "
        "extreme quantiles are reported separately.",
        transform=axis.transAxes,
        fontsize=9,
        color="#334155",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_static_figure(fig, FIGURES_DIR / "return_distribution.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    for axis, (symbol, heat) in zip(axes, seasonality.items()):
        sns.heatmap(
            heat,
            ax=axis,
            cmap="mako",
            cbar=axis is axes[-1],
            annot=False,
            linewidths=0.15,
            linecolor="white",
        )
        axis.set_title(
            f"{symbol}: absolute-return intensity varies by UTC window",
            fontsize=12,
            fontweight="bold",
        )
        axis.set_xlabel("Hour of day (UTC)")
        axis.set_ylabel("Weekday (Mon=0)" if axis is axes[0] else "")
    fig.text(
        0.01,
        0.01,
        "Key visual takeaway: Brighter cells mark higher mean absolute 3-minute returns; "
        "this is descriptive seasonality, not a trading signal.",
        fontsize=9,
        color="#334155",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_static_figure(fig, FIGURES_DIR / "temporal_volatility_heatmap.png")
    plt.close(fig)

    rolling = merged["BTCUSDT"].rolling(
        30 * 24 * 20, min_periods=15 * 24 * 20
    ).corr(merged["ETHUSDT"])
    rolling_daily = rolling.resample("D").last().dropna()
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(rolling_daily.index, rolling_daily.values, color="#0284C7", linewidth=1.5)
    axis.axhline(0, color="#64748B", linewidth=0.8)
    axis.set_ylim(-1, 1)
    axis.set_title("BTC–ETH co-movement evolves across market regimes", fontsize=15, pad=14)
    axis.set_xlabel("Date (UTC)")
    axis.set_ylabel("30-day rolling Pearson correlation of 3-minute returns")
    axis.text(
        0.01,
        -0.22,
        "Key visual takeaway: Rolling correlation summarizes co-movement dynamics; "
        "large-sample p-values are not independent-trial evidence.",
        transform=axis.transAxes,
        fontsize=9,
        color="#334155",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_static_figure(fig, FIGURES_DIR / "rolling_30d_cross_asset_correlation.png")
    plt.close(fig)

    mi_frame = pd.DataFrame(mutual_information).sort_index()
    fig, axis = plt.subplots(figsize=(10, 6))
    mi_frame.plot(
        kind="barh",
        ax=axis,
        color=[PALETTE["BTCUSDT"], PALETTE["ETHUSDT"]],
        width=0.78,
    )
    axis.invert_yaxis()
    axis.set_title(
        "Current-bar range ranks highest by estimated information about next-bar return",
        fontsize=13,
        pad=14,
    )
    axis.set_xlabel("Estimated mutual information (nats; sampled)")
    axis.set_ylabel("Feature at time t")
    axis.legend(title="Asset", frameon=False)
    axis.text(
        0.01,
        -0.20,
        "Key visual takeaway: Mutual information captures nonlinear association, "
        "not causation or out-of-sample predictability.",
        transform=axis.transAxes,
        fontsize=9,
        color="#334155",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_static_figure(fig, FIGURES_DIR / "mutual_information_future_return.png")
    plt.close(fig)

    scatter_parts = []
    for symbol in frames:
        asset_rows = panel.loc[panel["asset"] == symbol].copy()
        asset_rows["log_volume"] = np.log1p(
            frames[symbol]["volume"].to_numpy()
        )
        asset_rows = asset_rows[
            ["dt", "asset", "range_log", "log_volume", TARGET_NAME]
        ].dropna()
        sample_size = min(3_500, len(asset_rows))
        scatter_parts.append(
            asset_rows.iloc[rng.choice(len(asset_rows), sample_size, replace=False)]
        )
    cloud = pd.concat(scatter_parts, ignore_index=True)
    figure_3d = px.scatter_3d(
        cloud,
        x="range_log",
        y="log_volume",
        z=TARGET_NAME,
        color="asset",
        color_discrete_map=PALETTE,
        opacity=0.55,
        hover_data={
            "dt": True,
            "range_log": ":.5g",
            "log_volume": ":.4g",
            TARGET_NAME: ":.5g",
        },
        title="Current-bar range and activity versus the following 3-minute return",
    )
    figure_3d.update_traces(
        marker={"size": 2.5},
        hovertemplate=(
            "Asset: %{legendgroup}<br>Range log: %{x:.5g}"
            "<br>log(1+volume): %{y:.5g}<br>Next 3m log return: %{z:.5g}"
            "<br>Date UTC: %{customdata[0]}<extra></extra>"
        ),
    )
    figure_3d.update_layout(
        scene={
            "xaxis_title": "Current bar log(high/low)",
            "yaxis_title": "Current bar log(1 + volume)",
            "zaxis_title": "Next 3-minute log return",
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.15}},
        },
        margin={"l": 0, "r": 0, "t": 65, "b": 100},
    )
    figure_3d.add_annotation(
        x=0.01,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=(
            "Key visual takeaway: The next-bar return is an exploratory response; "
            "scatter does not establish a profitable strategy."
        ),
        showarrow=False,
        align="left",
    )
    save_interactive_figure(
        figure_3d,
        FIGURES_DIR / "current_bar_vs_future_return_3d.html",
    )

    summary_path = ROOT / "reports" / "eda_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    run_eda()
