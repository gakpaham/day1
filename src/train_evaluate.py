"""Temporal tuning, out-of-fold evaluation, diagnostics, and model export."""

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "reports" / ".mplconfig-cache")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURES,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_NAME,
)
from data_loader import load_all_market_data
from preprocessing import build_panel
from visualization import save_static_figure

MODEL_NAMES = [
    "Ridge",
    "HistGradientBoostingRegressor",
    "LGBMRegressor",
    "CatBoostRegressor",
]
CPU_THREADS = max(1, min(4, (os.cpu_count() or 2) - 1))


def build_pipeline(name: str, parameters: dict | None = None) -> Pipeline:
    """Build a model pipeline whose imputation/scaling is fit within each fold."""
    parameters = parameters or {}
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name == "Ridge":
        steps.append(("scaler", StandardScaler()))
        estimator = Ridge(alpha=parameters.get("alpha", 1.0), solver="cholesky")
    elif name == "HistGradientBoostingRegressor":
        estimator = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=parameters.get("max_iter", 60),
            learning_rate=parameters.get("learning_rate", 0.06),
            max_leaf_nodes=parameters.get("max_leaf_nodes", 31),
            l2_regularization=parameters.get("l2_regularization", 1.0),
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    elif name == "LGBMRegressor":
        estimator = LGBMRegressor(
            objective="regression_l1",
            n_estimators=parameters.get("n_estimators", 60),
            learning_rate=parameters.get("learning_rate", 0.05),
            num_leaves=parameters.get("num_leaves", 31),
            min_child_samples=parameters.get("min_child_samples", 100),
            reg_lambda=parameters.get("reg_lambda", 1.0),
            n_jobs=CPU_THREADS,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            random_state=RANDOM_STATE,
        )
    elif name == "CatBoostRegressor":
        estimator = CatBoostRegressor(
            loss_function="MAE",
            iterations=parameters.get("iterations", 60),
            depth=parameters.get("depth", 7),
            learning_rate=parameters.get("learning_rate", 0.05),
            l2_leaf_reg=parameters.get("l2_leaf_reg", 3.0),
            thread_count=CPU_THREADS,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
    else:
        raise ValueError(f"Unsupported model: {name}")
    steps.append(("model", estimator))
    return Pipeline(steps)


def suggest_parameters(name: str, trial: optuna.Trial) -> dict:
    """Suggest a bounded search configuration for one estimator family."""
    if name == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True)}
    if name == "HistGradientBoostingRegressor":
        return {
            "max_iter": trial.suggest_int("max_iter", 40, 80, step=20),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.025, 0.12, log=True
            ),
            "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [15, 31]),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-3, 10.0, log=True
            ),
        }
    if name == "LGBMRegressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 40, 100, step=20),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.1, log=True
            ),
            "num_leaves": trial.suggest_categorical("num_leaves", [15, 31]),
            "min_child_samples": trial.suggest_categorical(
                "min_child_samples", [50, 100, 200]
            ),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "CatBoostRegressor":
        return {
            "iterations": trial.suggest_int("iterations", 40, 100, step=20),
            "depth": trial.suggest_int("depth", 5, 7),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.1, log=True
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
        }
    raise ValueError(f"Unsupported model: {name}")


def row_indices_for_times(
    time_codes: np.ndarray, selected_time_codes: np.ndarray
) -> np.ndarray:
    """Map unique timestamp positions back to every asset row at those times."""
    mask = np.isin(time_codes, selected_time_codes)
    return np.flatnonzero(mask)


def calculate_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float | int | None]:
    """Return regression and direction metrics for an aligned prediction vector."""
    rank_ic = float(stats.spearmanr(actual, predicted).statistic)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "spearman_ic": rank_ic if np.isfinite(rank_ic) else None,
        "directional_accuracy": float(
            np.mean(np.sign(actual) == np.sign(predicted))
        ),
        "n": int(len(actual)),
    }


def zero_return_baseline(actual: np.ndarray) -> dict[str, float]:
    """Score the constant zero-return forecast using MAE and RMSE only."""
    return {
        "mae": float(np.mean(np.abs(actual))),
        "rmse": float(np.sqrt(np.mean(np.square(actual)))),
    }


def metric_by_asset(
    actual: pd.Series, predicted: pd.Series, assets: pd.Series
) -> dict[str, dict[str, float | int | None]]:
    """Calculate the same metrics separately for each asset."""
    result = {}
    for symbol in sorted(assets.unique()):
        mask = assets.to_numpy() == symbol
        result[symbol] = calculate_metrics(
            actual.to_numpy()[mask], predicted.to_numpy()[mask]
        )
    return result


def tune_model(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    trials: int,
) -> tuple[dict, float]:
    """Tune on a strictly later validation window, separate from OOF folds."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=f"{name}_future_return",
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_parameters(name, trial)
        pipeline = build_pipeline(name, parameters)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_valid)
        return float(mean_absolute_error(y_valid, predictions))

    study.optimize(objective, n_trials=trials, n_jobs=1)
    return dict(study.best_params), float(study.best_value)


def save_residual_diagnostics(
    actual: np.ndarray, predicted: np.ndarray, model_name: str
) -> None:
    """Save residual and normal-QQ diagnostics for the final chronological holdout."""
    residuals = actual - predicted
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(predicted, residuals, s=5, alpha=0.25, color="#0284C7")
    axes[0].axhline(0, color="#E11D48", linewidth=1)
    axes[0].set_title(f"{model_name}: holdout residuals vs prediction")
    axes[0].set_xlabel("Predicted next-bar log return")
    axes[0].set_ylabel("Residual")
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Residual QQ plot")
    fig.tight_layout()
    save_static_figure(fig, FIGURES_DIR / "residual_diagnostics.png")
    plt.close(fig)


def save_shap_diagnostics(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_explain: pd.DataFrame,
) -> None:
    """Save SHAP summary, importance, and one waterfall for the selected model."""
    background = x_train.sample(
        n=min(100, len(x_train)), random_state=RANDOM_STATE
    )
    explain = x_explain.sample(
        n=min(250, len(x_explain)), random_state=RANDOM_STATE
    )
    explainer = shap.Explainer(
        pipeline.predict,
        background,
        algorithm="permutation",
        feature_names=FEATURES,
    )
    values = explainer(explain, max_evals=2 * len(FEATURES) + 1)

    shap.plots.beeswarm(values, show=False, max_display=15)
    save_static_figure(plt.gcf(), FIGURES_DIR / "shap_summary.png")
    plt.close(plt.gcf())

    shap.plots.bar(values, show=False, max_display=15)
    save_static_figure(plt.gcf(), FIGURES_DIR / "shap_importance_bar.png")
    plt.close(plt.gcf())

    shap.plots.waterfall(values[0], show=False, max_display=15)
    save_static_figure(plt.gcf(), FIGURES_DIR / "shap_waterfall.png")
    plt.close(plt.gcf())


def write_readme(metrics: dict) -> None:
    """Write a portfolio README from the verified experiment results."""
    leaderboard = metrics["leaderboard"]
    table_rows = "\n".join(
        f"| {row['model']} | {row['oof']['mae']:.7g} | "
        f"{row.get('oof_mae_skill_vs_zero', float('nan')):.4%} | "
        f"{row['oof']['rmse']:.7g} | {row['oof']['spearman_ic']:.5g} | "
        f"{row['holdout']['mae']:.7g} |"
        for row in leaderboard
    )
    text = f"""# BTC–ETH 3-Minute Quantitative Research

Leakage-aware research on BTCUSDT and ETHUSDT 3-minute OHLCV bars. The
exploratory target is the next-bar log return, `log(close[t+1] / close[t])`.
This project is research, not investment advice or a claim of profitability.

## Data and scope

- Inputs: `data/processed/BTCUSDT_3m.parquet` and
  `data/processed/ETHUSDT_3m.parquet` (confirmed FINAL; source files are read-only).
- Combined observations: {metrics['data']['rows']:,}; time range:
  {metrics['data']['start_utc']} through {metrics['data']['end_utc']}.
- Validation is chronological. All assets at a timestamp stay in the same fold;
  one bar is purged at boundaries because the target uses the following close.
- Missing-value imputation and Ridge scaling are fitted inside each CV pipeline.
- Optuna searches use a deterministic one-in-ten timestamp training sample;
  final OOF folds and holdout evaluation use the full row set.
- OOF covers five expanding-window folds after the tuning period. The last 10%
  of unique timestamps is a separate chronological holdout.

## Leaderboard

Primary metric is MAE in log-return units. RMSE, rank IC, and direction accuracy
are secondary diagnostics; they do not include transaction costs or slippage.

| Model | OOF MAE | OOF MAE skill vs zero | OOF RMSE | OOF rank IC | Holdout MAE |
|---|---:|---:|---:|---:|---:|
{table_rows}

## Reproducibility

Use Python 3.14 and install dependencies from `requirements.txt`.

```powershell
python src\\eda_analysis.py
python src\\train_evaluate.py --trials {metrics['experiment']['optuna_trials_per_model']}
python src\\build_notebook.py
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 --inplace notebooks\\analysis.ipynb
```

See `reports/eda_summary.json`, `reports/metrics_summary.json`, and
`reports/oof_predictions.parquet` for machine-readable results. Figures are in
`reports/figures/`; the selected fitted pipeline is in `models/`.

## Interpretation limits

Returns are heavy-tailed and temporally dependent. Large-sample correlation
p-values are descriptive; use time-aware validation, cost/slippage modeling,
and regime robustness checks before any trading interpretation. Current data
end at {metrics['data']['end_utc']}.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def run_experiment(trials: int = 3, n_splits: int = 5) -> dict:
    """Run Optuna tuning, temporal OOF, final holdout, SHAP, and exports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_all_market_data()
    panel = build_panel(frames)
    panel = panel.dropna(subset=[TARGET_NAME]).reset_index(drop=True)
    times = pd.Index(panel["dt"].drop_duplicates().sort_values())
    time_lookup = {timestamp: index for index, timestamp in enumerate(times)}
    time_codes = panel["dt"].map(time_lookup).to_numpy(dtype=np.int64)
    holdout_start = int(len(times) * 0.90)
    dev_times = times[:holdout_start]
    tune_train_end = int(len(dev_times) * 0.60)
    tune_valid_end = int(len(dev_times) * 0.70)
    if tune_train_end < 2 or tune_valid_end <= tune_train_end:
        raise ValueError("Not enough unique timestamps for the requested temporal split.")

    x = panel[FEATURES]
    y = panel[TARGET_NAME]
    asset = panel["asset"]

    tuning_train_times = np.arange(0, tune_train_end - 1, dtype=np.int64)
    tuning_valid_times = np.arange(tune_train_end, tune_valid_end, dtype=np.int64)
    tuning_train_rows = row_indices_for_times(time_codes, tuning_train_times)
    tuning_valid_rows = row_indices_for_times(time_codes, tuning_valid_times)
    tuning_stride = 10
    tuning_train_rows = tuning_train_rows[
        time_codes[tuning_train_rows] % tuning_stride == 0
    ]
    x_tune_train = x.iloc[tuning_train_rows]
    y_tune_train = y.iloc[tuning_train_rows]
    x_tune_valid = x.iloc[tuning_valid_rows]
    y_tune_valid = y.iloc[tuning_valid_rows]

    baseline = build_pipeline("Ridge", {"alpha": 1.0})
    baseline.fit(x_tune_train, y_tune_train)
    baseline_tuning_metrics = calculate_metrics(
        y_tune_valid.to_numpy(), baseline.predict(x_tune_valid)
    )
    del baseline

    tuned_parameters = {}
    tuning_scores = {}
    for name in MODEL_NAMES:
        started = time.perf_counter()
        tuned_parameters[name], tuning_scores[name] = tune_model(
            name,
            x_tune_train,
            y_tune_train,
            x_tune_valid,
            y_tune_valid,
            trials,
        )
        print(
            f"Tuned {name}: validation MAE={tuning_scores[name]:.7g}; "
            f"best_params={tuned_parameters[name]}; "
            f"elapsed_seconds={time.perf_counter() - started:.1f}"
        )

    test_size = max(1, (len(dev_times) - tune_valid_end) // n_splits)
    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
        gap=1,
    )
    folds = list(splitter.split(np.arange(len(dev_times))))
    if len(folds) != n_splits:
        raise ValueError(f"Expected {n_splits} temporal folds, got {len(folds)}.")

    oof_frames = []
    oof_metrics = {}
    for name in MODEL_NAMES:
        started = time.perf_counter()
        prediction_parts = []
        for fold_index, (train_times, valid_times) in enumerate(folds, start=1):
            train_rows = row_indices_for_times(time_codes, train_times)
            valid_rows = row_indices_for_times(time_codes, valid_times)
            pipeline = build_pipeline(name, tuned_parameters[name])
            pipeline.fit(x.iloc[train_rows], y.iloc[train_rows])
            predictions = pipeline.predict(x.iloc[valid_rows])
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "row_index": valid_rows,
                        "fold": fold_index,
                        "prediction": predictions,
                    }
                )
            )
            print(
                f"OOF {name}: fold {fold_index}/{n_splits}; "
                f"train_rows={len(train_rows):,}; valid_rows={len(valid_rows):,}"
            )
            del pipeline

        model_predictions = pd.concat(prediction_parts, ignore_index=True)
        model_predictions = model_predictions.sort_values("row_index")
        row_indices = model_predictions["row_index"].to_numpy(dtype=np.int64)
        predicted = model_predictions["prediction"].to_numpy(dtype=float)
        actual = y.iloc[row_indices].to_numpy(dtype=float)
        oof_metrics[name] = {
            "overall": calculate_metrics(actual, predicted),
            "by_asset": metric_by_asset(
                y.iloc[row_indices].reset_index(drop=True),
                pd.Series(predicted),
                asset.iloc[row_indices].reset_index(drop=True),
            ),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        oof_frames.append(
            pd.DataFrame(
                {
                    "dt": panel["dt"].iloc[row_indices].reset_index(drop=True),
                    "asset": asset.iloc[row_indices].to_numpy(),
                    "fold": model_predictions["fold"].to_numpy(),
                    "target": actual,
                    "model": name,
                    "prediction": predicted,
                }
            )
        )
        print(
            f"OOF {name}: {oof_metrics[name]['overall']}; "
            f"elapsed_seconds={oof_metrics[name]['elapsed_seconds']:.1f}"
        )

    oof_predictions = pd.concat(oof_frames, ignore_index=True)
    oof_predictions.to_parquet(REPORTS_DIR / "oof_predictions.parquet", index=False)
    best_model_name = min(
        MODEL_NAMES, key=lambda model: oof_metrics[model]["overall"]["mae"]
    )

    final_train_times = np.arange(0, holdout_start - 1, dtype=np.int64)
    final_holdout_times = np.arange(holdout_start, len(times), dtype=np.int64)
    final_train_rows = row_indices_for_times(time_codes, final_train_times)
    holdout_rows = row_indices_for_times(time_codes, final_holdout_times)
    holdout_metrics = {}
    final_pipelines = {}
    for name in MODEL_NAMES:
        pipeline = build_pipeline(name, tuned_parameters[name])
        pipeline.fit(x.iloc[final_train_rows], y.iloc[final_train_rows])
        predicted = pipeline.predict(x.iloc[holdout_rows])
        holdout_metrics[name] = {
            "overall": calculate_metrics(y.iloc[holdout_rows].to_numpy(), predicted),
            "by_asset": metric_by_asset(
                y.iloc[holdout_rows].reset_index(drop=True),
                pd.Series(predicted),
                asset.iloc[holdout_rows].reset_index(drop=True),
            ),
        }
        print(f"Holdout {name}: {holdout_metrics[name]['overall']}")
        if name == best_model_name:
            final_pipelines[name] = pipeline
        else:
            del pipeline

    best_pipeline = final_pipelines[best_model_name]
    joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib", compress=3)
    save_residual_diagnostics(
        y.iloc[holdout_rows].to_numpy(),
        best_pipeline.predict(x.iloc[holdout_rows]),
        best_model_name,
    )
    save_shap_diagnostics(
        best_pipeline,
        x.iloc[final_train_rows],
        x.iloc[holdout_rows],
    )

    leaderboard = []
    zero_oof_target = oof_predictions.loc[
        oof_predictions["model"] == MODEL_NAMES[0], "target"
    ].to_numpy(dtype=float)
    zero_oof_metrics = zero_return_baseline(zero_oof_target)
    zero_holdout_metrics = zero_return_baseline(y.iloc[holdout_rows].to_numpy())
    for name in MODEL_NAMES:
        leaderboard.append(
            {
                "model": name,
                "tuning_validation_mae": tuning_scores[name],
                "best_parameters": tuned_parameters[name],
                "oof": oof_metrics[name]["overall"],
                "oof_mae_skill_vs_zero": float(
                    1
                    - oof_metrics[name]["overall"]["mae"]
                    / zero_oof_metrics["mae"]
                ),
                "oof_by_asset": oof_metrics[name]["by_asset"],
                "holdout": holdout_metrics[name]["overall"],
                "holdout_mae_skill_vs_zero": float(
                    1
                    - holdout_metrics[name]["overall"]["mae"]
                    / zero_holdout_metrics["mae"]
                ),
                "holdout_by_asset": holdout_metrics[name]["by_asset"],
            }
        )
    start_utc = panel["dt"].min().isoformat()
    end_utc = panel["dt"].max().isoformat()
    result = {
        "experiment": {
            "target": TARGET_NAME,
            "target_definition": "log(close[t+1] / close[t])",
            "feature_availability": "bar t and earlier only",
            "feature_names": FEATURES,
            "models": MODEL_NAMES,
            "selected_model_by_oof_mae": best_model_name,
            "optuna_trials_per_model": trials,
            "optuna_training_timestamp_stride": tuning_stride,
            "outer_time_series_splits": n_splits,
            "time_gap_bars": 1,
            "holdout_fraction_unique_timestamps": 0.10,
            "tuning_fraction_of_development": {
                "train_end": 0.60,
                "validation_end": 0.70,
                "timestamp_stride": tuning_stride,
            },
            "oof_fraction_of_development": float(
                (folds[-1][1][-1] - folds[0][1][0] + 1) / len(dev_times)
            ),
            "primary_metric": "MAE",
            "secondary_metrics": ["RMSE", "Spearman rank IC", "directional accuracy"],
            "seed": RANDOM_STATE,
            "cpu_threads_per_boosting_model": CPU_THREADS,
            "baseline_ridge_tuning_validation": baseline_tuning_metrics,
            "zero_return_baseline": {
                "oof": zero_oof_metrics,
                "holdout": zero_holdout_metrics,
            },
            "inference_caveat": (
                "Results are predictive metrics only; temporal dependence, "
                "transaction costs, slippage, and execution are not modeled."
            ),
        },
        "data": {
            "rows": int(len(panel)),
            "unique_timestamps": int(len(times)),
            "start_utc": start_utc,
            "end_utc": end_utc,
            "development_rows": int(len(final_train_rows)),
            "holdout_rows": int(len(holdout_rows)),
            "oof_rows_per_model": int(
                oof_predictions.loc[
                    oof_predictions["model"] == MODEL_NAMES[0]
                ].shape[0]
            ),
        },
        "leaderboard": leaderboard,
    }
    metrics_path = REPORTS_DIR / "metrics_summary.json"
    metrics_path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model": row["model"],
                "tuning_validation_mae": row["tuning_validation_mae"],
                "oof_mae": row["oof"]["mae"],
                "oof_rmse": row["oof"]["rmse"],
                "oof_spearman_ic": row["oof"]["spearman_ic"],
                "oof_mae_skill_vs_zero": row["oof_mae_skill_vs_zero"],
                "holdout_mae": row["holdout"]["mae"],
                "holdout_mae_skill_vs_zero": row["holdout_mae_skill_vs_zero"],
                "holdout_rmse": row["holdout"]["rmse"],
                "holdout_spearman_ic": row["holdout"]["spearman_ic"],
            }
            for row in leaderboard
        ]
    ).to_csv(REPORTS_DIR / "model_leaderboard.csv", index=False)
    write_readme(result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--splits", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.trials < 1:
        parser.error("--trials must be positive.")
    if arguments.splits < 2:
        parser.error("--splits must be at least 2.")
    run_experiment(trials=arguments.trials, n_splits=arguments.splits)
