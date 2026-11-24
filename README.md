# BTC–ETH 3-Minute Quantitative Research

Leakage-aware research on BTCUSDT and ETHUSDT 3-minute OHLCV bars. The
exploratory target is the next-bar log return, `log(close[t+1] / close[t])`.
This project is research, not investment advice or a claim of profitability.

## Data and scope

- Inputs: `data/processed/BTCUSDT_3m.parquet` and
  `data/processed/ETHUSDT_3m.parquet` (confirmed FINAL; source files are read-only).
- Combined observations: 2,374,262; time range:
  2019-09-08T17:57:00+00:00 through 2026-07-26T17:24:00+00:00.
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
| Ridge | 0.0009192601 | 0.0403% | 0.001453527 | 0.023036 | 0.0008264822 |
| HistGradientBoostingRegressor | 0.0009187647 | 0.0942% | 0.001453951 | 0.033866 | 0.0008259277 |
| LGBMRegressor | 0.000918784 | 0.0921% | 0.001455362 | 0.034278 | 0.0008254718 |
| CatBoostRegressor | 0.0009188555 | 0.0843% | 0.001453797 | 0.032916 | 0.0008261552 |

## Reproducibility

Use Python 3.14 and install dependencies from `requirements.txt`.

```powershell
python src\eda_analysis.py
python src\train_evaluate.py --trials 2
python src\build_notebook.py
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 --inplace notebooks\analysis.ipynb
```

See `reports/eda_summary.json`, `reports/metrics_summary.json`, and
`reports/oof_predictions.parquet` for machine-readable results. Figures are in
`reports/figures/`; the selected fitted pipeline is in `models/`.

## Interpretation limits

Returns are heavy-tailed and temporally dependent. Large-sample correlation
p-values are descriptive; use time-aware validation, cost/slippage modeling,
and regime robustness checks before any trading interpretation. Current data
end at 2026-07-26T17:24:00+00:00.
