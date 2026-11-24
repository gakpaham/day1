"""Project paths and shared modeling configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

DATA_FILES = {
    "BTCUSDT": DATA_DIR / "BTCUSDT_3m.parquet",
    "ETHUSDT": DATA_DIR / "ETHUSDT_3m.parquet",
}

RANDOM_STATE = 42
BAR_INTERVAL_MINUTES = 3
TARGET_NAME = "future_log_return_3m"
FEATURES = [
    "current_log_return",
    "bar_return",
    "return_lag_1",
    "return_lag_3",
    "return_lag_5",
    "realized_volatility_20",
    "range_log",
    "log_volume",
    "log_trades",
    "buy_ratio",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "asset_code",
]
