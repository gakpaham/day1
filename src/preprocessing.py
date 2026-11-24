"""Leakage-aware feature creation for next-bar return research."""

import numpy as np
import pandas as pd

from config import BAR_INTERVAL_MINUTES, FEATURES, TARGET_NAME


def make_model_frame(
    symbol: str,
    frame: pd.DataFrame,
    asset_codes: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Create predictors available at bar t and the next-bar return target."""
    asset_codes = asset_codes or {"BTCUSDT": 0, "ETHUSDT": 1}
    if symbol not in asset_codes:
        raise ValueError(f"No numeric asset code configured for {symbol!r}.")

    result = frame.sort_values("dt").reset_index(drop=True).copy()
    close = result["close"].astype("float64")
    open_price = result["open"].astype("float64")
    high = result["high"].astype("float64")
    low = result["low"].astype("float64")

    result["bar_return"] = np.log(close / close.shift(1))
    result["current_log_return"] = np.log(close / open_price)
    result["return_lag_1"] = result["bar_return"].shift(1)
    result["return_lag_3"] = result["bar_return"].shift(3)
    result["return_lag_5"] = result["bar_return"].shift(5)
    result["realized_volatility_20"] = (
        result["bar_return"].rolling(window=20, min_periods=5).std()
    )
    result["range_log"] = np.log(high / low)
    result["log_volume"] = np.log1p(result["volume"].astype("float64"))
    result["log_trades"] = np.log1p(result["trades"].astype("float64"))
    result["hour_sin"] = np.sin(2 * np.pi * result["dt"].dt.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["dt"].dt.hour / 24)
    weekday = result["dt"].dt.day_of_week
    result["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    result["asset_code"] = asset_codes[symbol]

    next_close = close.shift(-1)
    next_time = result["dt"].shift(-1)
    contiguous = next_time.sub(result["dt"]).eq(
        pd.Timedelta(minutes=BAR_INTERVAL_MINUTES)
    )
    result[TARGET_NAME] = np.log(next_close / close).where(contiguous)
    result["asset"] = symbol

    selected = result[["dt", "asset", *FEATURES, TARGET_NAME]].replace(
        [np.inf, -np.inf], np.nan
    )
    return selected


def build_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a timestamp-sorted panel while keeping asset labels explicit."""
    panel = pd.concat(
        [make_model_frame(symbol, frame) for symbol, frame in frames.items()],
        ignore_index=True,
    )
    return panel.sort_values(["dt", "asset"]).reset_index(drop=True)
