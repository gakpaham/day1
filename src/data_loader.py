"""Read-only loading and basic validation for the confirmed final datasets."""

from pathlib import Path

import pandas as pd

from config import DATA_FILES

REQUIRED_COLUMNS = {
    "dt",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",
    "taker_buy_base",
    "buy_ratio",
}


def load_market_data(path: Path, expected_symbol: str) -> pd.DataFrame:
    """Load one final Parquet series and validate its core columns and order."""
    frame = pd.read_parquet(path)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing_columns)}")

    frame["dt"] = pd.to_datetime(frame["dt"], utc=True)
    frame = frame.sort_values("dt").reset_index(drop=True)
    if frame["dt"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate timestamps.")
    if not frame["dt"].is_monotonic_increasing:
        raise ValueError(f"{path.name} timestamps are not increasing.")
    symbols = frame["symbol"].dropna().unique().tolist()
    if symbols != [expected_symbol]:
        raise ValueError(
            f"{path.name} expected symbol {expected_symbol!r}, found {symbols!r}."
        )
    return frame


def load_all_market_data() -> dict[str, pd.DataFrame]:
    """Load both confirmed assets without modifying the source files."""
    return {
        symbol: load_market_data(path, symbol)
        for symbol, path in DATA_FILES.items()
    }
