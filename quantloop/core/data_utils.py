"""DataFrame standardization and multi-asset merge utilities."""

from datetime import date, datetime
from typing import Any

import polars as pl

from quantloop.core.constants import OHLCV_ALIASES


def extract_date(timestamp: Any) -> date | None:
    """
    Extract date from various timestamp types.

    Supported formats:
    - datetime objects (datetime.datetime or datetime.date)
    - Unix timestamps (int or float, seconds or milliseconds)
    - String formats: "yyyy-mm-dd hh:mm:ss", "yyyy-mm-dd", ISO format with 'T'

    Args:
        timestamp: Timestamp in various formats

    Returns:
        date object or None if extraction fails
    """
    if timestamp is None:
        return None

    # Python datetime
    if isinstance(timestamp, datetime):
        return timestamp.date()

    # Already a date
    if isinstance(timestamp, date):
        return timestamp

    # Unix timestamp (int or float)
    if isinstance(timestamp, (int, float)):
        # Handle both seconds and milliseconds
        ts_value = timestamp
        if ts_value > 1e10:  # Likely milliseconds
            ts_value = ts_value / 1000
        try:
            return datetime.fromtimestamp(ts_value).date()
        except (ValueError, OSError):
            return None

    # String parsing - try common formats
    if isinstance(timestamp, str):
        # Remove common separators and extract date part
        # "yyyy-mm-dd hh:mm:ss" -> "yyyy-mm-dd"
        # "yyyy-mm-ddThh:mm:ss" -> "yyyy-mm-dd"
        # "yyyy-mm-dd" -> "yyyy-mm-dd"
        try:
            # Split by space or 'T' to get date part
            if " " in timestamp:
                date_part = timestamp.split(" ")[0]
            elif "T" in timestamp:
                date_part = timestamp.split("T")[0]
            else:
                date_part = timestamp

            # Parse yyyy-mm-dd
            parts = date_part.split("-")
            if len(parts) == 3:
                return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            pass

    # Try converting to string and parsing (for other types like Polars datetime)
    try:
        dt_str = str(timestamp)
        if " " in dt_str:
            date_part = dt_str.split(" ")[0]
        elif "T" in dt_str:
            date_part = dt_str.split("T")[0]
        else:
            date_part = dt_str

        parts = date_part.split("-")
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, AttributeError, IndexError):
        pass

    return None


def standardize_dataframe(
    df: pl.DataFrame,
    timestamp_col: str | None = None,
    auto_detect: bool = True,
) -> pl.DataFrame:
    """Standardize a DataFrame by renaming common timestamp and OHLCV column names.

    Detects common column name variants (e.g. ``"Date"`` -> ``"timestamp"``,
    ``"Open"`` -> ``"open"``) and renames them to the canonical lowercase names
    expected by the engine.

    Args:
        df: Input DataFrame.
        timestamp_col: Specific timestamp column to rename (if None, auto-detect).
        auto_detect: Auto-detect common column names (default True).

    Returns:
        DataFrame with standardized column names.

    Example:
        # Auto-detect and rename
        df = standardize_dataframe(df)

        # Specify timestamp column explicitly
        df = standardize_dataframe(df, timestamp_col="datetime")
    """
    renames: dict[str, str] = {}

    # --- Timestamp ---
    if "timestamp" not in df.columns:
        if timestamp_col and timestamp_col in df.columns:
            renames[timestamp_col] = "timestamp"
        elif auto_detect:
            common_names = ["date", "datetime", "time", "dt", "Date", "DateTime", "Time"]
            for name in common_names:
                if name in df.columns:
                    renames[name] = "timestamp"
                    break

    # --- OHLCV ---
    if auto_detect:
        for canonical, aliases in OHLCV_ALIASES.items():
            if canonical in df.columns:
                continue
            for alias in aliases:
                if alias in df.columns and alias not in renames:
                    renames[alias] = canonical
                    break

    if renames:
        df = df.rename(renames)

    return df


def merge_asset_dataframes(
    data_dict: dict[str, pl.DataFrame],
    price_column: str = "close",
) -> tuple[pl.DataFrame, dict[str, str]]:
    """
    Merge multiple asset dataframes into a single wide-format dataframe.

    Args:
        data_dict: Dictionary mapping asset names to their dataframes
        price_column: Name of the price column in each dataframe (default "close")

    Returns:
        Tuple of (merged_dataframe, price_columns_mapping)

    Example:
        btc_df = pl.DataFrame({"timestamp": [...], "close": [...]})
        eth_df = pl.DataFrame({"timestamp": [...], "close": [...]})

        merged_df, price_cols = merge_asset_dataframes({
            "BTC": btc_df,
            "ETH": eth_df
        })
        # merged_df has columns: timestamp, BTC_close, ETH_close
        # price_cols = {"BTC": "BTC_close", "ETH": "ETH_close"}
    """
    if not data_dict:
        raise ValueError("data_dict cannot be empty")

    # Standardize all dataframes
    standardized = {asset: standardize_dataframe(df) for asset, df in data_dict.items()}

    # Start with the first dataframe's timestamp
    first_asset = list(standardized.keys())[0]
    merged = (
        standardized[first_asset].select(["timestamp"]) if "timestamp" in standardized[first_asset].columns else None
    )

    # Build price columns mapping
    price_columns = {}

    # Merge all dataframes
    for asset, df in standardized.items():
        # Rename price column to include asset name
        new_col_name = f"{asset}_{price_column}"
        price_columns[asset] = new_col_name

        # Select timestamp and price columns
        if "timestamp" in df.columns:
            df_subset = df.select(["timestamp", price_column]).rename({price_column: new_col_name})

            merged = df_subset if merged is None else merged.join(df_subset, on="timestamp", how="full", coalesce=True)
        else:
            # No timestamp - add index and merge
            df_subset = df.select([price_column]).rename({price_column: new_col_name})
            merged = df_subset if merged is None else pl.concat([merged, df_subset], how="horizontal")

    if merged is not None and "timestamp" in merged.columns:
        merged = merged.sort("timestamp")

    if merged is None:
        raise ValueError("Failed to merge dataframes")

    return merged, price_columns
