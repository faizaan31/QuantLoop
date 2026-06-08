"""Core constants for QuantLoop."""

DEFAULT_ASSET_NAME = "asset"

OHLCV_ALIASES: dict[str, list[str]] = {
    "open": ["Open", "OPEN", "open_price", "Open_Price"],
    "high": ["High", "HIGH", "high_price", "High_Price"],
    "low": ["Low", "LOW", "low_price", "Low_Price"],
    "close": ["Close", "CLOSE", "close_price", "Close_Price", "adj_close", "Adj_Close", "Adj Close"],
    "volume": ["Volume", "VOLUME", "vol", "Vol"],
    "factor": ["Factor", "FACTOR", "adj_factor", "Adj_Factor", "split_factor"],
}
