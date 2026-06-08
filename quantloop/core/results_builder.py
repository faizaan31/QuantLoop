"""Assemble BacktestMetrics from a completed portfolio and market data."""

from typing import Any

import polars as pl

from quantloop.core.portfolio import Portfolio
from quantloop.metrics import calculate_metrics, liquidity_metrics, trade_level_metrics
from quantloop.results import BacktestMetrics, _backtest_metrics_from_dict


def compute_usd_metrics(
    equity_df: pl.DataFrame,
    exchange_rate: pl.DataFrame,
    metrics: dict[str, Any],
) -> None:
    """Compute USD-denominated metrics by converting the equity curve.

    Joins exchange rate data to the equity curve via forward-fill asof join,
    then computes key metrics on the USD-converted equity.

    Args:
        equity_df: DataFrame with ``timestamp`` and ``equity`` columns.
        exchange_rate: DataFrame with ``timestamp`` and ``rate`` columns.
        metrics: Metrics dict to update with USD fields.
    """
    rate_df = exchange_rate.sort("timestamp").select(
        pl.col("timestamp").alias("_rate_ts"),
        pl.col("rate"),
    )

    usd_df = equity_df.sort("timestamp").join_asof(
        rate_df,
        left_on="timestamp",
        right_on="_rate_ts",
        strategy="backward",
    )

    if usd_df["rate"].null_count() == usd_df.height:
        return

    usd_df = usd_df.with_columns(
        (pl.col("equity") * pl.col("rate").forward_fill()).alias("equity_usd"),
    ).drop_nulls("equity_usd")

    if usd_df.height == 0:
        return

    initial_equity_usd = float(usd_df["equity_usd"][0])
    usd_metrics = calculate_metrics(
        usd_df.select(pl.col("timestamp"), pl.col("equity_usd").alias("equity")),
        initial_equity_usd,
    )

    metrics["final_equity_usd"] = float(usd_df["equity_usd"][-1])
    metrics["total_return_usd"] = usd_metrics.get("total_return", 0.0)
    metrics["sharpe_ratio_usd"] = usd_metrics.get("sharpe_ratio", 0.0)
    metrics["max_drawdown_usd"] = usd_metrics.get("max_drawdown", 0.0)


def _compute_buy_hold_return(long_data: pl.DataFrame) -> float:
    """Return buy-and-hold return for the first symbol in long-format data."""
    if "symbol" in long_data.columns and "close" in long_data.columns:
        first_sym = long_data["symbol"][0]
        sym_prices = long_data.filter(pl.col("symbol") == first_sym)["close"].drop_nulls()
        if len(sym_prices) >= 2:
            first_price = float(sym_prices[0])
            last_price = float(sym_prices[-1])
            if first_price != 0:
                return (last_price - first_price) / first_price
    elif "close" in long_data.columns:
        prices = long_data["close"].drop_nulls()
        if len(prices) >= 2:
            first_price = float(prices[0])
            last_price = float(prices[-1])
            if first_price != 0:
                return (last_price - first_price) / first_price
    return 0.0


def build_backtest_metrics(
    portfolio: Portfolio,
    *,
    initial_cash: float,
    long_data: pl.DataFrame,
    data: pl.DataFrame,
    exchange_rate: pl.DataFrame | None = None,
) -> BacktestMetrics:
    """Build BacktestMetrics from a completed portfolio.

    Args:
        portfolio: Portfolio after the simulation has finished.
        initial_cash: Starting capital used for the run.
        long_data: Canonical long-format OHLCV data for buy-and-hold baseline.
        data: Legacy/wide data reference used for liquidity metrics.
        exchange_rate: Optional quote-to-USD rate series.

    Returns:
        BacktestMetrics with all performance data.
    """
    equity_df = pl.DataFrame(
        {
            "timestamp": portfolio.timestamps,
            "equity": portfolio.equity_curve,
        },
        strict=False,
    )

    metrics = calculate_metrics(equity_df, initial_cash)

    metrics["final_equity"] = portfolio.get_value()
    metrics["equity_peak"] = float(equity_df["equity"].max()) if len(equity_df) > 0 else initial_cash  # type: ignore[arg-type]
    metrics["final_positions"] = dict(portfolio.positions)
    metrics["final_cash"] = portfolio.cash

    trades_df = portfolio.get_trades()
    trade_stats = portfolio.get_trade_stats()
    metrics["trades"] = trades_df
    metrics["win_rate"] = trade_stats.win_rate
    metrics["return_annualized"] = metrics.get("cagr", 0.0)
    metrics["buy_hold_return"] = _compute_buy_hold_return(long_data)

    if len(trades_df) > 0:
        pct_col = trades_df["return_pct"]
        bars_col = trades_df["bars_held"]

        metrics["best_trade_pct"] = float(pct_col.max())  # type: ignore[arg-type]
        metrics["worst_trade_pct"] = float(pct_col.min())  # type: ignore[arg-type]
        metrics["avg_trade_pct"] = float(pct_col.mean())  # type: ignore[arg-type]
        metrics["max_trade_duration"] = float(bars_col.max())  # type: ignore[arg-type]
        metrics["avg_trade_duration"] = float(bars_col.mean())  # type: ignore[arg-type]

        tlm = trade_level_metrics(portfolio.trade_tracker.trades)
        metrics["expectancy"] = tlm["expectancy"]
        metrics["sqn"] = tlm["sqn"]
        metrics["kelly_criterion"] = tlm["kelly_criterion"]
    else:
        metrics["best_trade_pct"] = 0.0
        metrics["worst_trade_pct"] = 0.0
        metrics["avg_trade_pct"] = 0.0
        metrics["max_trade_duration"] = 0.0
        metrics["avg_trade_duration"] = 0.0
        metrics["expectancy"] = 0.0
        metrics["sqn"] = 0.0
        metrics["kelly_criterion"] = 0.0

    liq = liquidity_metrics(trades_df, data)
    metrics.update({k: v for k, v in liq.items() if v is not None})

    if exchange_rate is not None and len(equity_df) > 0:
        compute_usd_metrics(equity_df, exchange_rate, metrics)

    return _backtest_metrics_from_dict(metrics, trade_stats)
