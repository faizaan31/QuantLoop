"""Backtesting engine that executes strategy simulations."""

import gc
from typing import Any

import polars as pl

from quantloop.commissions import CommissionModel
from quantloop.core.constants import DEFAULT_ASSET_NAME
from quantloop.core.context import BacktestContext, RowAccessor
from quantloop.core.data_utils import merge_asset_dataframes, standardize_dataframe
from quantloop.core.portfolio import Portfolio
from quantloop.core.results_builder import build_backtest_metrics
from quantloop.core.strategy import Strategy
from quantloop.results import BacktestMetrics
from quantloop.slippage import SlippageModel
from quantloop.universe import UniverseContext, UniverseProvider


class Engine:
    """
    Backtesting engine that executes the strategy simulation.

    The engine:
    1. Preprocesses data using the strategy
    2. Iterates through each bar
    3. Updates portfolio prices
    4. Calls strategy.next()
    5. Records metrics
    """

    def __init__(
        self,
        strategy: Strategy,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        initial_cash: float = 100_000.0,
        commission: float | tuple[float, float] | CommissionModel = 0.0,
        slippage: float | SlippageModel = 0.0,
        price_columns: dict[str, str] | None = None,
        warmup: int | str = "auto",
        order_delay: int = 0,
        borrow_rate: float = 0.0,
        bars_per_day: float | None = None,
        max_position_size: float | None = None,
        max_total_exposure: float | None = None,
        max_drawdown_stop: float | None = None,
        daily_loss_limit: float | None = None,
        leverage: float = 1.0,
        maintenance_margin: float | None = None,
        fractional_shares: bool = True,
        factor_column: str | None = None,
        universe_provider: UniverseProvider | None = None,
        exchange_rate: pl.DataFrame | None = None,
    ):
        """
        Initialize the backtesting engine.

        Args:
            strategy: Strategy instance to backtest
            data: Polars DataFrame with price data OR dict mapping asset names to DataFrames
            initial_cash: Starting cash balance
            commission: Commission specification. Accepts a percentage float, a (fixed, percent) tuple,
                       or a CommissionModel instance
            slippage: Slippage rate as fraction or a SlippageModel instance
            price_columns: Dict mapping asset names to price columns
                          (default: auto-detected for dict input, {"asset": "close"} for single DataFrame)
            warmup: Number of bars to skip before executing strategy, or "auto" to automatically
                   detect when all indicators are ready (default "auto")
            order_delay: Number of bars to delay order execution (default 0, max realism is 1)
            borrow_rate: Annual borrow rate for short positions (e.g., 0.02 = 2% per year)
            bars_per_day: Number of bars in a trading day (used for day order expiry and borrow cost calculation)
            max_position_size: Maximum single position size as fraction of portfolio value (e.g., 0.5 = 50%)
            max_total_exposure: Maximum total exposure as fraction of portfolio value (e.g., 1.5 = 150%)
            max_drawdown_stop: Maximum drawdown before halting trading (e.g., 0.2 = 20%)
            daily_loss_limit: Maximum daily loss before halting trading for the day (e.g., 0.05 = 5%)
            leverage: Maximum leverage multiplier (e.g., 2.0 = 2x leverage). Default 1.0.
            maintenance_margin: Minimum margin ratio before margin call (e.g., 0.25 = 25%). Default None.
            fractional_shares: Whether to allow fractional share quantities (default True).
            factor_column: Optional column name for price adjustment factor. When set,
                          commissions are calculated on raw prices (adjusted_price / factor).
            universe_provider: Optional provider that filters tradeable symbols each bar.
                              When set, ``ctx.symbols`` contains only the filtered subset;
                              ``ctx.available_symbols`` contains all symbols with data.
            exchange_rate: Optional DataFrame with ``(timestamp, rate)`` columns for
                          quote-to-USD conversion. Rate is forward-filled to bar timestamps.
                          When provided, BacktestMetrics includes USD-denominated metrics.
        """
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.borrow_rate = borrow_rate
        self.bars_per_day = bars_per_day
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.max_drawdown_stop = max_drawdown_stop
        self.daily_loss_limit = daily_loss_limit
        self.leverage = leverage
        self.maintenance_margin = maintenance_margin
        self.fractional_shares = fractional_shares
        self.factor_column = factor_column
        self.universe_provider = universe_provider
        self.exchange_rate = exchange_rate

        # Validate warmup parameter
        if isinstance(warmup, str):
            if warmup != "auto":
                raise ValueError(f"warmup must be an integer or 'auto', got '{warmup}'")
        elif not isinstance(warmup, int):
            raise ValueError(f"warmup must be an integer or 'auto', got {type(warmup)}")

        self.warmup = warmup
        self.order_delay = order_delay

        # --- Normalize input to long format ---
        # _long_data: long-format DataFrame with 'symbol' column (internal canonical form)
        # self.data / self.price_columns: preserved for backward-compat _calculate_results()

        if isinstance(data, dict):
            # Form B: dict[str, pl.DataFrame] -> tag each DF and concat vertically
            frames: list[pl.DataFrame] = []
            for asset_name, asset_df in data.items():
                sdf = standardize_dataframe(asset_df)
                sdf = sdf.with_columns(pl.lit(asset_name).alias("symbol"))
                frames.append(sdf)
            self._long_data: pl.DataFrame = pl.concat(frames, how="diagonal_relaxed")
            if "timestamp" in self._long_data.columns:
                self._long_data = self._long_data.sort(["timestamp", "symbol"])

            # Legacy wide-format for _calculate_results buy-hold
            self.data, auto_price_columns = merge_asset_dataframes(data)
            self.price_columns = price_columns if price_columns is not None else auto_price_columns
        else:
            sdf = standardize_dataframe(data)

            if "symbol" in sdf.columns:
                # Form C: already long format
                self._long_data = sdf
                if "timestamp" in self._long_data.columns:
                    self._long_data = self._long_data.sort(["timestamp", "symbol"])
                # Legacy: store as-is, detect first symbol's close for buy-hold
                self.data = sdf
                self.price_columns = price_columns if price_columns is not None else {"_first_": "close"}
            else:
                # Form A: single-asset DataFrame -> add symbol=DEFAULT_ASSET_NAME
                sdf = sdf.with_columns(pl.lit(DEFAULT_ASSET_NAME).alias("symbol"))
                self._long_data = sdf

                self.data = sdf
                if price_columns is None:
                    if "close" in sdf.columns:
                        self.price_columns = {DEFAULT_ASSET_NAME: "close"}
                    else:
                        numeric_cols = [
                            c for c in sdf.columns if sdf[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                        ]
                        if numeric_cols:
                            self.price_columns = {DEFAULT_ASSET_NAME: numeric_cols[0]}
                        else:
                            raise ValueError("No price columns found in data")
                else:
                    self.price_columns = price_columns

        self.portfolio: Portfolio | None = None
        self.results: BacktestMetrics | None = None
        self.processed_data: pl.DataFrame | None = None

    def _calculate_auto_warmup(self, df: pl.DataFrame) -> int:
        """Calculate automatic warmup period by finding the first timestamp where all columns are non-null.

        For long-format data, this checks per-symbol rows and finds the first
        timestamp index where every symbol has all columns non-null.

        Args:
            df: Preprocessed long-format DataFrame with indicators.

        Returns:
            Integer warmup period (number of timestamp groups to skip).
        """
        skip_cols = {"timestamp", "date", "datetime", "time", "dt", "_index", "symbol"}
        cols_to_check = [col for col in df.columns if col not in skip_cols]

        if not cols_to_check:
            return 0

        # Add per-row validity flag
        df_with_valid = df.with_columns(
            pl.all_horizontal([pl.col(col).is_not_null() for col in cols_to_check]).alias("_all_valid")
        )

        # Detect timestamp column
        ts_col = None
        for candidate in ("timestamp", "date", "time", "_index"):
            if candidate in df.columns:
                ts_col = candidate
                break

        if ts_col is None or "symbol" not in df.columns:
            # Fallback: flat row-level check (single symbol or no grouping)
            all_valid = df_with_valid["_all_valid"]
            if not all_valid.any():
                n_unique = df[ts_col].n_unique() if ts_col else len(df)
                return max(0, n_unique - 1)
            return int(all_valid.arg_max())  # type: ignore[arg-type]

        # For multi-symbol: all symbols must be valid on a given timestamp
        ts_valid = df_with_valid.group_by(ts_col, maintain_order=True).agg(pl.col("_all_valid").all().alias("ts_valid"))
        valid_series = ts_valid["ts_valid"]
        if not valid_series.any():
            return max(0, len(ts_valid) - 1)
        return int(valid_series.arg_max())  # type: ignore[arg-type]

    def _clear_portfolio(self) -> None:
        """Eagerly release large lists inside the current portfolio.

        Clears equity curves, timestamps, orders, trades and positions so the
        memory is freed immediately rather than waiting for garbage collection.
        """
        if self.portfolio is not None:
            self.portfolio.equity_curve.clear()
            self.portfolio.timestamps.clear()
            self.portfolio.orders.clear()
            self.portfolio.trade_tracker.trades.clear()
            self.portfolio.trade_tracker.open_positions.clear()
            self.portfolio.positions.clear()

    def cleanup(self) -> None:
        """Release references to large internal objects for memory management.

        Call this after extracting results from a completed backtest to free
        memory occupied by the processed DataFrame, portfolio state, and
        intermediate data. Useful when running multiple sequential backtests
        in the same process to prevent memory exhaustion and segfaults.
        """
        self._clear_portfolio()
        self.portfolio = None
        self.processed_data = None
        self.results = None
        gc.collect()

    def __del__(self) -> None:
        """Release internal objects when the engine is garbage collected.

        Acts as a safety net for callers that do not invoke :meth:`cleanup`
        explicitly (e.g. when creating a new ``Engine`` per backtest inside a
        loop).  Does **not** call ``gc.collect()`` to avoid re-entrancy.
        """
        try:
            self._clear_portfolio()
            self.portfolio = None
            self.processed_data = None
            self.results = None
        except Exception:
            pass

    def run(self) -> BacktestMetrics:
        """Run the backtest simulation.

        Returns:
            BacktestMetrics with all performance metrics and trade data.
        """
        # Eagerly release large objects from any previous run so memory is
        # reclaimed before the new portfolio and processed data are allocated.
        self._clear_portfolio()
        self.portfolio = None
        self.processed_data = None
        self.results = None

        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_cash=self.initial_cash,
            commission=self.commission,
            slippage=self.slippage,
            order_delay=self.order_delay,
            borrow_rate=self.borrow_rate,
            bars_per_day=self.bars_per_day,
            max_position_size=self.max_position_size,
            max_total_exposure=self.max_total_exposure,
            max_drawdown_stop=self.max_drawdown_stop,
            daily_loss_limit=self.daily_loss_limit,
            leverage=self.leverage,
            maintenance_margin=self.maintenance_margin,
            fractional_shares=self.fractional_shares,
            factor_column=self.factor_column,
        )

        # Preprocess data using strategy (long-format)
        processed_data = self.strategy.preprocess(self._long_data)
        self.processed_data = processed_data

        # Ensure we have a timestamp column
        timestamp_col = None
        if "timestamp" in processed_data.columns:
            timestamp_col = "timestamp"
        elif "date" in processed_data.columns:
            timestamp_col = "date"
        elif "time" in processed_data.columns:
            timestamp_col = "time"
        else:
            # Use row index as timestamp — assign per-symbol group rank
            processed_data = processed_data.with_row_index("_index")
            timestamp_col = "_index"

        # Calculate warmup period if set to "auto"
        warmup_periods: int
        warmup_periods = self._calculate_auto_warmup(processed_data) if self.warmup == "auto" else self.warmup  # type: ignore

        # Call strategy initialization
        self.strategy.on_start(self.portfolio)

        # --- Main event loop: iterate by timestamp over long-format data ---
        # Group by timestamp to get all symbols' data per bar
        ts_col_series = processed_data[timestamp_col]
        unique_timestamps = ts_col_series.unique(maintain_order=True).sort()

        # Pre-partition data by timestamp for efficient lookup
        grouped = processed_data.partition_by(timestamp_col, as_dict=True, maintain_order=True)

        # Token lifecycle tracking
        first_seen_bar: dict[str, int] = {}
        bar_count: dict[str, int] = {}

        for idx, ts_value in enumerate(unique_timestamps):
            group_key = ts_value
            group_df = grouped.get(group_key)
            if group_df is None:
                # partition_by with single column returns scalar keys
                # Try tuple key as fallback
                group_df = grouped.get((group_key,))
            if group_df is None:
                continue

            current_prices: dict[str, float] = {}
            ohlc_data: dict[str, dict[str, float]] = {}
            bar_data: dict[str, dict[str, Any]] = {}

            for row_dict in group_df.iter_rows(named=True):
                sym = row_dict.get("symbol", DEFAULT_ASSET_NAME)
                close_val = row_dict.get("close")
                close_price = float(close_val) if close_val is not None else 0.0
                current_prices[sym] = close_price

                ohlc_data[sym] = {
                    "open": float(row_dict["open"]) if row_dict.get("open") is not None else close_price,
                    "high": float(row_dict["high"]) if row_dict.get("high") is not None else close_price,
                    "low": float(row_dict["low"]) if row_dict.get("low") is not None else close_price,
                    "close": close_price,
                }
                bar_data[sym] = row_dict

            current_timestamp = ts_value

            # Update token lifecycle tracking
            for sym in bar_data:
                if sym not in first_seen_bar:
                    first_seen_bar[sym] = idx
                bar_count[sym] = bar_count.get(sym, 0) + 1

            # Update factor data for commission calculation on raw prices
            if self.factor_column is not None:
                for sym, row_dict in bar_data.items():
                    factor_val = row_dict.get(self.factor_column)
                    if factor_val is not None:
                        self.portfolio._factors[sym] = float(factor_val)

            # Update portfolio with ALL current prices (including filtered-out symbols)
            self.portfolio._current_bar_data = bar_data
            self.portfolio.update_prices(current_prices, idx, ohlc_data, current_timestamp)

            # Determine tradeable universe
            available_symbols = list(bar_data.keys())
            if self.universe_provider is not None:
                universe_ctx = UniverseContext(
                    timestamp=current_timestamp,
                    bar_index=idx,
                    available_symbols=available_symbols,
                    bar_data=bar_data,
                    first_seen_bar=first_seen_bar,
                    bar_count=bar_count,
                )
                symbols_list = self.universe_provider.get_universe(universe_ctx)
            else:
                symbols_list = available_symbols

            # Create context for strategy
            row_accessor = RowAccessor(bar_data, symbols_list)
            ctx = BacktestContext(
                timestamp=current_timestamp,
                bar_index=idx,
                portfolio=self.portfolio,
                symbols=symbols_list,
                data=bar_data,
                row=row_accessor,
                first_seen_bar=first_seen_bar,
                bar_count=bar_count,
                available_symbols=available_symbols,
            )

            # Call strategy logic (skip warmup period)
            if idx >= warmup_periods:
                self.strategy.next(ctx)

                # Record equity only after warmup to avoid diluting metrics
                self.portfolio.record_equity(ctx.timestamp)

        # Call strategy finalization
        self.strategy.on_finish(self.portfolio)

        # Calculate and return results
        self.results = self._calculate_results()
        return self.results

    def _calculate_results(self) -> BacktestMetrics:
        """Calculate backtest metrics.

        Returns:
            BacktestMetrics with all performance data.
        """
        if not self.portfolio:
            return BacktestMetrics()

        return build_backtest_metrics(
            self.portfolio,
            initial_cash=self.initial_cash,
            long_data=self._long_data,
            data=self.data,
            exchange_rate=self.exchange_rate,
        )
