"""Strategy base classes and parameter descriptors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import polars as pl

from quantloop.core.context import BacktestContext

if TYPE_CHECKING:
    from quantloop.core.portfolio import Portfolio


class Param:
    """Descriptor that reads/writes strategy parameters through ``self.params``.

    Use the :func:`param` factory to create instances on a Strategy subclass::

        class MyStrategy(Strategy):
            fast = param(10)
            slow = param(30)

    When accessed on an instance, the descriptor returns
    ``self.params.get(name, default)``.  When set, it writes into
    ``self.params[name]``.
    """

    def __init__(self, default: Any = None) -> None:
        self.default = default
        self.name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return obj.params.get(self.name, self.default)

    def __set__(self, obj: Any, value: Any) -> None:
        obj.params[self.name] = value


def param(default: Any = None) -> Any:
    """Declare a strategy parameter with an optional default value.

    Usage::

        class MyStrategy(Strategy):
            fast_period = param(10)
            slow_period = param(30)

            def preprocess(self, df):
                return df.with_columns(
                    ind.sma("close", self.fast_period).alias("sma_fast"),
                    ind.sma("close", self.slow_period).alias("sma_slow"),
                )

    Parameters declared with ``param()`` are automatically populated from
    keyword arguments passed to ``Strategy.__init__()`` (or via ``backtest(params=...)``).
    No ``__init__`` override is needed.

    Args:
        default: Default value when the parameter is not provided.
    """
    return Param(default)


class Strategy(ABC):
    """
    Base class for trading strategies.

    Subclasses should implement:
    - preprocess(): Vectorized feature engineering using Polars
    - next(): Event-driven logic called on each bar

    Strategy parameters can be declared as class attributes using :func:`param`::

        class MyStrategy(Strategy):
            fast = param(10)
            slow = param(30)

    These are automatically populated from keyword arguments and accessible
    as ``self.fast`` / ``self.slow``.  The traditional ``self.params.get()``
    pattern is also supported.
    """

    def __init__(self, **params: Any) -> None:
        """
        Initialize strategy with parameters.

        Args:
            **params: Strategy parameters (e.g., sma_period=20)
        """
        self.params = params

    @abstractmethod
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Preprocess data using vectorized Polars operations.

        This method is called once before the backtest loop starts.
        Use it to calculate indicators, features, or ML predictions.

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns

        Example:
            def preprocess(self, df):
                from quantloop import indicators as ind
                return df.with_columns([
                    ind.sma("close", 20).alias("sma_20"),
                    ind.rsi("close", 14).alias("rsi_14")
                ])
        """
        pass

    @abstractmethod
    def next(self, ctx: BacktestContext) -> None:
        """
        Execute strategy logic for the current bar.

        This method is called on every bar during the backtest.
        Use ctx.portfolio to place orders.

        Args:
            ctx: BacktestContext containing current bar data and portfolio

        Example:
            def next(self, ctx):
                if ctx.row["rsi_14"] < 30:
                    ctx.portfolio.order_target_percent("BTC", 0.5)
                elif ctx.row["rsi_14"] > 70:
                    ctx.portfolio.close_position("BTC")
        """
        pass

    def on_start(self, portfolio: Portfolio) -> None:  # noqa: B027
        """
        Called once before the backtest starts.

        Override this to perform any initialization logic.

        Args:
            portfolio: The Portfolio instance
        """
        pass

    def on_finish(self, portfolio: Portfolio) -> None:  # noqa: B027
        """
        Called once after the backtest completes.

        Override this to perform cleanup or final analysis.

        Args:
            portfolio: The Portfolio instance
        """
        pass


class WeightStrategy(Strategy):
    """Strategy that expresses positions as target weights per symbol.

    Subclasses implement ``get_weights()`` instead of ``next()``.
    On each bar the returned weights are passed to ``Portfolio.rebalance()``,
    which uses the unified order execution path (commissions, slippage,
    stop-loss/take-profit, leverage all apply).

    Example::

        class EqualWeight(WeightStrategy):
            def preprocess(self, df):
                return df

            def get_weights(self, ctx):
                n = len(ctx.symbols)
                return {sym: 1.0 / n for sym in ctx.symbols}
    """

    @abstractmethod
    def get_weights(self, ctx: BacktestContext) -> dict[str, float]:
        """Return target portfolio weights for each symbol.

        Args:
            ctx: Current bar context.

        Returns:
            Mapping of symbol to target weight (fraction of portfolio value).
        """
        ...

    def next(self, ctx: BacktestContext) -> None:
        """Execute rebalance based on ``get_weights()``."""
        weights = self.get_weights(ctx)
        ctx.portfolio.rebalance(weights)
