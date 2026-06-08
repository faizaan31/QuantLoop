"""
Core backtesting engine and components.

This package provides the fundamental building blocks for backtesting:
- Portfolio: Manages positions and cash
- Strategy: Base class for defining trading strategies
- Engine: Executes the backtest simulation
- BacktestContext: Data container passed to strategy.next()
"""

from quantloop.core.constants import DEFAULT_ASSET_NAME
from quantloop.core.context import BacktestContext, RowAccessor
from quantloop.core.data_utils import extract_date, merge_asset_dataframes, standardize_dataframe
from quantloop.core.engine import Engine
from quantloop.core.portfolio import Portfolio
from quantloop.core.strategy import Param, Strategy, WeightStrategy, param

# Backward-compatible aliases (used in tests and docs)
_RowAccessor = RowAccessor

__all__ = [
    "DEFAULT_ASSET_NAME",
    "BacktestContext",
    "RowAccessor",
    "_RowAccessor",
    "extract_date",
    "merge_asset_dataframes",
    "standardize_dataframe",
    "Engine",
    "Portfolio",
    "Param",
    "Strategy",
    "WeightStrategy",
    "param",
]
