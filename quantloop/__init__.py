"""
QuantLoop — quantitative backtesting with Polars vectorization and event-driven execution.

Features:
- Polars-based for high performance
- Event-driven architecture with vectorized preprocessing
- Support for fractional and multi-asset positions
- ML model integration ready
- Parallel execution for evolutionary optimization
"""

from quantloop import analysis, data, indicators, indicators_defi, metrics
from quantloop.analysis import (
    LookAheadResult,
    MonteCarloResult,
    PermutationTestResult,
    compute_next_actions,
    detect_look_ahead_bias,
    monte_carlo,
    permutation_test,
)
from quantloop.commissions import (
    CommissionModel,
    CustomCommission,
    FixedPlusPercentCommission,
    MakerTakerCommission,
    PercentCommission,
    TieredCommission,
)
from quantloop.core import (
    BacktestContext,
    Engine,
    Portfolio,
    Strategy,
    WeightStrategy,
    merge_asset_dataframes,
    param,
    standardize_dataframe,
)
from quantloop.orders import Order, OrderStatus, OrderType
from quantloop.results import BacktestMetrics, OptimizeResult, TradeStats
from quantloop.runner import (
    backtest,
    backtest_batch,
    optimize,
    optimize_bayesian,
    optimize_multi,
    walk_forward_analysis,
)
from quantloop.sizers import (
    FixedRiskSizer,
    FixedSizer,
    KellySizer,
    MaxPositionSizer,
    PercentSizer,
    Sizer,
    VolatilitySizer,
)
from quantloop.slippage import AMMSlippage, FlatSlippage, SlippageModel
from quantloop.trades import Trade, TradeTracker
from quantloop.universe import (
    AgeFilter,
    AllSymbols,
    CompositeFilter,
    TopN,
    UniverseContext,
    UniverseProvider,
    VolumeFilter,
)
from quantloop.weight_backtest import WeightBacktestResult, backtest_weights

try:
    from quantloop import plotting
except ImportError:
    plotting = None  # type: ignore[assignment]

__version__ = "0.1.0"
__all__ = [
    "Strategy",
    "WeightStrategy",
    "param",
    "Portfolio",
    "Engine",
    "BacktestContext",
    "backtest",
    "backtest_batch",
    "backtest_weights",
    "optimize",
    "optimize_multi",
    "optimize_bayesian",
    "walk_forward_analysis",
    "standardize_dataframe",
    "merge_asset_dataframes",
    "indicators",
    "metrics",
    "Order",
    "OrderType",
    "OrderStatus",
    "Trade",
    "TradeTracker",
    "WeightBacktestResult",
    "Sizer",
    "FixedSizer",
    "PercentSizer",
    "FixedRiskSizer",
    "KellySizer",
    "VolatilitySizer",
    "MaxPositionSizer",
    "CommissionModel",
    "PercentCommission",
    "FixedPlusPercentCommission",
    "MakerTakerCommission",
    "TieredCommission",
    "CustomCommission",
    "analysis",
    "data",
    "compute_next_actions",
    "monte_carlo",
    "detect_look_ahead_bias",
    "permutation_test",
    "MonteCarloResult",
    "LookAheadResult",
    "PermutationTestResult",
    "BacktestMetrics",
    "OptimizeResult",
    "TradeStats",
    "plotting",
    "UniverseProvider",
    "UniverseContext",
    "AllSymbols",
    "AgeFilter",
    "VolumeFilter",
    "TopN",
    "CompositeFilter",
    "SlippageModel",
    "FlatSlippage",
    "AMMSlippage",
    "indicators_defi",
]
