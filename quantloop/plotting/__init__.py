"""
Visualization module for QuantLoop backtesting results.

Requires plotly as an optional dependency: pip install plotly
"""

from quantloop.plotting.charts import (
    plot_backtest,
    plot_param_heatmap,
    plot_permutation_test,
    plot_returns_distribution,
    plot_sensitivity,
)

__all__ = [
    "plot_backtest",
    "plot_param_heatmap",
    "plot_permutation_test",
    "plot_returns_distribution",
    "plot_sensitivity",
]
