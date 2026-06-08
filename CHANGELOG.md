# Changelog

All notable changes to QuantLoop are documented here.

## [Unreleased]

### Changed
- Documentation and install flow updated for clone-and-install from GitHub
- Added dev dependencies (`yfinance`, `pyarrow`) for examples and tests

## [0.1.0] - 2026-06-08

### Added
- QuantLoop backtesting engine with Polars vectorized preprocessing and event-driven execution
- `Strategy` / `WeightStrategy` API with `preprocess()` and `next()` lifecycle hooks
- `Portfolio` with market, limit, stop, stop-limit, and bracket orders
- Risk management: stop-loss, take-profit, trailing stops, margin, leverage, position limits
- 25+ technical indicators, DeFi indicators, and optional TA-Lib integration
- Weight-based backtesting via `backtest_weights()` and `WeightStrategy`
- Parallel optimization: grid search, Pareto, Bayesian, walk-forward analysis
- Advanced analysis: Monte Carlo, permutation testing, look-ahead bias detection
- Multi-asset long-format data, dynamic universe filtering, AMM slippage models
- Trade data pipeline for DEX/AMM trade aggregation into OHLCV bars
- Optional Plotly visualization
