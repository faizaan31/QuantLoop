# QuantLoop

**QuantLoop** is a quantitative backtesting library for strategy research and optimization. It combines vectorized [Polars](https://pola.rs/) preprocessing with an event-driven execution loop for flexible, realistic strategy logic.

## Features

- **Hybrid architecture** — vectorized preprocessing (Polars) + event-driven execution loop
- **25+ built-in indicators** — SMA, EMA, RSI, MACD, Bollinger Bands, ATR, SuperTrend, ADX, and more
- **Complete order system** — market, limit, stop, stop-limit, bracket orders, day/GTC orders
- **Risk management** — stop-loss, take-profit, trailing stops, position size limits, drawdown stops
- **Short selling** — negative positions, borrow costs, position reversals
- **Margin & leverage** — configurable leverage, margin tracking, margin calls
- **Commission models** — percentage, fixed, maker/taker, volume-tiered, custom
- **Position sizing** — fixed, percent, fixed-risk, Kelly, volatility-based
- **Weight-based backtesting** — declarative portfolio allocation with `backtest_weights()` or `WeightStrategy`, rebalance scheduling, stop-loss/take-profit, next-actions output
- **Multi-asset** — pass a dict of DataFrames or a long-format DataFrame with `symbol` column; all OHLC data preserved
- **Dynamic universe** — `UniverseProvider` protocol filters tradeable symbols per bar; built-in `AgeFilter`, `VolumeFilter`, `TopN`, `CompositeFilter`; token lifecycle tracking via `ctx.first_seen_bar` / `ctx.bar_count`
- **Parallel optimization** — grid search, multi-objective Pareto, Bayesian optimization
- **Walk-forward analysis** — rolling and anchored train/test splits
- **Advanced analysis** — Monte Carlo simulation, look-ahead bias detection, permutation testing
- **Visualization** — interactive Plotly charts (price, equity, drawdown, trade markers, heatmaps)
- **AMM-aware execution** — pluggable `SlippageModel` with `FlatSlippage` and `AMMSlippage` (constant-product formula)
- **Exchange rate support** — `Engine(exchange_rate=...)` converts equity curve to USD; reports dual quote/USD metrics
- **DeFi indicators** — `buy_sell_ratio`, `net_flow`, `trade_intensity`, `pump_detector`, `rug_pull_detector`, AMM `price_impact_estimate`, `liquidity_ratio`, and more
- **Trade data pipeline** — validate and aggregate raw DEX/AMM trades into OHLCV bars (time-based or trade-count), with buy/sell volume split, VWAP, and optional USD conversion
- **Data utilities** — validation, cleaning, OHLCV resampling
- **Optional TA-Lib integration** — wrap any TA-Lib function into Polars expressions

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/faizaan31/QuantLoop.git
cd QuantLoop
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras (install alongside dev):

```bash
pip install -e ".[plotting]"   # Plotly charts
pip install -e ".[talib]"       # TA-Lib integration (requires system TA-Lib library)
```

## Development

Run the test suite:

```bash
pytest tests/ -q --ignore=tests/test_talib_real.py
```

Lint and type-check:

```bash
ruff check quantloop tests
ruff format quantloop tests
mypy quantloop
```

Run an example:

```bash
python examples/example.py
```

The Quick Start below uses Yahoo Finance data — install it if needed:

```bash
pip install yfinance
```

## Quick Start

```python
import polars as pl
import yfinance as yf
from quantloop import Engine, Strategy
from quantloop import indicators as ind
from quantloop.core import BacktestContext
from quantloop.plotting import plot_backtest


class SMACross(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            ind.sma("close", 10).alias("sma_fast"),
            ind.sma("close", 30).alias("sma_slow"),
        ).with_columns(
            ind.crossover("sma_fast", "sma_slow").alias("buy"),
            ind.crossunder("sma_fast", "sma_slow").alias("sell"),
        )

    def next(self, ctx: BacktestContext) -> None:
        if ctx.row.get("buy"):
            ctx.portfolio.order_target_percent("asset", 1.0)
        elif ctx.row.get("sell"):
            ctx.portfolio.close_position("asset")


# Download data from Yahoo Finance (pip install yfinance)
ticker = yf.download("AAPL", start="2016-01-01", end="2026-01-01", auto_adjust=True)
ticker = ticker.droplevel("Ticker", axis=1).reset_index()
data = pl.from_pandas(ticker)

# Run backtest
engine = Engine(SMACross(), data, commission=.005, initial_cash=100_000)
results = engine.run()

print(results)

# Interactive chart saved to HTML (requires pip install -e ".[plotting]")
fig = plot_backtest(engine, title="SMA Crossover — AAPL", indicators=["sma_fast", "sma_slow"])
fig.write_html("backtest.html")
```

## Weight-Based Backtesting

For portfolio allocation strategies, skip the event loop entirely — just supply target weights per (date, symbol):

```python
import polars as pl
from quantloop import backtest_weights

# data: long-format DataFrame with columns date, symbol, close, weight
result = backtest_weights(
    data,
    resample="M",           # rebalance monthly
    resample_offset="2d",   # delay 2 trading days after month boundary
    fee_ratio=0.001,
    stop_loss=0.10,          # 10% per-position stop-loss
    position_limit=0.5,      # max 50% in any single name
    initial_capital=100_000,
)

print(result.metrics)        # standard BacktestMetrics
print(result.trades.head())  # per-trade log
print(result.next_actions)   # forward-looking rebalance actions
```

## Trade Data & DeFi Backtesting

QuantLoop can ingest raw DEX/AMM trade data, aggregate it into OHLCV bars, apply DeFi-specific indicators, and backtest with AMM-aware slippage — all in a single pipeline.

```python
import polars as pl
from quantloop import Engine, Strategy, indicators_defi as defi
from quantloop.core import BacktestContext
from quantloop.data.trades import aggregate_trades, validate_trades
from quantloop.slippage import AMMSlippage
from quantloop.universe import AgeFilter, CompositeFilter, VolumeFilter

# 1. Load and validate raw trades
trades = pl.read_parquet("trades.parquet")
assert validate_trades(trades.sort("symbol", "timestamp")).valid

# 2. Aggregate to 5-minute OHLCV bars
bars = aggregate_trades(trades.sort("symbol", "timestamp"), "5m", min_trades=3)

# 3. Define a strategy using DeFi indicators
class PumpMomentum(Strategy):
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            defi.buy_sell_ratio().over("symbol").alias("bs_ratio"),
            defi.trade_intensity(window=10).over("symbol").alias("intensity"),
        )

    def next(self, ctx: BacktestContext) -> None:
        for sym in ctx.symbols:
            row = ctx.row(sym)
            if row.get("bs_ratio", 0) > 0.7 and row.get("intensity", 0) > 2.0:
                ctx.portfolio.order_target_percent(sym, 0.1)
            elif row.get("bs_ratio", 0) < 0.3:
                ctx.portfolio.close_position(sym)

# 4. Run with AMM slippage and universe filtering
engine = Engine(
    PumpMomentum(),
    bars,
    initial_cash=100.0,
    commission=0.01,
    slippage=AMMSlippage(),
    universe_provider=CompositeFilter(AgeFilter(min_bars=5), VolumeFilter(min_volume=1.0)),
)
results = engine.run()
print(results)
```

## Examples

| Example | Description |
|---|---|
| [`example.py`](examples/example.py) | Basic SMA crossover |
| [`example_sma_crossover_stoploss.py`](examples/example_sma_crossover_stoploss.py) | SMA crossover with ATR stop-loss and trailing stop |
| [`example_rsi_bracket_orders.py`](examples/example_rsi_bracket_orders.py) | RSI mean reversion with bracket orders |
| [`example_momentum_rotation.py`](examples/example_momentum_rotation.py) | Multi-asset momentum rotation |
| [`example_ml_strategy.py`](examples/example_ml_strategy.py) | ML model integration |
| [`example_walk_forward.py`](examples/example_walk_forward.py) | Walk-forward analysis workflow |
| [`example_advanced_analysis.py`](examples/example_advanced_analysis.py) | Full workflow: optimization, heatmaps, Monte Carlo, permutation test |
| [`example_limit_orders.py`](examples/example_limit_orders.py) | Limit orders and stop-loss |
| [`example_trade_analysis.py`](examples/example_trade_analysis.py) | Trade-level analysis |
| [`example_plotting.py`](examples/example_plotting.py) | Interactive chart generation |
| [`example_commission.py`](examples/example_commission.py) | Commission model comparison |
| [`example_multi_asset.py`](examples/example_multi_asset.py) | Multi-asset dict input |
| [`example_weight_backtest.py`](examples/example_weight_backtest.py) | Weight-based portfolio backtest |
| [`example_defi_trades.py`](examples/example_defi_trades.py) | DeFi trade data pipeline with AMM slippage |

## Contributing

Contributions are welcome. Typical workflow:

1. Fork the repo and clone your fork
2. `pip install -e ".[dev]"`
3. Make changes and add tests in `tests/`
4. Run `pytest tests/ -q`, `ruff check quantloop tests`, and `mypy quantloop`
5. Open a pull request with a clear description of the change

See [Getting Started Guide](docs/getting-started.md) for a full walkthrough.

## Documentation

- [Getting Started Guide](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Complete Reference](docs/complete-reference.md)

## License

[MIT](LICENSE)
