"""Backtest context passed to strategy.next() on each bar."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quantloop.core.portfolio import Portfolio


class RowAccessor:
    """Provides dual access to bar data: ``ctx.row["close"]`` (property) and ``ctx.row("BTC")`` (method).

    In single-asset mode ``ctx.row`` behaves like a plain dict (backward-compatible).
    In multi-asset mode ``ctx.row("BTC")`` returns the dict for a specific symbol,
    while ``ctx.row`` (no call) returns the single symbol's dict if there is exactly one.
    """

    __slots__ = ("_data", "_symbols")

    def __init__(self, data: dict[str, dict[str, Any]], symbols: list[str]) -> None:
        self._data = data
        self._symbols = symbols

    # --- dict-like access (backward compat for single-asset ``ctx.row["close"]``) ---

    def __getitem__(self, key: str) -> Any:
        if len(self._symbols) == 1:
            return self._data[self._symbols[0]][key]
        raise KeyError(
            f"Ambiguous row access with {len(self._symbols)} symbols. Use ctx.row('SYMBOL')['{key}'] instead."
        )

    def __contains__(self, key: object) -> bool:
        if len(self._symbols) == 1:
            return key in self._data[self._symbols[0]]
        return False

    def get(self, key: str, default: Any = None) -> Any:
        if len(self._symbols) == 1:
            return self._data[self._symbols[0]].get(key, default)
        return default

    def keys(self) -> Any:
        if len(self._symbols) == 1:
            return self._data[self._symbols[0]].keys()
        raise RuntimeError("Ambiguous: multiple symbols. Use ctx.row('SYMBOL').keys().")

    def values(self) -> Any:
        if len(self._symbols) == 1:
            return self._data[self._symbols[0]].values()
        raise RuntimeError("Ambiguous: multiple symbols. Use ctx.row('SYMBOL').values().")

    def items(self) -> Any:
        if len(self._symbols) == 1:
            return self._data[self._symbols[0]].items()
        raise RuntimeError("Ambiguous: multiple symbols. Use ctx.row('SYMBOL').items().")

    # --- callable access (multi-asset ``ctx.row("BTC")``) ---

    def __call__(self, symbol: str | None = None) -> dict[str, Any]:
        if symbol is None:
            if len(self._symbols) == 1:
                return self._data[self._symbols[0]]
            raise ValueError(
                f"Must specify symbol when {len(self._symbols)} symbols are present. Available: {self._symbols}"
            )
        if symbol not in self._data:
            raise KeyError(f"Symbol '{symbol}' not available this bar. Available: {self._symbols}")
        return self._data[symbol]

    def __repr__(self) -> str:
        if len(self._symbols) == 1:
            return repr(self._data[self._symbols[0]])
        return f"RowAccessor(symbols={self._symbols})"


@dataclass
class BacktestContext:
    """Context object passed to Strategy.next() on each bar.

    Attributes:
        timestamp: Current timestamp.
        bar_index: Current bar index in the dataset.
        portfolio: Reference to the Portfolio instance.
        symbols: Tradeable symbols on this bar (after universe filtering).
        data: Per-symbol bar data ``{symbol: {col: value, ...}}``.
        row: Dual-access helper — use ``ctx.row["close"]`` in single-asset mode
            or ``ctx.row("BTC")["close"]`` in multi-asset mode.
        first_seen_bar: Bar index when each symbol first appeared in the data.
        bar_count: Number of bars each symbol has been active so far.
        available_symbols: All symbols with data on this bar (before universe filtering).
    """

    timestamp: Any
    bar_index: int
    portfolio: "Portfolio"
    symbols: list[str]
    data: dict[str, dict[str, Any]]
    row: RowAccessor = field(default_factory=lambda: RowAccessor({}, []))  # set by Engine
    first_seen_bar: dict[str, int] = field(default_factory=dict)
    bar_count: dict[str, int] = field(default_factory=dict)
    available_symbols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Build RowAccessor from data if not already set properly
        if not self.row._symbols and self.data:
            object.__setattr__(self, "row", RowAccessor(self.data, self.symbols))
