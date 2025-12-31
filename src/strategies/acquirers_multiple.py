"""Acquirer's Multiple strategy implementation (Tobias Carlisle)."""

import logging

from src.calculations.ranking import rank_series
from src.data.models.ticker_data import TickerData
from src.data.types import StrategyResultDict
from src.data.validators import (
    filter_valid_tickers,
    validate_ticker_data_for_acquirers_multiple,
)
from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger("magicformula")


class AcquirersMultipleStrategy(BaseStrategy):
    """Acquirer's Multiple strategy: Rank by EV/EBIT (lower is better).

    The Acquirer's Multiple ranks stocks by:
    EV/EBIT ratio - lower values indicate better value.
    """

    def calculate(self, ticker_data: list[TickerData]) -> list[StrategyResultDict]:
        """Calculate Acquirer's Multiple scores.

        Args:
            ticker_data: List of TickerData objects.

        Returns:
            List of dictionaries with symbol, metrics, and rank.
        """
        valid_tickers = filter_valid_tickers(
            ticker_data,
            validate_ticker_data_for_acquirers_multiple,
        )

        if not valid_tickers:
            logger.warning("No valid tickers for Acquirer's Multiple calculation")
            return []

        data = [ticker.to_dict() for ticker in valid_tickers]

        data = rank_series(data, "acquirers_multiple", ascending=True, method="min")

        def sort_key(x: StrategyResultDict) -> float:
            rank_val = x.get("acquirers_multiple_rank")
            if isinstance(rank_val, (int, float)):
                return float(rank_val)
            return float("inf")

        data = sorted(data, key=sort_key)

        return data

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "Acquirer's Multiple"
