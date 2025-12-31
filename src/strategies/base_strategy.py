"""Base strategy class using Strategy pattern."""

from abc import ABC, abstractmethod

from src.data.models.ticker_data import TickerData
from src.data.types import StrategyResultDict


class BaseStrategy(ABC):
    """Abstract base class for investment strategies."""

    @abstractmethod
    def calculate(self, ticker_data: list[TickerData]) -> list[StrategyResultDict]:
        """Calculate strategy scores for a list of tickers.

        Args:
            ticker_data: List of TickerData objects.

        Returns:
            List of dictionaries with symbol, score, and rank.
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of the strategy.

        Returns:
            Strategy name.
        """
        pass
