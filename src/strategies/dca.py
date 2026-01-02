"""Dollar Cost Averaging (DCA) strategy implementation."""

import logging
from datetime import date, timedelta

import pandas as pd

from src.data.models.ticker_data import TickerData
from src.data.types import StrategyResultDict
from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger("magicformula")


class DCAStrategy(BaseStrategy):
    """Dollar Cost Averaging strategy simulation.

    Simulates periodic investments over time and calculates returns
    accounting for dividends and slippage.
    """

    def __init__(
        self,
        investment_amount: float = 1000.0,
        frequency: str = "monthly",
        start_date: date | None = None,
        end_date: date | None = None,
        dividend_reinvestment: bool = True,
        slippage_bps: int = 10,
    ) -> None:
        """Initialize DCA strategy.

        Args:
            investment_amount: Amount to invest per period.
            frequency: Investment frequency ('monthly', 'weekly', 'daily').
            start_date: Start date for DCA simulation.
            end_date: End date for DCA simulation.
            dividend_reinvestment: Whether to reinvest dividends.
            slippage_bps: Slippage in basis points (e.g., 10 = 0.1%).
        """
        self.investment_amount = investment_amount
        self.frequency = frequency
        self.start_date = start_date or date.today() - timedelta(days=365)
        self.end_date = end_date or date.today()
        self.dividend_reinvestment = dividend_reinvestment
        self.slippage_bps = slippage_bps

    def calculate(self, ticker_data: list[TickerData]) -> list[StrategyResultDict]:
        """Calculate DCA returns for each ticker.

        Args:
            ticker_data: List of TickerData objects.

        Returns:
            List of dictionaries with DCA simulation results.
        """
        results: list[StrategyResultDict] = []

        for ticker in ticker_data:
            if not ticker.symbol:
                continue

            try:
                dca_result = self._simulate_dca(ticker.symbol)
                results.append(
                    {
                        "symbol": ticker.symbol,
                        **dca_result,
                    }
                )
            except Exception as e:
                logger.error(f"Error calculating DCA for {ticker.symbol}: {e}")
                continue

        def sort_key(x: StrategyResultDict) -> float:
            return_val = x.get("total_return_pct")
            if isinstance(return_val, (int, float)):
                return float(return_val)
            return float("-inf")

        results = sorted(results, key=sort_key, reverse=True)

        return results

    def _simulate_dca(self, symbol: str) -> dict[str, float | int]:
        """Simulate DCA for a single ticker.

        Properly handles dividends on their ex-dividend dates, not just purchase dates.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Dictionary with DCA simulation results.
        """
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        hist = ticker.history(
            start=self.start_date.strftime("%Y-%m-%d"),
            end=(self.end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )

        if hist.empty:
            return {
                "total_invested": 0.0,
                "total_value": 0.0,
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "num_purchases": 0,
                "num_dividends": 0,
            }

        purchase_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=self._get_pandas_freq(),
        )
        purchase_dates_set = {pd.Timestamp(d).date() for d in purchase_dates}

        total_invested = 0.0
        total_shares = 0.0
        num_purchases = 0
        num_dividends = 0

        slippage_multiplier = 1.0 + (self.slippage_bps / 10000.0)

        for date_idx in hist.index:
            date_obj = date_idx.date()

            if date_obj in purchase_dates_set:
                price_row = hist.loc[hist.index == date_idx]
                if not price_row.empty:
                    price = float(price_row["Close"].iloc[0]) * slippage_multiplier
                    shares_purchased = self.investment_amount / price

                    total_invested += self.investment_amount
                    total_shares += shares_purchased
                    num_purchases += 1

            if self.dividend_reinvestment and "Dividends" in hist.columns:
                dividend_row = hist.loc[hist.index == date_idx]
                if not dividend_row.empty:
                    dividend = float(dividend_row["Dividends"].iloc[0])
                    if dividend > 0 and total_shares > 0:
                        price = float(dividend_row["Close"].iloc[0])
                        dividend_value = total_shares * dividend
                        additional_shares = dividend_value / price
                        total_shares += additional_shares
                        num_dividends += 1

        final_price = float(hist["Close"].iloc[-1])
        total_value = total_shares * final_price
        total_return = total_value - total_invested
        total_return_pct = (total_return / total_invested * 100) if total_invested > 0 else 0.0

        return {
            "total_invested": total_invested,
            "total_value": total_value,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "num_purchases": num_purchases,
            "num_dividends": num_dividends,
            "final_price": final_price,
        }

    def _get_pandas_freq(self) -> str:
        """Get pandas frequency string.

        Returns:
            Pandas frequency string.
        """
        freq_map = {
            "daily": "D",
            "weekly": "W",
            "monthly": "M",
        }
        return freq_map.get(self.frequency, "M")

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "Dollar Cost Averaging"
