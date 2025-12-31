"""Magic Formula strategy implementation (Joel Greenblatt)."""

import logging

from src.calculations.ranking import calculate_combined_rank, rank_series
from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus
from src.data.types import StrategyResultDict
from src.data.validators import (
    filter_valid_tickers,
    is_financial_sector,
    validate_ticker_data_for_magic_formula,
)
from src.strategies.base_strategy import BaseStrategy

logger = logging.getLogger("magicformula")


class MagicFormulaStrategy(BaseStrategy):
    """Magic Formula strategy: Rank by Earnings Yield and Return on Capital.

    The Magic Formula ranks stocks by:
    1. Earnings Yield (EBIT / Enterprise Value) - higher is better
    2. Return on Capital (EBIT / Capital Employed) - higher is better

    Final score is the sum of both ranks (lower is better).
    """

    def calculate(self, ticker_data: list[TickerData]) -> list[StrategyResultDict]:
        """Calculate Magic Formula scores.

        Args:
            ticker_data: List of TickerData objects.

        Returns:
            List of dictionaries with symbol, metrics, ranks, and score.
            Includes ALL tickers (even invalid ones) so CSV output has complete data.
        """
        valid_tickers = filter_valid_tickers(
            ticker_data,
            validate_ticker_data_for_magic_formula,
        )

        if not valid_tickers:
            logger.warning("No valid tickers for Magic Formula calculation")
            # Log why tickers were rejected (use INFO level so user can see)
            rejection_summary: dict[str, int] = {}
            for ticker in ticker_data:
                if ticker.status == TickerStatus.ACTIVE:
                    reasons = []
                    if is_financial_sector(ticker):
                        reasons.append(f"financial sector ({ticker.sector})")
                    if ticker.earnings_yield is None:
                        reasons.append("missing earnings_yield")
                    elif ticker.earnings_yield <= 0:
                        reasons.append(f"earnings_yield <= 0 ({ticker.earnings_yield:.4f})")
                    if ticker.return_on_capital is None:
                        reasons.append("missing return_on_capital")
                    elif ticker.return_on_capital <= 0:
                        reasons.append(f"return_on_capital <= 0 ({ticker.return_on_capital:.4f})")
                    if ticker.quality_score is not None and ticker.quality_score < 0.5:
                        reasons.append(f"low quality_score ({ticker.quality_score:.2f})")
                    if reasons:
                        reason_key = ", ".join(reasons)
                        rejection_summary[reason_key] = rejection_summary.get(reason_key, 0) + 1
                        logger.info(
                            f"{ticker.symbol} rejected: {reason_key}",
                        )

            # Log summary
            if rejection_summary:
                logger.info("Rejection summary:")
                for reason, count in sorted(
                    rejection_summary.items(), key=lambda x: x[1], reverse=True
                ):
                    logger.info(f"  {count} ticker(s): {reason}")

            # Return ALL tickers (even invalid ones) so CSV output includes all data
            # Invalid tickers will have None ranks/scores but still show calculated metrics
            logger.info("Returning all tickers with calculated metrics (including invalid ones)")
            return [ticker.to_dict() for ticker in ticker_data]

        # Calculate ranks only for valid tickers
        data = [ticker.to_dict() for ticker in valid_tickers]

        data = rank_series(data, "earnings_yield", ascending=False, method="min")
        data = rank_series(data, "return_on_capital", ascending=False, method="min")

        data = calculate_combined_rank(
            data,
            ["earnings_yield_rank", "return_on_capital_rank"],
            "magic_formula_score",
        )

        def sort_key(x: StrategyResultDict) -> float:
            score_val = x.get("magic_formula_score")
            if isinstance(score_val, (int, float)):
                return float(score_val)
            return float("inf")

        data = sorted(data, key=sort_key)

        # Add invalid tickers to output (without ranks) so CSV has complete data
        valid_symbols = {
            str(t.get("symbol", "")).upper() for t in data if t.get("symbol") is not None
        }
        for ticker in ticker_data:
            if ticker.symbol.upper() not in valid_symbols:
                ticker_dict = ticker.to_dict()
                # Mark as invalid
                ticker_dict["magic_formula_score"] = None
                ticker_dict["earnings_yield_rank"] = None
                ticker_dict["return_on_capital_rank"] = None
                data.append(ticker_dict)

        return data

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "Magic Formula"
