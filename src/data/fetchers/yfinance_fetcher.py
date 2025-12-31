"""YFinance data fetcher implementation."""

import logging
import time
from datetime import datetime

from src.calculations.exceptions import InvalidDataError, MissingDataError
from src.calculations.financial_metrics import (
    calculate_acquirers_multiple,
    calculate_earnings_yield,
    calculate_enterprise_value,
    calculate_return_on_capital,
    extract_financial_data_from_yfinance,
)
from src.data.cache import TickerCache
from src.data.fetchers.base_fetcher import BaseFetcher
from src.data.fetchers.fetcher_utils import (
    fetch_fundamental_data_shared,
    fetch_price_data_batch,
    fetch_price_data_shared,
)
from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus
from src.data.quality import assess_data_quality
from src.utils.date_utils import get_last_business_day
from src.utils.decorators import retry

logger = logging.getLogger("magicformula")


class YFinanceFetcher(BaseFetcher):
    """YFinance implementation of BaseFetcher."""

    def __init__(
        self,
        sleep_time: float = 1.0,
        use_cache: bool = True,
        cache_path: str | None = None,
    ) -> None:
        """Initialize YFinance fetcher.

        Args:
            sleep_time: Time to sleep between requests (seconds).
            use_cache: Whether to use caching.
            cache_path: Optional path to cache database.
        """
        self.sleep_time = sleep_time
        self.use_cache = use_cache
        self.cache = TickerCache(cache_path) if use_cache else None

    def fetch_ticker_data(self, symbol: str) -> TickerData:
        """Fetch financial data for a single ticker.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            TickerData object with financial metrics.
        """
        if self.use_cache and self.cache:
            cached_data = self.cache.get(symbol)
            if cached_data:
                logger.debug(f"Using cached data for {symbol}")
                return cached_data

        logger.info(f"Fetching data for {symbol}")

        last_business_day = get_last_business_day()

        # Fetch data with retry for network errors only
        max_retries = 3
        retry_delay = 1.0
        for attempt in range(max_retries):
            try:
                price_data = fetch_price_data_shared(symbol, last_business_day)
                fundamental_data = fetch_fundamental_data_shared(symbol)
                break
            except (ConnectionError, TimeoutError, ValueError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Network error fetching {symbol} (attempt {attempt + 1}), retrying...: {e}")
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Failed to fetch {symbol} after {max_retries} attempts: {e}")
                    raise

        financial_data = extract_financial_data_from_yfinance(
            fundamental_data,
            symbol=symbol,
        )

        enterprise_value = None
        earnings_yield = None
        return_on_capital = None
        acquirers_multiple = None

        if financial_data["market_cap"]:
            enterprise_value = calculate_enterprise_value(
                financial_data["market_cap"],
                financial_data["total_debt"] or 0.0,
                financial_data["cash"] or 0.0,
            )

            if financial_data["ebit"] and enterprise_value:
                try:
                    earnings_yield = calculate_earnings_yield(
                        financial_data["ebit"],
                        enterprise_value,
                    )
                except (MissingDataError, ValueError, InvalidDataError) as e:
                    logger.debug(
                        f"{symbol}: Could not calculate earnings_yield: {e}",
                    )

                try:
                    acquirers_multiple = calculate_acquirers_multiple(
                        financial_data["ebit"],
                        enterprise_value,
                    )
                except (MissingDataError, ValueError, InvalidDataError) as e:
                    logger.debug(
                        f"{symbol}: Could not calculate acquirers_multiple: {e}",
                    )

            if financial_data["ebit"] and (
                financial_data["net_working_capital"] is not None
                or financial_data["net_fixed_assets"] is not None
            ):
                try:
                    return_on_capital = calculate_return_on_capital(
                        financial_data["ebit"],
                        financial_data["net_working_capital"],
                        financial_data["net_fixed_assets"],
                    )
                except (MissingDataError, ValueError, InvalidDataError) as e:
                    logger.debug(
                        f"{symbol}: Could not calculate return_on_capital: {e}",
                    )
        else:
            logger.debug(f"{symbol}: Missing market_cap, cannot calculate EV-based metrics")

        if earnings_yield is None or return_on_capital is None:
            logger.debug(
                f"{symbol}: Missing Magic Formula metrics - "
                f"EBIT: {financial_data.get('ebit')}, "
                f"EV: {enterprise_value}, "
                f"NWC: {financial_data.get('net_working_capital')}, "
                f"NFA: {financial_data.get('net_fixed_assets')}, "
                f"EY: {earnings_yield}, "
                f"ROC: {return_on_capital}",
            )

        ticker_data = TickerData(
            symbol=symbol,
            sector=financial_data.get("sector"),
            industry=financial_data.get("industry"),
            price=price_data["price"],
            market_cap=financial_data["market_cap"],
            total_debt=financial_data["total_debt"],
            cash=financial_data["cash"],
            ebit=financial_data["ebit"],
            net_working_capital=financial_data["net_working_capital"],
            net_fixed_assets=financial_data["net_fixed_assets"],
            enterprise_value=enterprise_value,
            earnings_yield=earnings_yield,
            return_on_capital=return_on_capital,
            acquirers_multiple=acquirers_multiple,
            price_index_6month=price_data["price_index_6month"],
            price_index_12month=price_data["price_index_12month"],
            book_to_market=fundamental_data.get("book_to_market"),
            free_cash_flow_yield=fundamental_data.get("free_cash_flow_yield"),
            price_to_sales=fundamental_data.get("price_to_sales"),
            data_timestamp=datetime.now(),
        )

        ticker_data = assess_data_quality(ticker_data)

        if self.use_cache and self.cache:
            self.cache.set(symbol, ticker_data)

        return ticker_data

    def fetch_multiple_tickers(self, symbols: list[str]) -> list[TickerData]:
        """Fetch financial data for multiple tickers sequentially.

        Args:
            symbols: List of stock ticker symbols.

        Returns:
            List of TickerData objects.
        """
        results: list[TickerData] = []

        for i, symbol in enumerate(symbols):
            try:
                ticker_data = self.fetch_ticker_data(symbol)
                results.append(ticker_data)
            except (InvalidDataError, MissingDataError) as e:
                # Calculation errors - don't retry, create error ticker
                logger.warning(
                    f"{symbol}: Calculation error (not retrying): {e}",
                )
                error_ticker = TickerData(
                    symbol=symbol,
                    status=TickerStatus.DATA_UNAVAILABLE,
                    quality_score=0.0,
                )
                results.append(error_ticker)
            except Exception as e:
                # Network/API errors - log and create error ticker
                logger.error(f"Error fetching data for {symbol}: {e}")
                error_ticker = TickerData(
                    symbol=symbol,
                    status=TickerStatus.DATA_UNAVAILABLE,
                    quality_score=0.0,
                )
                results.append(error_ticker)

            if i < len(symbols) - 1:
                time.sleep(self.sleep_time)

        return results


