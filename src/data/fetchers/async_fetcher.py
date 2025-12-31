"""Async data fetcher with rate limiting."""

import asyncio
import logging
import time
from datetime import datetime

from src.calculations.exceptions import InvalidDataError, RateLimitError
from src.calculations.financial_metrics import (
    calculate_acquirers_multiple,
    calculate_earnings_yield,
    calculate_enterprise_value,
    calculate_return_on_capital,
    extract_financial_data_from_yfinance,
)
from src.data.fetchers.base_fetcher import BaseFetcher
from src.data.fetchers.fetcher_utils import (
    fetch_fundamental_data_shared,
    fetch_price_data_shared,
)
from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus
from src.data.quality import assess_data_quality
from src.utils.date_utils import get_last_business_day

logger = logging.getLogger("magicformula")


class AsyncYFinanceFetcher(BaseFetcher):
    """Async YFinance implementation with rate limiting."""

    def __init__(
        self,
        max_concurrent: int = 5,
        requests_per_second: float = 2.0,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        use_cache: bool = True,
    ) -> None:
        """Initialize async fetcher.

        Args:
            max_concurrent: Maximum concurrent requests.
            requests_per_second: Rate limit (requests per second).
            retry_attempts: Number of retry attempts on failure.
            retry_delay: Initial delay between retries (seconds).
            use_cache: Whether to use caching (requires cache instance).
        """
        self.max_concurrent = max_concurrent
        self.requests_per_second = requests_per_second
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.use_cache = use_cache
        self._rate_limiter = asyncio.Semaphore(max_concurrent)
        self._last_request_time = 0.0
        self._min_request_interval = 1.0 / requests_per_second

    async def fetch_ticker_data(self, symbol: str) -> TickerData:
        """Fetch financial data for a single ticker (async).

        Args:
            symbol: Stock ticker symbol.

        Returns:
            TickerData object with financial metrics.
        """
        async with self._rate_limiter:
            await self._wait_for_rate_limit()

            for attempt in range(self.retry_attempts):
                try:
                    return await self._fetch_ticker_data_internal(symbol)
                except RateLimitError as e:
                    if attempt < self.retry_attempts - 1:
                        wait_time = self.retry_delay * (2**attempt)
                        logger.warning(
                            f"Rate limited for {symbol}, retrying in {wait_time}s",
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                except Exception as e:
                    if attempt < self.retry_attempts - 1:
                        wait_time = self.retry_delay * (2**attempt)
                        logger.warning(
                            f"Error fetching {symbol} (attempt {attempt + 1}), retrying in {wait_time}s: {e}",
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch {symbol} after {self.retry_attempts} attempts: {e}")
                        return self._create_error_ticker_data(symbol, str(e))

            return self._create_error_ticker_data(symbol, "Max retries exceeded")

    async def _wait_for_rate_limit(self) -> None:
        """Wait to respect rate limit."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._min_request_interval:
            wait_time = self._min_request_interval - time_since_last
            await asyncio.sleep(wait_time)

        self._last_request_time = time.time()

    async def _fetch_ticker_data_internal(self, symbol: str) -> TickerData:
        """Internal async fetch implementation.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            TickerData object.
        """
        loop = asyncio.get_event_loop()
        last_business_day = get_last_business_day()

        price_data = await loop.run_in_executor(
            None,
            fetch_price_data_shared,
            symbol,
            last_business_day,
        )

        fundamental_data = await loop.run_in_executor(
            None,
            fetch_fundamental_data_shared,
            symbol,
        )

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
                except (ValueError, Exception) as e:
                    logger.debug(
                        f"{symbol}: Could not calculate earnings_yield: {e}",
                    )

                try:
                    acquirers_multiple = calculate_acquirers_multiple(
                        financial_data["ebit"],
                        enterprise_value,
                    )
                except (ValueError, Exception) as e:
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
                except (ValueError, Exception, InvalidDataError) as e:
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
        return ticker_data


    def _create_error_ticker_data(self, symbol: str, error_msg: str) -> TickerData:
        """Create error ticker data.

        Args:
            symbol: Ticker symbol.
            error_msg: Error message.

        Returns:
            TickerData with error status.
        """
        return TickerData(
            symbol=symbol,
            status=TickerStatus.DATA_UNAVAILABLE,
            data_timestamp=datetime.now(),
            quality_score=0.0,
        )

    def fetch_multiple_tickers(self, symbols: list[str]) -> list[TickerData]:
        """Fetch multiple tickers (sync wrapper for async).

        Args:
            symbols: List of ticker symbols.

        Returns:
            List of TickerData objects.
        """
        return asyncio.run(self.fetch_multiple_tickers_async(symbols))

    async def fetch_multiple_tickers_async(
        self,
        symbols: list[str],
    ) -> list[TickerData]:
        """Fetch multiple tickers concurrently.

        Args:
            symbols: List of ticker symbols.

        Returns:
            List of TickerData objects.
        """
        tasks = [self.fetch_ticker_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ticker_data_list: list[TickerData] = []
        for result in results:
            if isinstance(result, TickerData):
                ticker_data_list.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in async fetch: {result}")

        return ticker_data_list

