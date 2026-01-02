"""Async data fetcher with rate limiting."""

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp
else:
    try:
        import aiohttp
    except ImportError:
        aiohttp = None  # type: ignore[assignment, unused-ignore]

from src.calculations.exceptions import RateLimitError
from src.calculations.financial_metrics import extract_financial_data_from_yfinance
from src.data.fetchers.base_fetcher import BaseFetcher
from src.data.fetchers.fetcher_utils import (
    build_ticker_data,
    calculate_metrics_from_financial_data_async,
    fetch_fundamental_data_shared,
    fetch_price_data_shared,
)
from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus
from src.utils.date_utils import get_last_business_day

logger = logging.getLogger("magicformula")


class AsyncYFinanceFetcher(BaseFetcher):
    """Async YFinance implementation with rate limiting."""

    def __init__(
        self,
        max_concurrent: int = 10,  # Conservative default to avoid overwhelming API
        requests_per_second: float = 5.0,  # Conservative default to respect rate limits
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
        self._rate_lock = asyncio.Lock()  # Protect rate limit state
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
                except RateLimitError:
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
                        logger.error(
                            f"Failed to fetch {symbol} after {self.retry_attempts} attempts: {e}"
                        )
                        return self._create_error_ticker_data(symbol, str(e))

            return self._create_error_ticker_data(symbol, "Max retries exceeded")

    async def _wait_for_rate_limit(self) -> None:
        """Wait to respect rate limit (thread-safe, lock released during sleep)."""
        # Calculate wait time while holding lock
        async with self._rate_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            wait_time = (
                self._min_request_interval - time_since_last
                if time_since_last < self._min_request_interval
                else 0.0
            )

        # Release lock during sleep to allow other coroutines to calculate their wait times
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        # Re-acquire lock to update timestamp atomically
        async with self._rate_lock:
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

        # Use async calculation function to avoid blocking event loop
        import aiohttp

        async with aiohttp.ClientSession() as session:
            # Returns (metrics, updated_financial_data) - updated_financial_data includes Alpha Vantage data
            metrics, updated_financial_data = await calculate_metrics_from_financial_data_async(
                financial_data, symbol, session
            )
        enterprise_value = metrics["enterprise_value"]
        earnings_yield = metrics["earnings_yield"]
        return_on_capital = metrics["return_on_capital"]
        acquirers_multiple = metrics["acquirers_multiple"]

        # Debug logging for missing Magic Formula metrics
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

        # Use shared function to build TickerData
        # Use updated_financial_data which includes Alpha Vantage data (NWC, NFA)
        return build_ticker_data(
            symbol=symbol,
            price_data=price_data,
            financial_data=updated_financial_data,
            fundamental_data=fundamental_data,
            metrics=metrics,
        )

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
            sector=None,
            industry=None,
            price=None,
            market_cap=None,
            total_debt=None,
            cash=None,
            ebit=None,
            net_working_capital=None,
            net_fixed_assets=None,
            enterprise_value=None,
            earnings_yield=None,
            return_on_capital=None,
            acquirers_multiple=None,
            price_index_6month=None,
            price_index_12month=None,
            book_to_market=None,
            free_cash_flow_yield=None,
            price_to_sales=None,
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
        """Fetch multiple tickers concurrently with batch price fetching.

        Uses batch price fetching to reduce API calls, then fetches fundamentals
        in parallel for each ticker.

        Args:
            symbols: List of ticker symbols.

        Returns:
            List of TickerData objects.
        """
        from src.data.fetchers.fetcher_utils import fetch_price_data_batch
        from src.utils.date_utils import get_last_business_day

        # Batch fetch all price data in one API call (much faster!)
        logger.info(f"Batch fetching price data for {len(symbols)} symbols...")
        loop = asyncio.get_event_loop()
        last_business_day = get_last_business_day()
        batch_price_data = await loop.run_in_executor(
            None,
            fetch_price_data_batch,
            symbols,
            last_business_day,
        )

        # Create shared aiohttp session for Alpha Vantage requests (connection pooling)
        if aiohttp is None:
            raise ImportError("aiohttp is required for async operations")
        async with aiohttp.ClientSession() as session:
            # Now fetch fundamentals in parallel (one per ticker) with rate limiting
            async def fetch_with_price(symbol: str) -> TickerData:
                """Fetch ticker data using pre-fetched price data with rate limiting."""
                # Check price data before acquiring semaphore to avoid deadlock
                price_data = batch_price_data.get(symbol, {})
                if not price_data or price_data.get("price") is None:
                    # Batch price fetch failed - create error TickerData instead of calling
                    # fetch_ticker_data() which would try to acquire semaphore again (deadlock risk)
                    logger.warning(f"{symbol}: Batch price fetch failed, creating error TickerData")
                    return self._create_error_ticker_data(
                        symbol,
                        "Batch price fetch failed - no price data available",
                    )

                async with self._rate_limiter:
                    await self._wait_for_rate_limit()

                    # Fetch fundamentals only (price already done)
                    loop = asyncio.get_event_loop()
                    fundamental_data = await loop.run_in_executor(
                        None,
                        fetch_fundamental_data_shared,
                        symbol,
                    )

                    financial_data = extract_financial_data_from_yfinance(
                        fundamental_data,
                        symbol=symbol,
                    )

                    # Use async calculation function to avoid blocking event loop
                    # Reuse shared session for Alpha Vantage requests
                    # Returns (metrics, updated_financial_data) - updated_financial_data includes Alpha Vantage data
                    (
                        metrics,
                        updated_financial_data,
                    ) = await calculate_metrics_from_financial_data_async(
                        financial_data, symbol, session
                    )
                    enterprise_value = metrics["enterprise_value"]
                    earnings_yield = metrics["earnings_yield"]
                    return_on_capital = metrics["return_on_capital"]
                    acquirers_multiple = metrics["acquirers_multiple"]

                    # Debug logging for missing Magic Formula metrics
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

                    # Use shared function to build TickerData
                    # Use updated_financial_data which includes Alpha Vantage data (NWC, NFA)
                    return build_ticker_data(
                        symbol=symbol,
                        price_data=price_data,
                        financial_data=updated_financial_data,
                        fundamental_data=fundamental_data,
                        metrics=metrics,
                    )

            # Fetch all fundamentals in parallel
            tasks = [fetch_with_price(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            ticker_data_list: list[TickerData] = []
            for i, result in enumerate(results):
                if isinstance(result, TickerData):
                    ticker_data_list.append(result)
                elif isinstance(result, Exception):
                    # Create error TickerData for failed symbol
                    failed_symbol = symbols[i] if i < len(symbols) else "Unknown"
                    logger.error(f"Error fetching {failed_symbol}: {result}")
                    error_ticker = self._create_error_ticker_data(
                        failed_symbol,
                        f"Async fetch error: {result}",
                    )
                    ticker_data_list.append(error_ticker)

            return ticker_data_list
