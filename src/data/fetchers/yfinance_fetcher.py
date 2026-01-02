"""YFinance data fetcher implementation."""

import logging
import time

from src.calculations.exceptions import InvalidDataError, MissingDataError
from src.calculations.financial_metrics import extract_financial_data_from_yfinance
from src.data.cache import TickerCache
from src.data.fetchers.base_fetcher import BaseFetcher
from src.data.fetchers.fetcher_utils import (
    build_ticker_data,
    calculate_metrics_from_financial_data,
    fetch_fundamental_data_shared,
    fetch_price_data_shared,
)
from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus
from src.utils.date_utils import get_last_business_day

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
                    logger.warning(
                        f"Network error fetching {symbol} (attempt {attempt + 1}), retrying...: {e}"
                    )
                    time.sleep(retry_delay * (2**attempt))
                else:
                    logger.error(f"Failed to fetch {symbol} after {max_retries} attempts: {e}")
                    raise

        financial_data = extract_financial_data_from_yfinance(
            fundamental_data,
            symbol=symbol,
        )

        # Use shared calculation function to eliminate duplication
        # Returns (metrics, updated_financial_data) - updated_financial_data includes Alpha Vantage data
        metrics, updated_financial_data = calculate_metrics_from_financial_data(
            financial_data, symbol
        )
        enterprise_value = metrics["enterprise_value"]
        earnings_yield = metrics["earnings_yield"]
        return_on_capital = metrics["return_on_capital"]

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
        ticker_data = build_ticker_data(
            symbol=symbol,
            price_data=price_data,
            financial_data=updated_financial_data,
            fundamental_data=fundamental_data,
            metrics=metrics,
        )

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
                error_ticker = self._create_error_ticker_data(symbol)
                results.append(error_ticker)
            except Exception as e:
                # Network/API errors - log and create error ticker
                logger.error(f"Error fetching data for {symbol}: {e}")
                error_ticker = self._create_error_ticker_data(symbol)
                results.append(error_ticker)

            if i < len(symbols) - 1:
                time.sleep(self.sleep_time)

        return results

    def _create_error_ticker_data(self, symbol: str) -> TickerData:
        """Create error ticker data.

        Args:
            symbol: Ticker symbol.

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
            data_timestamp=None,
            quality_score=0.0,
        )
