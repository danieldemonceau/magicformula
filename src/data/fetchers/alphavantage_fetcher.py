"""Alpha Vantage API fallback fetcher for missing financial data.

This module provides async-compatible Alpha Vantage API integration with:
- Shared rate limiting across all requests (prevents API limit violations)
- Caching to avoid redundant API calls
- Async/await support for non-blocking operation
- Configurable timeouts and rate limits
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

    # Type alias for aiohttp.ClientSession when available
    ClientSessionType = aiohttp.ClientSession
    # Type alias for Alpha Vantage API response (JSON dict)
    AlphaVantageResponse = dict[
        str, str | int | float | None | list[dict[str, str | int | float | None]]
    ]
else:
    try:
        import aiohttp

        ClientSessionType = aiohttp.ClientSession
    except ImportError:
        aiohttp = None  # type: ignore[assignment, unused-ignore]
        ClientSessionType = type(None)  # type: ignore[assignment, unused-ignore]

from config.settings_pydantic import settings

logger = logging.getLogger("magicformula")

# Alpha Vantage API endpoints
BASE_URL = "https://www.alphavantage.co/query"

# Shared rate limiter for all Alpha Vantage requests
_rate_limiter_lock = asyncio.Lock()
_last_request_time: float = 0.0


def _validate_symbol(symbol: str) -> str:
    """Validate and normalize ticker symbol.

    Args:
        symbol: Ticker symbol to validate.

    Returns:
        Normalized uppercase symbol.

    Raises:
        ValueError: If symbol is invalid.
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError(f"Invalid symbol: {symbol}")
    symbol_upper = symbol.strip().upper()
    if not symbol_upper or len(symbol_upper) > 10:
        raise ValueError(f"Invalid symbol format: {symbol}")
    return symbol_upper


async def _wait_for_rate_limit() -> None:
    """Wait to respect Alpha Vantage rate limits using shared rate limiter.

    Alpha Vantage free tier: 5 calls per minute = 12 seconds between calls.
    Uses a shared lock to ensure all requests respect the same rate limit.
    """
    global _last_request_time

    async with _rate_limiter_lock:
        now = asyncio.get_event_loop().time()
        time_since_last = now - _last_request_time
        rate_limit_delay = getattr(settings, "av_rate_limit_delay", 12.1)

        if time_since_last < rate_limit_delay:
            wait_time = rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before Alpha Vantage request")
            await asyncio.sleep(wait_time)

        _last_request_time = asyncio.get_event_loop().time()


class AlphaVantageCache:
    """Simple cache for Alpha Vantage API responses."""

    def __init__(self, cache_path: Path | str | None = None) -> None:
        """Initialize cache.

        Args:
            cache_path: Path to SQLite database file. If None, uses settings default.
        """
        if cache_path is None:
            cache_path = settings.cache_path.parent / "alphavantage_cache.db"
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS av_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON av_cache(expires_at)
                """,
            )
            conn.commit()

    def _make_key(self, symbol: str, function: str) -> str:
        """Create cache key.

        Args:
            symbol: Ticker symbol.
            function: Alpha Vantage function name.

        Returns:
            Cache key string.
        """
        return f"{symbol.upper()}:{function}"

    def get(self, symbol: str, function: str) -> AlphaVantageResponse | None:
        """Get cached data.

        Args:
            symbol: Ticker symbol.
            function: Alpha Vantage function name.

        Returns:
            Cached data dict or None if not found/expired.
        """
        if not getattr(settings, "cache_enabled", True):
            return None

        try:
            cache_key = self._make_key(symbol, function)
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT data, expires_at
                    FROM av_cache
                    WHERE cache_key = ? AND expires_at > ?
                    """,
                    (cache_key, datetime.now().isoformat()),
                )
                row = cursor.fetchone()

                if row:
                    parsed_data = json.loads(row[0])
                    # Type narrowing: json.loads returns Any, but we know it's our expected structure
                    if isinstance(parsed_data, dict):
                        logger.debug(f"Alpha Vantage cache hit for {symbol}:{function}")
                        return parsed_data
                    return None

                return None
        except Exception as e:
            logger.debug(f"Error reading Alpha Vantage cache: {e}")
            return None

    def set(self, symbol: str, function: str, data: AlphaVantageResponse) -> None:
        """Cache data.

        Args:
            symbol: Ticker symbol.
            function: Alpha Vantage function name.
            data: Data to cache.
        """
        if not getattr(settings, "cache_enabled", True):
            return

        try:
            cache_key = self._make_key(symbol, function)
            expires_at = datetime.now() + timedelta(
                hours=getattr(settings, "cache_expiration_hours", 24)
            )

            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO av_cache (cache_key, data, timestamp, expires_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        json.dumps(data),
                        datetime.now().isoformat(),
                        expires_at.isoformat(),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Error writing Alpha Vantage cache: {e}")


# Global cache instance
_av_cache = AlphaVantageCache()


async def _fetch_alphavantage_async(
    symbol: str,
    function: str,
    session: "ClientSessionType | None" = None,
) -> "AlphaVantageResponse":
    """Fetch data from Alpha Vantage API (async).

    Args:
        symbol: Stock ticker symbol.
        function: Alpha Vantage function name (OVERVIEW, BALANCE_SHEET, INCOME_STATEMENT).
        session: Optional aiohttp session for connection pooling.

    Returns:
        Dictionary with API response data or empty dict if error.
    """
    symbol = _validate_symbol(symbol)
    api_key = getattr(settings, "av_apikey", None)
    if not api_key:
        logger.warning("Alpha Vantage API key not configured")
        return {}

    # Check cache first
    cached = _av_cache.get(symbol, function)
    if cached is not None:
        return cached

    # Wait for rate limit
    await _wait_for_rate_limit()

    if aiohttp is None:
        logger.warning("aiohttp is not installed. Alpha Vantage async fetching is unavailable.")
        return {}

    timeout = aiohttp.ClientTimeout(total=getattr(settings, "av_timeout_seconds", 10))
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key,
    }

    try:
        if session:
            async with session.get(BASE_URL, params=params, timeout=timeout) as response:
                response.raise_for_status()
                json_data = await response.json()
                data = json_data if isinstance(json_data, dict) else {}
        else:
            async with (
                aiohttp.ClientSession(timeout=timeout) as temp_session,
                temp_session.get(BASE_URL, params=params) as response,
            ):
                response.raise_for_status()
                json_data = await response.json()
                data = json_data if isinstance(json_data, dict) else {}

        # Check for API errors
        if "Error Message" in data:
            logger.warning(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
            return {}
        if "Note" in data:
            logger.warning(f"Alpha Vantage rate limit for {symbol}: {data['Note']}")
            return {}

        # Cache successful response
        _av_cache.set(symbol, function, data)

        return data

    except Exception as e:
        if aiohttp is not None and isinstance(e, aiohttp.ClientError):
            logger.warning(f"Alpha Vantage request error for {symbol}: {e}")
        elif isinstance(e, (KeyError, ValueError, TypeError)):
            logger.warning(f"Alpha Vantage parsing error for {symbol}: {e}")
        else:
            logger.warning(f"Alpha Vantage error for {symbol}: {e}")
        return {}


def _parse_numeric(value: str | int | float | None) -> float | None:
    """Parse numeric value from Alpha Vantage response.

    Args:
        value: Value to parse (can be string, int, float, or None).

    Returns:
        Parsed float or None if invalid.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if cleaned == "None" or cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


async def fetch_company_overview_alphavantage_async(
    symbol: str,
    session: "ClientSessionType | None" = None,
) -> dict[str, float | None]:
    """Fetch company overview (including calculated ratios) from Alpha Vantage API (async).

    Args:
        symbol: Stock ticker symbol.
        session: Optional aiohttp session for connection pooling.

    Returns:
        Dictionary with return_on_capital if available.
    """
    symbol = _validate_symbol(symbol)
    data = await _fetch_alphavantage_async(symbol, "OVERVIEW", session)

    if not data:
        return {}

    # Type narrowing: data.get() returns the union type, but we know it's a dict
    roc_value = data.get("ReturnOnCapitalTTM")
    roic_value = data.get("ReturnOnInvestedCapitalTTM")
    return_on_capital = _parse_numeric(
        roc_value if isinstance(roc_value, (str, int, float, type(None))) else None
    )
    roic = _parse_numeric(
        roic_value if isinstance(roic_value, (str, int, float, type(None))) else None
    )

    # Use ROIC as ROC if available (they're similar metrics)
    if return_on_capital is None and roic is not None:
        return_on_capital = roic

    result: dict[str, float | None] = {}
    if return_on_capital is not None:
        result["return_on_capital"] = return_on_capital
        logger.debug(
            f"{symbol}: Found pre-calculated ROC in Alpha Vantage: {return_on_capital:.4f}"
        )

    return result


async def fetch_balance_sheet_alphavantage_async(
    symbol: str,
    session: "ClientSessionType | None" = None,
) -> dict[str, float | None]:
    """Fetch balance sheet data from Alpha Vantage API (async).

    Args:
        symbol: Stock ticker symbol.
        session: Optional aiohttp session for connection pooling.

    Returns:
        Dictionary with balance sheet data.
    """
    symbol = _validate_symbol(symbol)
    data = await _fetch_alphavantage_async(symbol, "BALANCE_SHEET", session)

    if not data:
        return {}

    annual_reports_raw = data.get("annualReports", [])
    if not annual_reports_raw or not isinstance(annual_reports_raw, list):
        logger.debug(f"No annual reports found for {symbol} in Alpha Vantage")
        return {}

    latest_report_raw = annual_reports_raw[0]
    if not isinstance(latest_report_raw, dict):
        logger.debug(f"Invalid report format for {symbol} in Alpha Vantage")
        return {}

    latest_report: dict[str, str | int | float | None] = latest_report_raw

    current_assets_raw = latest_report.get("totalCurrentAssets")
    current_liabilities_raw = latest_report.get("totalCurrentLiabilities")
    property_plant_equipment_raw = latest_report.get("propertyPlantAndEquipmentNet")
    property_plant_equipment_fallback_raw = latest_report.get("propertyPlantEquipment")

    current_assets = _parse_numeric(
        current_assets_raw
        if isinstance(current_assets_raw, (str, int, float, type(None)))
        else None
    )
    current_liabilities = _parse_numeric(
        current_liabilities_raw
        if isinstance(current_liabilities_raw, (str, int, float, type(None)))
        else None
    )
    property_plant_equipment = _parse_numeric(
        property_plant_equipment_raw
        if isinstance(property_plant_equipment_raw, (str, int, float, type(None)))
        else None
    )
    # Fallback for PPE if not available
    if property_plant_equipment is None:
        property_plant_equipment = _parse_numeric(
            property_plant_equipment_fallback_raw
            if isinstance(property_plant_equipment_fallback_raw, (str, int, float, type(None)))
            else None
        )

    return {
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "net_ppe": property_plant_equipment,
    }


async def fetch_income_statement_alphavantage_async(
    symbol: str,
    session: "ClientSessionType | None" = None,
) -> dict[str, float | None]:
    """Fetch income statement data from Alpha Vantage API (async).

    Args:
        symbol: Stock ticker symbol.
        session: Optional aiohttp session for connection pooling.

    Returns:
        Dictionary with income statement data.
    """
    symbol = _validate_symbol(symbol)
    data = await _fetch_alphavantage_async(symbol, "INCOME_STATEMENT", session)

    if not data:
        return {}

    annual_reports_raw = data.get("annualReports", [])
    if not annual_reports_raw or not isinstance(annual_reports_raw, list):
        logger.debug(f"No annual reports found for {symbol} in Alpha Vantage")
        return {}

    latest_report_raw = annual_reports_raw[0]
    if not isinstance(latest_report_raw, dict):
        logger.debug(f"Invalid report format for {symbol} in Alpha Vantage")
        return {}

    latest_report: dict[str, str | int | float | None] = latest_report_raw

    operating_income_raw = latest_report.get("operatingIncome")
    ebit_raw = latest_report.get("ebit")

    operating_income = _parse_numeric(
        operating_income_raw
        if isinstance(operating_income_raw, (str, int, float, type(None)))
        else None
    )
    ebit = _parse_numeric(ebit_raw if isinstance(ebit_raw, (str, int, float, type(None))) else None)
    if ebit is None:
        ebit = operating_income

    return {
        "ebit": ebit,
        "operating_income": operating_income,
    }


async def fetch_missing_financial_data_alphavantage_async(
    symbol: str,
    missing_fields: list[str],
    session: "ClientSessionType | None" = None,
) -> dict[str, float | None]:
    """Fetch missing financial data from Alpha Vantage as fallback (async).

    First tries to get pre-calculated ROC from company overview, then falls back to balance sheet.

    Args:
        symbol: Stock ticker symbol.
        missing_fields: List of fields that are missing.
        session: Optional aiohttp session for connection pooling.

    Returns:
        Dictionary with fetched data (only for requested missing fields).
    """
    symbol = _validate_symbol(symbol)
    result: dict[str, float | None] = {}

    # First, try to get pre-calculated ROC from company overview (if requested)
    if "return_on_capital" in missing_fields:
        logger.debug(
            f"Fetching company overview from Alpha Vantage for {symbol} (looking for pre-calculated ROC)..."
        )
        overview = await fetch_company_overview_alphavantage_async(symbol, session)
        if "return_on_capital" in overview and overview["return_on_capital"] is not None:
            result["return_on_capital"] = overview["return_on_capital"]
            logger.debug(
                f"{symbol}: Using pre-calculated ROC from Alpha Vantage: {result['return_on_capital']:.4f}"
            )
            # Remove from missing_fields so we don't try to calculate it
            missing_fields = [f for f in missing_fields if f != "return_on_capital"]

    # Check if we need balance sheet data
    needs_balance_sheet = any(
        field in missing_fields
        for field in [
            "net_working_capital",
            "net_fixed_assets",
            "current_assets",
            "current_liabilities",
        ]
    )

    if needs_balance_sheet:
        logger.debug(
            f"Fetching balance sheet from Alpha Vantage for {symbol} (missing: {missing_fields})"
        )
        balance_sheet = await fetch_balance_sheet_alphavantage_async(symbol, session)

        if balance_sheet:
            current_assets = balance_sheet.get("current_assets")
            current_liabilities = balance_sheet.get("current_liabilities")
            net_ppe = balance_sheet.get("net_ppe")

            # Calculate net_working_capital if we have the components
            if (
                "net_working_capital" in missing_fields
                and current_assets is not None
                and current_liabilities is not None
            ):
                result["net_working_capital"] = current_assets - current_liabilities
                logger.debug(
                    f"{symbol}: Fetched net_working_capital from Alpha Vantage: "
                    f"{result['net_working_capital']:,.0f}"
                )

            # Use net PPE as net_fixed_assets
            if "net_fixed_assets" in missing_fields and net_ppe is not None:
                result["net_fixed_assets"] = net_ppe
                logger.debug(
                    f"{symbol}: Fetched net_fixed_assets from Alpha Vantage: "
                    f"{result['net_fixed_assets']:,.0f}"
                )

    # Check if we need income statement data
    if "ebit" in missing_fields:
        logger.debug(f"Fetching income statement from Alpha Vantage for {symbol} (missing: EBIT)")
        income_statement = await fetch_income_statement_alphavantage_async(symbol, session)

        if income_statement and "ebit" in income_statement:
            result["ebit"] = income_statement["ebit"]
            logger.debug(f"{symbol}: Fetched EBIT from Alpha Vantage: {result['ebit']:,.0f}")

    return result


# Synchronous wrappers for backward compatibility (run in thread pool when called from async)
def fetch_missing_financial_data_alphavantage(
    symbol: str,
    missing_fields: list[str],
) -> dict[str, float | None]:
    """Synchronous wrapper for fetch_missing_financial_data_alphavantage_async.

    For use in synchronous contexts. In async contexts, use the async version directly.

    Args:
        symbol: Stock ticker symbol.
        missing_fields: List of fields that are missing.

    Returns:
        Dictionary with fetched data.
    """
    import warnings

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Cannot use asyncio.run() in a running event loop
            # Return empty dict and warn - caller should use async version
            warnings.warn(
                "Cannot use synchronous Alpha Vantage fetcher in async context. "
                "Use fetch_missing_financial_data_alphavantage_async instead. "
                "Returning empty dict.",
                UserWarning,
                stacklevel=2,
            )
            return {}
        else:
            return loop.run_until_complete(
                fetch_missing_financial_data_alphavantage_async(symbol, missing_fields)
            )
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(fetch_missing_financial_data_alphavantage_async(symbol, missing_fields))
