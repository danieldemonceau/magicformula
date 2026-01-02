"""Data quality assessment and outlier detection."""

import logging
from datetime import datetime, timedelta

from config.settings_pydantic import settings
from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus

logger = logging.getLogger("magicformula")

OUTLIER_THRESHOLDS = {
    "pe_ratio": settings.outlier_pe_ratio,
    "price_to_sales": settings.outlier_price_to_sales,
    "earnings_yield": settings.outlier_earnings_yield,
    "return_on_capital": settings.outlier_return_on_capital,
    "acquirers_multiple": settings.outlier_acquirers_multiple,
    "market_cap": settings.outlier_market_cap,
    "enterprise_value": settings.outlier_enterprise_value,
    "price": settings.outlier_price,
}

STALENESS_THRESHOLD_DAYS = settings.staleness_threshold_days


def detect_outliers(ticker_data: TickerData) -> list[str]:
    """Detect outliers in ticker data.

    Args:
        ticker_data: TickerData object to check.

    Returns:
        List of outlier field names detected.
    """
    outliers: list[str] = []

    if ticker_data.earnings_yield is not None and ticker_data.earnings_yield > 0:
        pe_ratio = 1.0 / ticker_data.earnings_yield
        if pe_ratio > OUTLIER_THRESHOLDS["pe_ratio"]:
            outliers.append("pe_ratio")
            logger.warning(f"{ticker_data.symbol}: Suspicious P/E ratio: {pe_ratio:.2f}")

    if (
        ticker_data.price_to_sales is not None
        and ticker_data.price_to_sales > OUTLIER_THRESHOLDS["price_to_sales"]
    ):
        outliers.append("price_to_sales")
        logger.warning(
            f"{ticker_data.symbol}: Suspicious P/S ratio: {ticker_data.price_to_sales:.2f}"
        )

    if (
        ticker_data.earnings_yield is not None
        and ticker_data.earnings_yield > OUTLIER_THRESHOLDS["earnings_yield"]
    ):
        outliers.append("earnings_yield")
        logger.warning(
            f"{ticker_data.symbol}: Suspicious Earnings Yield: {ticker_data.earnings_yield:.2%}"
        )

    if (
        ticker_data.return_on_capital is not None
        and ticker_data.return_on_capital > OUTLIER_THRESHOLDS["return_on_capital"]
    ):
        outliers.append("return_on_capital")
        logger.warning(f"{ticker_data.symbol}: Suspicious ROC: {ticker_data.return_on_capital:.2%}")

    if (
        ticker_data.acquirers_multiple is not None
        and ticker_data.acquirers_multiple > OUTLIER_THRESHOLDS["acquirers_multiple"]
    ):
        outliers.append("acquirers_multiple")
        logger.warning(
            f"{ticker_data.symbol}: Suspicious Acquirer's Multiple: {ticker_data.acquirers_multiple:.2f}"
        )

    if (
        ticker_data.market_cap is not None
        and ticker_data.market_cap > OUTLIER_THRESHOLDS["market_cap"]
    ):
        outliers.append("market_cap")
        logger.warning(
            f"{ticker_data.symbol}: Suspicious Market Cap: ${ticker_data.market_cap:,.0f}"
        )

    if (
        ticker_data.enterprise_value is not None
        and ticker_data.enterprise_value > OUTLIER_THRESHOLDS["enterprise_value"]
    ):
        outliers.append("enterprise_value")
        logger.warning(
            f"{ticker_data.symbol}: Suspicious Enterprise Value: ${ticker_data.enterprise_value:,.0f}"
        )

    if ticker_data.price is not None and ticker_data.price > OUTLIER_THRESHOLDS["price"]:
        outliers.append("price")
        logger.warning(f"{ticker_data.symbol}: Suspicious Price: ${ticker_data.price:,.2f}")

    return outliers


def check_staleness(ticker_data: TickerData) -> bool:
    """Check if ticker data is stale.

    Args:
        ticker_data: TickerData object to check.

    Returns:
        True if data is stale, False otherwise.
    """
    if ticker_data.data_timestamp is None:
        # If timestamp is None but we have financial data, assume it's fresh (just fetched)
        # This handles CSV input where timestamp might not be set initially
        has_financial_data = (
            ticker_data.market_cap is not None
            or ticker_data.ebit is not None
            or ticker_data.enterprise_value is not None
        )
        # Return True (stale) if no timestamp AND no financial data
        return not has_financial_data

    age = datetime.now() - ticker_data.data_timestamp
    is_stale = age > timedelta(days=STALENESS_THRESHOLD_DAYS)

    if is_stale:
        logger.warning(f"{ticker_data.symbol}: Data is stale (age: {age.days} days)")

    return is_stale


def calculate_quality_score(ticker_data: TickerData) -> float:
    """Calculate data quality score (0-1, higher is better).

    Includes all fields needed for Magic Formula and Acquirer's Multiple strategies.

    Args:
        ticker_data: TickerData object to score.

    Returns:
        Quality score between 0 and 1.
    """
    score = 1.0

    # Core fields required for calculations
    critical_fields = [
        "price",
        "market_cap",
        "ebit",
        "enterprise_value",
    ]
    missing_critical = sum(1 for field in critical_fields if getattr(ticker_data, field) is None)
    score -= missing_critical * 0.2

    # NWC and NFA are only critical if we don't have pre-calculated ROC
    # If ROC is available (from any source), NWC/NFA are not required
    if ticker_data.return_on_capital is None:
        # ROC not available - need NWC and NFA to calculate it
        if ticker_data.net_working_capital is None:
            score -= 0.1
        if ticker_data.net_fixed_assets is None:
            score -= 0.1
    # If ROC is available, NWC/NFA missing is not penalized (they may have come from AV)

    outliers = detect_outliers(ticker_data)
    score -= len(outliers) * 0.1

    if check_staleness(ticker_data):
        score -= 0.2

    if ticker_data.status != TickerStatus.ACTIVE:
        score -= 0.3

    return max(0.0, min(1.0, score))


def assess_data_quality(ticker_data: TickerData) -> TickerData:
    """Assess and update data quality for a ticker.

    Creates a new TickerData object with updated quality metrics (immutable approach).

    Args:
        ticker_data: TickerData object to assess.

    Returns:
        New TickerData with quality score and status updated.
    """
    outliers = detect_outliers(ticker_data)
    is_stale = check_staleness(ticker_data)

    # Only mark as DELISTED if price is missing AND we have no other financial data
    # This prevents marking CSV-sourced tickers as DELISTED when they have valid market_cap/EBIT
    has_financial_data = (
        ticker_data.market_cap is not None
        or ticker_data.ebit is not None
        or ticker_data.enterprise_value is not None
    )

    # Determine new status
    new_status = ticker_data.status
    if (ticker_data.price is None or ticker_data.price <= 0) and not has_financial_data:
        # Only mark as DELISTED if we have no financial data at all
        if ticker_data.status == TickerStatus.ACTIVE:
            new_status = TickerStatus.DATA_UNAVAILABLE
        else:
            new_status = TickerStatus.DELISTED
    elif is_stale:
        new_status = TickerStatus.STALE
    elif len(outliers) > 2:
        new_status = TickerStatus.DATA_UNAVAILABLE

    # Set timestamp BEFORE calculating quality score (so staleness check uses it)
    new_timestamp = ticker_data.data_timestamp
    if new_timestamp is None:
        new_timestamp = datetime.now()

    # Create temporary TickerData for quality score calculation
    # Use model_copy to avoid duplicate keyword arguments
    temp_ticker = ticker_data.model_copy(
        update={
            "status": new_status,
            "data_timestamp": new_timestamp,
        }
    )
    quality_score = calculate_quality_score(temp_ticker)

    # Return new TickerData with all updates using model_copy
    return ticker_data.model_copy(
        update={
            "status": new_status,
            "data_timestamp": new_timestamp,
            "quality_score": quality_score,
        }
    )
