"""Data validation utilities."""

from collections.abc import Callable

from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus

EXCLUDED_SECTORS = {
    "Financial Services",
    "Financial",
    "Banks",
    "Insurance",
    "Real Estate",
    "REIT",
    "Real Estate Investment Trust",
}


def is_financial_sector(ticker_data: TickerData) -> bool:
    """Check if ticker is in a financial sector.

    Args:
        ticker_data: TickerData object to check.

    Returns:
        True if ticker is in excluded financial sector.
    """
    if not ticker_data.sector:
        return False

    sector_upper = ticker_data.sector.upper()
    return any(excluded.upper() in sector_upper for excluded in EXCLUDED_SECTORS)


def validate_ticker_data_for_magic_formula(ticker_data: TickerData) -> bool:
    """Validate that ticker data has required fields for Magic Formula.

    Args:
        ticker_data: TickerData object to validate.

    Returns:
        True if valid, False otherwise.
    """
    if ticker_data.status != TickerStatus.ACTIVE:
        return False

    if is_financial_sector(ticker_data):
        return False

    if ticker_data.earnings_yield is None or ticker_data.earnings_yield <= 0:
        return False

    if ticker_data.return_on_capital is None or ticker_data.return_on_capital <= 0:
        return False

    return not (ticker_data.quality_score is not None and ticker_data.quality_score < 0.5)


def validate_ticker_data_for_acquirers_multiple(ticker_data: TickerData) -> bool:
    """Validate that ticker data has required fields for Acquirer's Multiple.

    Args:
        ticker_data: TickerData object to validate.

    Returns:
        True if valid, False otherwise.
    """
    if ticker_data.status != TickerStatus.ACTIVE:
        return False

    if is_financial_sector(ticker_data):
        return False

    if ticker_data.acquirers_multiple is None or ticker_data.acquirers_multiple <= 0:
        return False

    return not (ticker_data.quality_score is not None and ticker_data.quality_score < 0.5)


def filter_valid_tickers(
    ticker_data_list: list[TickerData],
    validation_func: Callable[[TickerData], bool],
) -> list[TickerData]:
    """Filter tickers based on validation function.

    Args:
        ticker_data_list: List of TickerData objects.
        validation_func: Validation function that returns bool.

    Returns:
        Filtered list of valid TickerData objects.
    """
    return [
        ticker
        for ticker in ticker_data_list
        if validation_func(ticker) and ticker.status == TickerStatus.ACTIVE
    ]
