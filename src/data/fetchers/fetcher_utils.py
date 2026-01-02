"""Shared utility functions for fetching price and fundamental data."""

import logging
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

if TYPE_CHECKING:
    from typing import TypeAlias

    import aiohttp

    # Type alias for aiohttp.ClientSession when available
    ClientSessionType: TypeAlias = aiohttp.ClientSession
else:
    try:
        import aiohttp

        # Runtime assignment (mypy uses TYPE_CHECKING version)
        ClientSessionType = aiohttp.ClientSession  # type: ignore[assignment, misc]
    except ImportError:
        # Create a dummy class for type hints when aiohttp is not available
        class _DummyClientSession:
            pass

        aiohttp = type("aiohttp", (), {"ClientSession": _DummyClientSession})()  # type: ignore[assignment, unused-ignore]
        ClientSessionType = _DummyClientSession  # type: ignore[assignment, misc]

from src.data.models.ticker_data import TickerData
from src.data.quality import assess_data_quality
from src.data.types import FinancialDataDict, YFinanceInfoDict
from src.utils.date_utils import get_business_day_months_ago, get_last_business_day

logger = logging.getLogger("magicformula")


def fetch_price_data_shared(
    symbol: str,
    last_business_day: date | None = None,
) -> dict[str, float | None]:
    """Shared function to fetch price data for a ticker.

    Used by both sync and async fetchers to reduce duplication.

    Args:
        symbol: Ticker symbol.
        last_business_day: Last business day to use. If None, uses today.

    Returns:
        Dictionary with price data.
    """
    if last_business_day is None:
        last_business_day = get_last_business_day()

    try:
        # Fetch 1 year of historical data to calculate price indices
        start_date = get_business_day_months_ago(12, last_business_day)
        end_date = last_business_day + timedelta(
            days=1
        )  # Fetch up to and including last_business_day

        hist = yf.download(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
        )

        if hist.empty:
            logger.warning(f"{symbol}: No historical data found for price fetching.")
            return {"price": None, "price_index_6month": None, "price_index_12month": None}

        return _extract_price_data_from_hist(hist, symbol, last_business_day)

    except Exception as e:
        logger.error(f"Error fetching price data for {symbol}: {e}")
        return {"price": None, "price_index_6month": None, "price_index_12month": None}


def fetch_fundamental_data_shared(symbol: str) -> YFinanceInfoDict:
    """Shared function to fetch fundamental data for a ticker.

    Used by both sync and async fetchers to reduce duplication.
    Fetches both ticker.info AND ticker.balance_sheet for complete data.

    Args:
        symbol: Ticker symbol.

    Returns:
        Dictionary with fundamental data including balance sheet items.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        book_to_market = None
        if info.get("priceToBook") and info["priceToBook"] > 0:
            book_to_market = 1.0 / float(info["priceToBook"])

        free_cash_flow_yield = None
        if info.get("freeCashflow") and info.get("marketCap") and info["marketCap"] > 0:
            free_cash_flow_yield = float(info["freeCashflow"]) / float(info["marketCap"])

        price_to_sales = (
            float(info["priceToSalesTrailing12Months"])
            if info.get("priceToSalesTrailing12Months")
            else None
        )

        balance_sheet_data = _extract_balance_sheet_data(ticker)
        income_statement_data = _extract_income_statement_data(ticker)

        return {
            **info,
            "book_to_market": book_to_market,
            "free_cash_flow_yield": free_cash_flow_yield,
            "price_to_sales": price_to_sales,
            "totalCurrentAssets": balance_sheet_data.get("current_assets"),
            "totalCurrentLiabilities": balance_sheet_data.get("current_liabilities"),
            "propertyPlantEquipment": balance_sheet_data.get("net_ppe"),
            "totalAssets": balance_sheet_data.get("total_assets"),
            "operatingIncome": income_statement_data.get("operating_income"),
        }

    except Exception as e:
        logger.error(f"Error fetching fundamental data for {symbol}: {e}")
        return {}


def _extract_balance_sheet_data(ticker: yf.Ticker) -> dict[str, float | None]:
    """Extract balance sheet data needed for ROC calculation.

    Args:
        ticker: yfinance Ticker object.

    Returns:
        Dictionary with balance sheet items.
    """
    try:
        bs = ticker.balance_sheet
        if bs.empty:
            return {}

        latest = bs.iloc[:, 0]

        current_assets = latest.get("Current Assets") or latest.get("Total Current Assets")
        current_liabilities = latest.get("Current Liabilities") or latest.get(
            "Total Current Liabilities"
        )
        net_ppe = (
            latest.get("Net PPE")
            or latest.get("Property Plant And Equipment Net")
            or latest.get("Net Property, Plant and Equipment")
        )
        total_assets = latest.get("Total Assets")

        return {
            "current_assets": float(current_assets) if pd.notna(current_assets) else None,
            "current_liabilities": float(current_liabilities)
            if pd.notna(current_liabilities)
            else None,
            "net_ppe": float(net_ppe) if pd.notna(net_ppe) else None,
            "total_assets": float(total_assets) if pd.notna(total_assets) else None,
        }

    except Exception as e:
        logger.debug(f"Could not extract balance sheet data: {e}")
        return {}


def _extract_income_statement_data(ticker: yf.Ticker) -> dict[str, float | None]:
    """Extract income statement data needed for EBIT calculation.

    Args:
        ticker: yfinance Ticker object.

    Returns:
        Dictionary with income statement items.
    """
    try:
        financials = ticker.financials
        if financials.empty:
            return {}

        latest = financials.iloc[:, 0]

        # Operating Income is typically the same as EBIT
        operating_income = (
            latest.get("Operating Income")
            or latest.get("Operating Income (Loss)")
            or latest.get("Total Operating Income")
            or latest.get("EBIT")  # Sometimes directly listed as EBIT
        )

        return {
            "operating_income": float(operating_income) if pd.notna(operating_income) else None,
        }

    except Exception as e:
        logger.debug(f"Could not extract income statement data: {e}")
        return {}


def _extract_float_or_none(value: str | int | float | bool | None) -> float | None:
    """Extract float value from YFinanceInfoDict value, returning None if not numeric.

    Args:
        value: Value from YFinanceInfoDict (can be str, int, float, bool, or None).

    Returns:
        float | None: Float value if numeric, None otherwise.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_ticker_data(
    symbol: str,
    price_data: dict[str, float | None],
    financial_data: FinancialDataDict,
    fundamental_data: YFinanceInfoDict,
    metrics: dict[str, float | None],
) -> TickerData:
    """Build TickerData from extracted data components.

    This shared function eliminates duplication across all fetcher implementations.

    Args:
        symbol: Ticker symbol.
        price_data: Dictionary with price data from fetch_price_data_shared.
        financial_data: Dictionary from extract_financial_data_from_yfinance.
        fundamental_data: Dictionary from fetch_fundamental_data_shared.
        metrics: Dictionary from calculate_metrics_from_financial_data.

    Returns:
        TickerData object with quality assessment applied.
    """
    # Ensure sector and industry are strings
    sector_val: str | None = (
        str(financial_data.get("sector"))
        if financial_data.get("sector") and isinstance(financial_data.get("sector"), str)
        else None
    )
    industry_val: str | None = (
        str(financial_data.get("industry"))
        if financial_data.get("industry") and isinstance(financial_data.get("industry"), str)
        else None
    )

    # Ensure numeric fields are floats
    market_cap_raw = financial_data.get("market_cap")
    market_cap_val: float | None = (
        float(market_cap_raw)
        if market_cap_raw is not None and isinstance(market_cap_raw, (int, float))
        else None
    )
    total_debt_raw = financial_data.get("total_debt")
    total_debt_val: float | None = (
        float(total_debt_raw)
        if total_debt_raw is not None and isinstance(total_debt_raw, (int, float))
        else None
    )
    cash_raw = financial_data.get("cash")
    cash_val: float | None = (
        float(cash_raw) if cash_raw is not None and isinstance(cash_raw, (int, float)) else None
    )
    ebit_raw = financial_data.get("ebit")
    ebit_val: float | None = (
        float(ebit_raw) if ebit_raw is not None and isinstance(ebit_raw, (int, float)) else None
    )
    nwc_raw = financial_data.get("net_working_capital")
    nwc_val: float | None = (
        float(nwc_raw) if nwc_raw is not None and isinstance(nwc_raw, (int, float)) else None
    )
    nfa_raw = financial_data.get("net_fixed_assets")
    nfa_val: float | None = (
        float(nfa_raw) if nfa_raw is not None and isinstance(nfa_raw, (int, float)) else None
    )

    ticker_data = TickerData(
        symbol=symbol,
        sector=sector_val,
        industry=industry_val,
        price=price_data.get("price"),
        market_cap=market_cap_val,
        total_debt=total_debt_val,
        cash=cash_val,
        ebit=ebit_val,
        net_working_capital=nwc_val,
        net_fixed_assets=nfa_val,
        enterprise_value=metrics.get("enterprise_value"),
        earnings_yield=metrics.get("earnings_yield"),
        return_on_capital=metrics.get("return_on_capital"),
        acquirers_multiple=metrics.get("acquirers_multiple"),
        price_index_6month=price_data.get("price_index_6month"),
        price_index_12month=price_data.get("price_index_12month"),
        book_to_market=_extract_float_or_none(fundamental_data.get("book_to_market")),
        free_cash_flow_yield=_extract_float_or_none(fundamental_data.get("free_cash_flow_yield")),
        price_to_sales=_extract_float_or_none(fundamental_data.get("price_to_sales")),
        data_timestamp=datetime.now(),
        quality_score=None,  # Will be set by assess_data_quality
    )

    return assess_data_quality(ticker_data)


async def calculate_metrics_from_financial_data_async(
    financial_data: FinancialDataDict,
    symbol: str,
    session: "ClientSessionType | None" = None,
) -> tuple[dict[str, float | None], FinancialDataDict]:
    """Calculate financial metrics from extracted financial data (async version).

    This async version is used by async fetchers to avoid blocking the event loop.

    Args:
        financial_data: Dictionary with financial data from extract_financial_data_from_yfinance.
        symbol: Ticker symbol (for logging).
        session: Optional aiohttp session for connection pooling.

    Returns:
        Tuple of (metrics dict, updated financial_data dict).
        The updated financial_data includes Alpha Vantage data (NWC, NFA).
    """
    from src.calculations.exceptions import InvalidDataError
    from src.calculations.financial_metrics import (
        calculate_acquirers_multiple,
        calculate_earnings_yield,
        calculate_enterprise_value,
        calculate_return_on_capital,
    )

    enterprise_value = None
    earnings_yield = None
    return_on_capital_raw = financial_data.get("return_on_capital")
    return_on_capital: float | None = None
    if return_on_capital_raw is not None and isinstance(return_on_capital_raw, (int, float)):
        return_on_capital = float(return_on_capital_raw)
        if symbol:
            logger.debug(
                f"{symbol}: Using pre-calculated ROC from yfinance: {return_on_capital:.4f}"
            )
    acquirers_multiple = None

    market_cap_val = financial_data.get("market_cap")
    if market_cap_val and isinstance(market_cap_val, (int, float)):
        market_cap_float = float(market_cap_val)
        total_debt_val = financial_data.get("total_debt")
        total_debt_float = (
            float(total_debt_val)
            if total_debt_val and isinstance(total_debt_val, (int, float))
            else 0.0
        )
        cash_val = financial_data.get("cash")
        cash_float = float(cash_val) if cash_val and isinstance(cash_val, (int, float)) else 0.0

        enterprise_value = calculate_enterprise_value(
            market_cap_float,
            total_debt_float,
            cash_float,
        )

        ebit_val = financial_data.get("ebit")
        if ebit_val and isinstance(ebit_val, (int, float)) and enterprise_value:
            ebit_float = float(ebit_val)
            try:
                earnings_yield = calculate_earnings_yield(ebit_float, enterprise_value)
            except (ValueError, InvalidDataError) as e:
                logger.debug(f"{symbol}: Could not calculate earnings_yield: {e}")

            try:
                acquirers_multiple = calculate_acquirers_multiple(ebit_float, enterprise_value)
            except (ValueError, InvalidDataError) as e:
                logger.debug(f"{symbol}: Could not calculate acquirers_multiple: {e}")

            nwc_val = financial_data.get("net_working_capital")
            nfa_val = financial_data.get("net_fixed_assets")

            # Track which fields we've already attempted from Alpha Vantage to avoid duplicate calls
            av_attempted_fields: set[str] = set()
            updated_financial_data = {**financial_data}

            # If we don't have pre-calculated ROC, try to get it from Alpha Vantage first
            if return_on_capital is None and ebit_float > 0:
                try:
                    from src.data.fetchers.alphavantage_fetcher import (
                        fetch_missing_financial_data_alphavantage_async,
                    )

                    av_data = await fetch_missing_financial_data_alphavantage_async(
                        symbol,
                        ["return_on_capital"],
                        session,
                    )
                    av_attempted_fields.add("return_on_capital")

                    if (
                        "return_on_capital" in av_data
                        and av_data["return_on_capital"] is not None
                        and isinstance(av_data["return_on_capital"], (int, float))
                    ):
                        return_on_capital = float(av_data["return_on_capital"])
                        logger.info(
                            f"{symbol}: Using pre-calculated ROC from Alpha Vantage: {return_on_capital:.4f}"
                        )
                except Exception as e:
                    logger.debug(f"{symbol}: Could not fetch ROC from Alpha Vantage: {e}")

            # If still no ROC, calculate it ourselves using NWC and NFA
            if return_on_capital is None and ebit_float > 0:
                # Try Alpha Vantage fallback if NWC/NFA data is missing
                if nwc_val is None or nfa_val is None:
                    missing_fields = []
                    if nwc_val is None:
                        missing_fields.append("net_working_capital")
                    if nfa_val is None:
                        missing_fields.append("net_fixed_assets")
                    # Only try ROC again if we haven't already attempted it
                    if return_on_capital is None and "return_on_capital" not in av_attempted_fields:
                        missing_fields.append("return_on_capital")

                    if missing_fields:
                        try:
                            from src.data.fetchers.alphavantage_fetcher import (
                                fetch_missing_financial_data_alphavantage_async,
                            )

                            av_data = await fetch_missing_financial_data_alphavantage_async(
                                symbol,
                                missing_fields,
                                session,
                            )
                            av_attempted_fields.update(missing_fields)

                            # Update with Alpha Vantage data
                            if (
                                "net_working_capital" in av_data
                                and av_data["net_working_capital"] is not None
                            ):
                                nwc_val = av_data["net_working_capital"]
                                updated_financial_data["net_working_capital"] = nwc_val
                            if (
                                "net_fixed_assets" in av_data
                                and av_data["net_fixed_assets"] is not None
                            ):
                                nfa_val = av_data["net_fixed_assets"]
                                updated_financial_data["net_fixed_assets"] = nfa_val
                            if (
                                "return_on_capital" in av_data
                                and av_data["return_on_capital"] is not None
                            ):
                                return_on_capital = float(av_data["return_on_capital"])
                                logger.info(
                                    f"{symbol}: Using pre-calculated ROC from Alpha Vantage: {return_on_capital:.4f}"
                                )
                        except Exception as e:
                            logger.debug(f"{symbol}: Alpha Vantage fallback failed: {e}")

                if (nwc_val is not None and isinstance(nwc_val, (int, float))) or (
                    nfa_val is not None and isinstance(nfa_val, (int, float))
                ):
                    nwc_float = (
                        float(nwc_val)
                        if nwc_val is not None and isinstance(nwc_val, (int, float))
                        else 0.0
                    )
                    nfa_float = (
                        float(nfa_val)
                        if nfa_val is not None and isinstance(nfa_val, (int, float))
                        else 0.0
                    )
                    try:
                        return_on_capital = calculate_return_on_capital(
                            ebit_float,
                            nwc_float,
                            nfa_float,
                        )
                        logger.debug(f"{symbol}: Calculated ROC: {return_on_capital:.4f}")
                    except (ValueError, InvalidDataError) as e:
                        logger.debug(f"{symbol}: Could not calculate return_on_capital: {e}")
    else:
        logger.debug(f"{symbol}: Missing market_cap, cannot calculate EV-based metrics")
        if "updated_financial_data" not in locals():
            updated_financial_data = {**financial_data}

    metrics_dict = {
        "enterprise_value": enterprise_value,
        "earnings_yield": earnings_yield,
        "return_on_capital": return_on_capital,
        "acquirers_multiple": acquirers_multiple,
    }

    # Ensure updated_financial_data is always defined
    if "updated_financial_data" not in locals():
        updated_financial_data = {**financial_data}

    return metrics_dict, updated_financial_data


def calculate_metrics_from_financial_data(
    financial_data: FinancialDataDict,
    symbol: str,
) -> tuple[dict[str, float | None], FinancialDataDict]:
    """Calculate financial metrics from extracted financial data.

    This shared function eliminates code duplication across fetchers.
    First checks for pre-calculated ROC from yfinance, then tries Alpha Vantage,
    then calculates if needed.

    Args:
        financial_data: Dictionary with financial data from extract_financial_data_from_yfinance.
        symbol: Ticker symbol (for logging).

    Returns:
        Tuple of (metrics dict, updated financial_data dict).
        The updated financial_data includes Alpha Vantage data (NWC, NFA).
    """
    from src.calculations.exceptions import InvalidDataError
    from src.calculations.financial_metrics import (
        calculate_acquirers_multiple,
        calculate_earnings_yield,
        calculate_enterprise_value,
        calculate_return_on_capital,
    )

    enterprise_value = None
    earnings_yield = None
    # Check if we already have pre-calculated ROC from yfinance
    return_on_capital_raw = financial_data.get("return_on_capital")
    return_on_capital: float | None = None
    if return_on_capital_raw is not None and isinstance(return_on_capital_raw, (int, float)):
        return_on_capital = float(return_on_capital_raw)
        if symbol:
            logger.debug(
                f"{symbol}: Using pre-calculated ROC from yfinance: {return_on_capital:.4f}"
            )
    acquirers_multiple = None

    market_cap_val = financial_data.get("market_cap")
    if market_cap_val and isinstance(market_cap_val, (int, float)):
        market_cap_float = float(market_cap_val)
        total_debt_val = financial_data.get("total_debt")
        total_debt_float = (
            float(total_debt_val)
            if total_debt_val and isinstance(total_debt_val, (int, float))
            else 0.0
        )
        cash_val = financial_data.get("cash")
        cash_float = float(cash_val) if cash_val and isinstance(cash_val, (int, float)) else 0.0

        enterprise_value = calculate_enterprise_value(
            market_cap_float,
            total_debt_float,
            cash_float,
        )

        ebit_val = financial_data.get("ebit")
        if ebit_val and isinstance(ebit_val, (int, float)) and enterprise_value:
            ebit_float = float(ebit_val)
            try:
                earnings_yield = calculate_earnings_yield(ebit_float, enterprise_value)
            except (ValueError, InvalidDataError) as e:
                logger.debug(f"{symbol}: Could not calculate earnings_yield: {e}")

            try:
                acquirers_multiple = calculate_acquirers_multiple(ebit_float, enterprise_value)
            except (ValueError, InvalidDataError) as e:
                logger.debug(f"{symbol}: Could not calculate acquirers_multiple: {e}")

            nwc_val = financial_data.get("net_working_capital")
            nfa_val = financial_data.get("net_fixed_assets")

            # Track which fields we've already attempted from Alpha Vantage to avoid duplicate calls
            av_attempted_fields: set[str] = set()
            updated_financial_data = {**financial_data}

            # Try Alpha Vantage fallback if data is missing (synchronous version)
            # Note: This is for sync fetchers only. Async fetchers use calculate_metrics_from_financial_data_async
            if (nwc_val is None or nfa_val is None) and ebit_float > 0:
                missing_fields = []
                if nwc_val is None:
                    missing_fields.append("net_working_capital")
                if nfa_val is None:
                    missing_fields.append("net_fixed_assets")
                # Also try to get ROC if we still don't have it
                if return_on_capital is None:
                    missing_fields.append("return_on_capital")

                if missing_fields:
                    try:
                        from src.data.fetchers.alphavantage_fetcher import (
                            fetch_missing_financial_data_alphavantage,
                        )

                        av_data = fetch_missing_financial_data_alphavantage(
                            symbol,
                            missing_fields,
                        )
                        av_attempted_fields.update(missing_fields)

                        # Update local variables and updated_financial_data dict
                        if (
                            "net_working_capital" in av_data
                            and av_data["net_working_capital"] is not None
                        ):
                            nwc_val = av_data["net_working_capital"]
                            updated_financial_data["net_working_capital"] = nwc_val
                        if (
                            "net_fixed_assets" in av_data
                            and av_data["net_fixed_assets"] is not None
                        ):
                            nfa_val = av_data["net_fixed_assets"]
                            updated_financial_data["net_fixed_assets"] = nfa_val
                        if (
                            "return_on_capital" in av_data
                            and av_data["return_on_capital"] is not None
                            and isinstance(av_data["return_on_capital"], (int, float))
                        ):
                            return_on_capital = float(av_data["return_on_capital"])
                            logger.info(
                                f"{symbol}: Using pre-calculated ROC from Alpha Vantage: {return_on_capital:.4f}"
                            )
                    except Exception as e:
                        logger.debug(f"{symbol}: Alpha Vantage fallback failed: {e}")

            if (nwc_val is not None and isinstance(nwc_val, (int, float))) or (
                nfa_val is not None and isinstance(nfa_val, (int, float))
            ):
                nwc_float = (
                    float(nwc_val)
                    if nwc_val is not None and isinstance(nwc_val, (int, float))
                    else 0.0
                )
                nfa_float = (
                    float(nfa_val)
                    if nfa_val is not None and isinstance(nfa_val, (int, float))
                    else 0.0
                )
                try:
                    return_on_capital = calculate_return_on_capital(
                        ebit_float,
                        nwc_float,
                        nfa_float,
                    )
                except (ValueError, InvalidDataError) as e:
                    logger.debug(f"{symbol}: Could not calculate return_on_capital: {e}")
    else:
        logger.debug(f"{symbol}: Missing market_cap, cannot calculate EV-based metrics")
        updated_financial_data = {**financial_data}

    metrics_dict = {
        "enterprise_value": enterprise_value,
        "earnings_yield": earnings_yield,
        "return_on_capital": return_on_capital,
        "acquirers_multiple": acquirers_multiple,
    }

    return metrics_dict, updated_financial_data


def fetch_price_data_batch(
    symbols: list[str],
    last_business_day: date | None = None,
) -> dict[str, dict[str, float | None]]:
    """Batch fetch price data for multiple symbols using yfinance.download().

    This is more efficient than fetching one-by-one as it makes a single API call.

    Args:
        symbols: List of ticker symbols (case-insensitive, normalized to uppercase).
        last_business_day: Last business day to use. If None, uses today.

    Returns:
        Dictionary mapping symbol (original case) to price data dictionary.
    """
    if last_business_day is None:
        last_business_day = get_last_business_day()

    if not symbols:
        return {}

    try:
        start_date = get_business_day_months_ago(12, last_business_day)
        end_date = last_business_day + timedelta(days=1)

        hist = yf.download(
            symbols,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
        )

        if hist.empty:
            return {
                symbol: {"price": None, "price_index_6month": None, "price_index_12month": None}
                for symbol in symbols
            }

        results: dict[str, dict[str, float | None]] = {}

        if len(symbols) == 1:
            symbol = symbols[0]
            price_data = _extract_price_data_from_hist(hist, symbol, last_business_day)
            results[symbol] = price_data
        else:
            # Normalize symbols to uppercase for lookup (yfinance returns uppercase)
            symbol_upper_map = {s.upper(): s for s in symbols}

            # Check if DataFrame has MultiIndex columns
            has_multiindex = isinstance(hist.columns, pd.MultiIndex)

            for symbol_upper, symbol_original in symbol_upper_map.items():
                # Check if symbol exists in DataFrame
                if has_multiindex:
                    # For MultiIndex, check level 1 (symbol names)
                    if symbol_upper not in hist.columns.levels[1]:
                        results[symbol_original] = {
                            "price": None,
                            "price_index_6month": None,
                            "price_index_12month": None,
                        }
                        continue
                else:
                    # Single symbol DataFrame - should not happen with multiple symbols
                    # but handle gracefully
                    results[symbol_original] = {
                        "price": None,
                        "price_index_6month": None,
                        "price_index_12month": None,
                    }
                    continue

                try:
                    # Extract data using uppercase symbol
                    symbol_hist = hist.xs("Close", level=0, axis=1)[symbol_upper]
                    if isinstance(symbol_hist, pd.Series):
                        symbol_df = pd.DataFrame({symbol_upper: symbol_hist})
                        symbol_df.columns = pd.MultiIndex.from_product([["Close"], [symbol_upper]])
                    else:
                        symbol_df = hist.xs(symbol_upper, level=1, axis=1)

                    price_data = _extract_price_data_from_hist(
                        symbol_df,
                        symbol_original,  # Use original case for logging
                        last_business_day,
                    )
                    results[symbol_original] = price_data  # Store with original case
                except (KeyError, IndexError) as e:
                    logger.warning(
                        f"Could not extract price data for {symbol_original} from batch: {e}"
                    )
                    results[symbol_original] = {
                        "price": None,
                        "price_index_6month": None,
                        "price_index_12month": None,
                    }

        return results

    except Exception as e:
        logger.error(f"Error in batch price data fetch: {e}")
        return {symbol: fetch_price_data_shared(symbol, last_business_day) for symbol in symbols}


def _extract_price_data_from_hist(
    hist: pd.DataFrame,
    symbol: str,
    last_business_day: date,
) -> dict[str, float | None]:
    """Extract price data from a historical DataFrame.

    Args:
        hist: Historical price DataFrame.
        symbol: Ticker symbol (for logging).
        last_business_day: Last business day.

    Returns:
        Dictionary with price data.
    """
    try:
        if hist.empty:
            return {"price": None, "price_index_6month": None, "price_index_12month": None}

        if isinstance(hist.columns, pd.MultiIndex):
            close_col = hist["Close"]
        else:
            close_col = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]

        last_price_row = close_col.loc[close_col.index.date == last_business_day]
        if not last_price_row.empty:
            price = float(last_price_row.iloc[0].item())
        else:
            price = float(close_col.iloc[-1].item())

        six_months_ago = get_business_day_months_ago(6, last_business_day)
        twelve_months_ago = get_business_day_months_ago(12, last_business_day)

        price_6m_row = close_col.loc[close_col.index.date == six_months_ago]
        price_6m = float(price_6m_row.iloc[0].item()) if not price_6m_row.empty else None

        price_12m_row = close_col.loc[close_col.index.date == twelve_months_ago]
        if not price_12m_row.empty:
            price_12m = float(price_12m_row.iloc[0].item())
        else:
            price_12m = float(close_col.iloc[0].item())

        price_index_6month = price / price_6m if price_6m and price_6m > 0 else None
        price_index_12month = price / price_12m if price_12m and price_12m > 0 else None

        return {
            "price": price,
            "price_index_6month": price_index_6month,
            "price_index_12month": price_index_12month,
        }

    except Exception as e:
        logger.error(f"Error extracting price data for {symbol}: {e}")
        return {"price": None, "price_index_6month": None, "price_index_12month": None}
