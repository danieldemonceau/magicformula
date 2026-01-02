"""Financial metrics calculations (EBIT, EV, ROC, etc.)."""

from src.calculations.exceptions import InvalidDataError, MissingDataError
from src.data.types import YFinanceInfoDict


def calculate_earnings_yield(ebit: float, enterprise_value: float) -> float:
    """Calculate Earnings Yield (EBIT / Enterprise Value).

    Args:
        ebit: Earnings Before Interest and Taxes.
        enterprise_value: Enterprise Value (Market Cap + Debt - Cash).

    Returns:
        Earnings Yield as a decimal (e.g., 0.15 for 15%).

    Raises:
        MissingDataError: If required data is missing.
        InvalidDataError: If enterprise_value is zero or negative.
    """
    if ebit is None:
        raise MissingDataError("EBIT is required for Earnings Yield calculation")
    if enterprise_value is None:
        raise MissingDataError("Enterprise Value is required for Earnings Yield calculation")
    if enterprise_value <= 0:
        raise InvalidDataError("Enterprise Value must be positive for Earnings Yield")

    return ebit / enterprise_value


def calculate_enterprise_value(
    market_cap: float,
    total_debt: float,
    cash_and_equivalents: float,
) -> float:
    """Calculate Enterprise Value.

    Formula: EV = Market Cap + Total Debt - Cash & Equivalents

    Args:
        market_cap: Market capitalization.
        total_debt: Total debt (short-term + long-term).
        cash_and_equivalents: Cash and cash equivalents.

    Returns:
        Enterprise Value.

    Raises:
        MissingDataError: If market_cap is missing.
    """
    if market_cap is None:
        raise MissingDataError("Market Cap is required for Enterprise Value calculation")

    total_debt = total_debt or 0.0
    cash_and_equivalents = cash_and_equivalents or 0.0

    return market_cap + total_debt - cash_and_equivalents


def calculate_return_on_capital(
    ebit: float,
    net_working_capital: float,
    net_fixed_assets: float,
) -> float:
    """Calculate Return on Capital (ROC).

    Formula: ROC = EBIT / (Net Working Capital + Net Fixed Assets)

    Args:
        ebit: Earnings Before Interest and Taxes.
        net_working_capital: Net Working Capital (Current Assets - Current Liabilities).
        net_fixed_assets: Net Fixed Assets (PP&E - Depreciation).

    Returns:
        Return on Capital as a decimal (e.g., 0.25 for 25%).

    Raises:
        MissingDataError: If required data is missing.
        InvalidDataError: If denominator is zero or negative.
    """
    if ebit is None:
        raise MissingDataError("EBIT is required for ROC calculation")

    net_working_capital = net_working_capital or 0.0
    net_fixed_assets = net_fixed_assets or 0.0

    capital_employed = net_working_capital + net_fixed_assets

    if capital_employed <= 0:
        raise InvalidDataError("Capital Employed must be positive for ROC calculation")

    return ebit / capital_employed


def calculate_acquirers_multiple(ebit: float, enterprise_value: float) -> float:
    """Calculate Acquirer's Multiple (EV / EBIT).

    Lower values indicate better value.

    Args:
        ebit: Earnings Before Interest and Taxes.
        enterprise_value: Enterprise Value.

    Returns:
        Acquirer's Multiple (EV/EBIT ratio).

    Raises:
        MissingDataError: If required data is missing.
        InvalidDataError: If EBIT is zero or negative.
    """
    if ebit is None:
        raise MissingDataError("EBIT is required for Acquirer's Multiple calculation")
    if enterprise_value is None:
        raise MissingDataError("Enterprise Value is required for Acquirer's Multiple calculation")
    if ebit <= 0:
        raise InvalidDataError("EBIT must be positive for Acquirer's Multiple")

    return enterprise_value / ebit


def extract_financial_data_from_yfinance(
    info: YFinanceInfoDict,
    symbol: str | None = None,
) -> dict[str, float | None | str | bool]:
    """Extract financial metrics from yfinance ticker.info dictionary.

    Args:
        info: yfinance ticker.info dictionary.
        symbol: Optional ticker symbol for logging.

    Returns:
        Dictionary with extracted financial metrics.
    """
    import logging

    logger = logging.getLogger("magicformula")

    market_cap_raw = info.get("marketCap")
    market_cap = (
        float(market_cap_raw)
        if market_cap_raw and isinstance(market_cap_raw, (int, float))
        else None
    )

    total_debt_raw = info.get("totalDebt")
    total_debt = (
        float(total_debt_raw)
        if total_debt_raw and isinstance(total_debt_raw, (int, float))
        else None
    )

    cash_raw = info.get("totalCash") or info.get("cashAndCashEquivalents") or info.get("cash")
    cash = float(cash_raw) if cash_raw and isinstance(cash_raw, (int, float)) else None

    ebit_raw = info.get("ebit")
    ebit = float(ebit_raw) if ebit_raw and isinstance(ebit_raw, (int, float)) else None

    ebitda_raw = info.get("ebitda")
    ebitda = float(ebitda_raw) if ebitda_raw and isinstance(ebitda_raw, (int, float)) else None

    operating_income_raw = info.get("operatingIncome")  # From income statement
    operating_income = (
        float(operating_income_raw)
        if operating_income_raw and isinstance(operating_income_raw, (int, float))
        else None
    )
    is_ebitda_fallback = False

    # Try multiple sources for EBIT (in order of preference):
    # 1. Direct EBIT from ticker.info
    # 2. Operating Income from income statement (usually = EBIT)
    # 3. Adjusted EBITDA as last resort
    if ebit is None:
        if operating_income is not None:
            # Operating Income is typically the same as EBIT
            ebit = operating_income
            if symbol:
                logger.debug(
                    f"{symbol}: Using Operating Income as EBIT: ${ebit:,.0f}",
                )
        elif ebitda is not None:
            # Fallback: EBITDA × multiplier (configurable, defaults to 0.75)
            # This is a rough approximation and may not be accurate for all companies
            from config.settings_pydantic import settings

            multiplier = getattr(settings, "ebitda_to_ebit_multiplier", 0.75)
            ebit = ebitda * multiplier
            is_ebitda_fallback = True
            if symbol:
                logger.debug(
                    f"{symbol}: EBIT unavailable, using EBITDA × {multiplier} = ${ebit:,.0f}",
                )

    current_assets_raw = info.get("totalCurrentAssets")
    current_assets = (
        float(current_assets_raw)
        if current_assets_raw and isinstance(current_assets_raw, (int, float))
        else None
    )

    current_liabilities_raw = info.get("totalCurrentLiabilities")
    current_liabilities = (
        float(current_liabilities_raw)
        if current_liabilities_raw and isinstance(current_liabilities_raw, (int, float))
        else None
    )

    net_working_capital = None
    if current_assets is not None and current_liabilities is not None:
        net_working_capital = current_assets - current_liabilities

    net_fixed_assets_raw = (
        info.get("netTangibleAssets")
        or info.get("propertyPlantEquipment")
        or info.get("ppeNet")
        or info.get("grossPPE")
    )
    net_fixed_assets = (
        float(net_fixed_assets_raw)
        if net_fixed_assets_raw and isinstance(net_fixed_assets_raw, (int, float))
        else None
    )

    if net_fixed_assets is None:
        total_assets_raw = info.get("totalAssets")
        total_assets = (
            float(total_assets_raw)
            if total_assets_raw and isinstance(total_assets_raw, (int, float))
            else None
        )
        if total_assets is not None and current_assets is not None:
            net_fixed_assets = total_assets - current_assets
            if symbol:
                logger.debug(
                    f"{symbol}: Using approximated net_fixed_assets "
                    f"(Total Assets - Current Assets = {net_fixed_assets:,.0f})"
                )

    # Ensure sector and industry are strings or None
    sector = info.get("sector")
    industry = info.get("industry")
    sector_str: str | None = str(sector) if sector and isinstance(sector, str) else None
    industry_str: str | None = str(industry) if industry and isinstance(industry, str) else None

    return {
        "market_cap": float(market_cap) if market_cap else None,
        "total_debt": float(total_debt) if total_debt else None,
        "cash": float(cash) if cash else None,
        "ebit": float(ebit) if ebit else None,
        "ebitda": float(ebitda) if ebitda else None,
        "is_ebitda_fallback": is_ebitda_fallback,
        "net_working_capital": float(net_working_capital) if net_working_capital else None,
        "net_fixed_assets": float(net_fixed_assets) if net_fixed_assets else None,
        "sector": sector_str,
        "industry": industry_str,
    }
