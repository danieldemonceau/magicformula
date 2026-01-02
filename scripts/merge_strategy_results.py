#!/usr/bin/env python3
"""Merge Magic Formula and Acquirer's Multiple results into a single CSV.

This script:
1. Loads both MF and AM input CSVs
2. Calculates MF metrics for ALL tickers (both MF and AM inputs)
3. Calculates AM metrics for ALL tickers (both MF and AM inputs)
4. Merges intelligently with column deduplication
5. Labels calculated vs input metrics appropriately
"""

import argparse
import asyncio
import csv
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.calculations.financial_metrics import (
    calculate_acquirers_multiple,
    calculate_earnings_yield,
    calculate_enterprise_value,
    calculate_return_on_capital,
)
from src.data.fetchers.async_fetcher import AsyncYFinanceFetcher
from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus
from src.strategies.acquirers_multiple import AcquirersMultipleStrategy
from src.strategies.magic_formula import MagicFormulaStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("magicformula")


def identify_common_columns(mf_fieldnames: list[str], am_fieldnames: list[str]) -> dict[str, str]:
    """Identify common columns between MF and AM CSVs.

    Returns a mapping of column name to source preference ('mf', 'am', or 'both').
    """
    common: dict[str, str] = {}
    mf_set = set(mf_fieldnames)
    am_set = set(am_fieldnames)

    for col in mf_set & am_set:
        # Common columns - prefer non-empty values
        common[col] = "both"

    return common


def merge_row_values(
    mf_value: str | None,
    am_value: str | None,
    prefer_mf: bool = True,
) -> str | None:
    """Merge two values, preferring non-empty and non-None values.

    Args:
        mf_value: Value from Magic Formula CSV.
        am_value: Value from Acquirer's Multiple CSV.
        prefer_mf: If both are non-empty, prefer MF value.

    Returns:
        Best merged value.
    """
    # Convert empty strings to None
    mf_val = mf_value if mf_value and str(mf_value).strip() else None
    am_val = am_value if am_value and str(am_value).strip() else None

    if mf_val is None and am_val is None:
        return None
    if mf_val is None:
        return am_val
    if am_val is None:
        return mf_val

    # Both have values - prefer MF if prefer_mf is True
    return mf_val if prefer_mf else am_val


def merge_csv_results(
    magic_formula_csv: Path,
    acquirers_multiple_csv: Path,
    output_csv: Path,
    symbol_column: str = "Ticker",
) -> None:
    """Merge Magic Formula and Acquirer's Multiple results with full calculations.

    Args:
        magic_formula_csv: Path to Magic Formula results CSV.
        acquirers_multiple_csv: Path to Acquirer's Multiple results CSV.
        output_csv: Path to output merged CSV.
        symbol_column: Name of the symbol column (default: "Ticker").
    """
    # Resolve paths to absolute to avoid permission errors
    magic_formula_csv = magic_formula_csv.resolve()
    acquirers_multiple_csv = acquirers_multiple_csv.resolve()
    output_csv = output_csv.resolve()

    logger.info("=" * 80)
    logger.info("Loading input CSVs...")
    logger.info(f"  Magic Formula: {magic_formula_csv}")
    logger.info(f"  Acquirer's Multiple: {acquirers_multiple_csv}")

    # Read Magic Formula CSV
    mf_rows: list[dict[str, str]] = []
    mf_fieldnames: list[str] = []
    if magic_formula_csv.exists():
        with magic_formula_csv.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            mf_fieldnames = list(reader.fieldnames or [])
            mf_rows = list(reader)
        logger.info(f"Loaded {len(mf_rows)} rows from Magic Formula CSV")
    else:
        logger.error(f"Magic Formula CSV not found: {magic_formula_csv}")
        sys.exit(1)

    # Read Acquirer's Multiple CSV
    am_rows: list[dict[str, str]] = []
    am_fieldnames: list[str] = []
    if acquirers_multiple_csv.exists():
        with acquirers_multiple_csv.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            am_fieldnames = list(reader.fieldnames or [])
            am_rows = list(reader)
        logger.info(f"Loaded {len(am_rows)} rows from Acquirer's Multiple CSV")
    else:
        logger.error(f"Acquirer's Multiple CSV not found: {acquirers_multiple_csv}")
        sys.exit(1)

    # Identify common columns
    common_columns = identify_common_columns(mf_fieldnames, am_fieldnames)
    logger.info(f"Identified {len(common_columns)} common columns")

    # Collect all unique tickers from both CSVs
    all_symbols: set[str] = set()
    mf_lookup: dict[str, dict[str, str]] = {}
    am_lookup: dict[str, dict[str, str]] = {}

    for row in mf_rows:
        symbol_val = row.get(symbol_column, "")
        symbol = str(symbol_val).upper().strip() if symbol_val else ""
        if symbol:
            all_symbols.add(symbol)
            mf_lookup[symbol] = row

    for row in am_rows:
        symbol_val = row.get(symbol_column, "")
        symbol = str(symbol_val).upper().strip() if symbol_val else ""
        if symbol:
            all_symbols.add(symbol)
            am_lookup[symbol] = row

    logger.info(f"Found {len(all_symbols)} unique tickers across both CSVs")

    # Convert CSV rows to TickerData objects
    logger.info("Converting CSV data to TickerData objects...")
    ticker_data_list: list[TickerData] = []

    for symbol in sorted(all_symbols):
        mf_row = mf_lookup.get(symbol, {})
        am_row = am_lookup.get(symbol, {})

        # Merge base data (prefer non-empty values)
        merged_data: dict[str, str | float | None] = {}

        # Start with MF data
        for key, value in mf_row.items():
            if (
                value
                and str(value).strip()
                and str(value).strip().lower() not in ("none", "null", "")
            ):
                try:
                    # Try to convert to float if it looks numeric
                    if "." in str(value) or "e" in str(value).lower():
                        merged_data[key] = float(value)
                    else:
                        merged_data[key] = value
                except (ValueError, TypeError):
                    merged_data[key] = value

        # Add AM data, preferring AM for metrics that come from AM input
        for key, value in am_row.items():
            if key == symbol_column:
                continue

            # For metrics that come from AM input, prefer AM value
            is_am_metric = any(
                metric in key.lower()
                for metric in ["acquirers_multiple", "ev", "ebit", "enterprise_value"]
            )

            if (
                is_am_metric
                and value
                and str(value).strip()
                and str(value).strip().lower() not in ("none", "null", "")
            ):
                try:
                    if "." in str(value) or "e" in str(value).lower():
                        merged_data[f"{key}_from_am"] = float(value)
                    else:
                        merged_data[f"{key}_from_am"] = value
                except (ValueError, TypeError):
                    merged_data[f"{key}_from_am"] = value
            elif (
                (key not in merged_data or not merged_data.get(key))
                and value
                and str(value).strip()
                and str(value).strip().lower() not in ("none", "null", "")
            ):
                # Only add if not already present or if current value is empty
                try:
                    if "." in str(value) or "e" in str(value).lower():
                        merged_data[key] = float(value)
                    else:
                        merged_data[key] = value
                except (ValueError, TypeError):
                    merged_data[key] = value

        # Ensure symbol is set
        merged_data["symbol"] = symbol

        # Convert to TickerData (will validate and set defaults)
        try:
            # Filter to only valid TickerData fields and convert types
            ticker_dict: dict = {}
            for k, v in merged_data.items():
                if k in TickerData.model_fields:
                    # Convert string numbers to float
                    if isinstance(v, str) and v.strip():
                        try:
                            if "." in v or "e" in v.lower():
                                ticker_dict[k] = float(v)
                            else:
                                ticker_dict[k] = v
                        except (ValueError, TypeError):
                            ticker_dict[k] = v
                    else:
                        ticker_dict[k] = v

            ticker = TickerData.model_validate(ticker_dict, strict=False)
            ticker_data_list.append(ticker)
        except Exception as e:
            logger.warning(f"Could not create TickerData for {symbol}: {e}, using minimal data")
            # Create minimal TickerData with all required fields
            ticker = TickerData(
                symbol=symbol,
                status=TickerStatus.ACTIVE,
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
                quality_score=None,
            )
            ticker_data_list.append(ticker)

    logger.info(f"Created {len(ticker_data_list)} TickerData objects")

    # Fetch missing data and calculate all metrics for all tickers
    logger.info("Fetching missing data and calculating metrics for all tickers...")
    symbols_to_fetch = [td.symbol for td in ticker_data_list]
    # Use more conservative rate limiting to avoid yfinance API throttling
    # yfinance has strict limits on fundamental data (balance sheets, income statements)
    fetcher = AsyncYFinanceFetcher(
        max_concurrent=3,  # Reduced from 10 to avoid overwhelming yfinance
        requests_per_second=2.0,  # Reduced from 5.0 to respect yfinance rate limits
        use_cache=True,
    )
    fetched_data = asyncio.run(fetcher.fetch_multiple_tickers_async(symbols_to_fetch))

    # Create lookup for fetched data
    fetched_lookup = {td.symbol.upper(): td for td in fetched_data}

    # Merge fetched data with CSV data (CSV data takes precedence)
    final_ticker_data: list[TickerData] = []
    for csv_ticker in ticker_data_list:
        fetched_ticker = fetched_lookup.get(csv_ticker.symbol.upper())
        if fetched_ticker:
            # Merge: use CSV values where available, otherwise use fetched values
            merged = csv_ticker.model_copy(deep=True)
            # Fill in missing fields from fetched data
            for field in TickerData.model_fields:
                csv_val = getattr(merged, field, None)
                fetched_val = getattr(fetched_ticker, field, None)
                if csv_val is None and fetched_val is not None:
                    setattr(merged, field, fetched_val)
            final_ticker_data.append(merged)
        else:
            final_ticker_data.append(csv_ticker)

    logger.info(f"Final dataset: {len(final_ticker_data)} tickers with complete data")

    # Calculate missing metrics for all tickers
    from src.calculations.exceptions import InvalidDataError, MissingDataError

    updated_ticker_data: list[TickerData] = []
    for ticker in final_ticker_data:
        ticker_dict = ticker.model_dump()

        # Calculate enterprise_value if missing
        if ticker_dict.get("market_cap") and not ticker_dict.get("enterprise_value"):
            ticker_dict["enterprise_value"] = calculate_enterprise_value(
                ticker_dict["market_cap"],
                ticker_dict.get("total_debt") or 0.0,
                ticker_dict.get("cash") or 0.0,
            )

        # Calculate earnings_yield if missing
        if (
            ticker_dict.get("ebit")
            and ticker_dict.get("enterprise_value")
            and not ticker_dict.get("earnings_yield")
        ):
            try:
                ticker_dict["earnings_yield"] = calculate_earnings_yield(
                    ticker_dict["ebit"],
                    ticker_dict["enterprise_value"],
                )
            except Exception:
                # Calculation failed (invalid data), leave as None
                pass

        # Calculate return_on_capital if missing
        if (
            ticker_dict.get("ebit")
            and not ticker_dict.get("return_on_capital")
            and (ticker_dict.get("net_working_capital") or ticker_dict.get("net_fixed_assets"))
        ):
            try:
                ticker_dict["return_on_capital"] = calculate_return_on_capital(
                    ticker_dict["ebit"],
                    ticker_dict.get("net_working_capital") or 0.0,
                    ticker_dict.get("net_fixed_assets") or 0.0,
                )
            except (InvalidDataError, MissingDataError, ValueError):
                # Calculation failed (invalid/missing data), leave as None
                pass

        # Calculate acquirers_multiple if missing
        if (
            ticker_dict.get("ebit")
            and ticker_dict.get("enterprise_value")
            and not ticker_dict.get("acquirers_multiple")
        ):
            try:
                ticker_dict["acquirers_multiple"] = calculate_acquirers_multiple(
                    ticker_dict["ebit"],
                    ticker_dict["enterprise_value"],
                )
            except Exception:
                # Calculation failed (invalid data), leave as None
                pass

        updated_ticker = TickerData(**ticker_dict)
        updated_ticker_data.append(updated_ticker)

    # Run both strategies on all tickers
    logger.info("Calculating Magic Formula metrics for all tickers...")
    mf_strategy = MagicFormulaStrategy()
    mf_results = mf_strategy.calculate(updated_ticker_data)

    logger.info("Calculating Acquirer's Multiple metrics for all tickers...")
    am_strategy = AcquirersMultipleStrategy()
    am_results = am_strategy.calculate(updated_ticker_data)

    # Create lookup for strategy results
    mf_results_lookup: dict[str, dict] = {}
    for result in mf_results:
        symbol = str(result.get("symbol", "")).upper().strip()
        if symbol:
            mf_results_lookup[symbol] = result

    am_results_lookup: dict[str, dict] = {}
    for result in am_results:
        symbol = str(result.get("symbol", "")).upper().strip()
        if symbol:
            am_results_lookup[symbol] = result

    # Build final merged rows with intelligent deduplication
    logger.info("Building final merged output...")
    merged_rows: list[dict[str, Any]] = []
    all_fieldnames_set: set[str] = set()

    # Define preferred column order (user-requested order)
    preferred_order = [
        # TOP INVESTMENT DECISION COLUMNS
        "top_pick",
        symbol_column,
        "composite_rank",
        "composite_score",
        "magic_formula_score",
        "acquirers_multiple",
        "price_index_6month",
        # Company identification (consolidated)
        "name",
        "sector",
        "industry",
        # Core metrics
        "earnings_yield",
        "return_on_capital",
        "earnings_yield_rank",
        "return_on_capital_rank",
        "acquirers_multiple_rank",
        # Market data
        "market_cap",
        "price",
        "price_from",
        # Financial metrics
        "enterprise_value",
        "ebit",
        "total_debt",
        "cash",
        "net_working_capital",
        "net_fixed_assets",
        # Additional metrics
        "book_to_market",
        "free_cash_flow_yield",
        "price_to_sales",
        "price_index_12month",
        # AM-specific metrics (calculated for all)
        "debt_to_equity",
        "dividend_yield",
        "roa",
        "price_change_pct",
        "shareholder_yield",
        "buyback_yield",
        # Quality and status
        "quality_score",
        "status",
        "data_timestamp",
    ]

    # Fields to consolidate (map various names to canonical name)
    name_fields = ["Name", "Company Name (in alphabetical order)", "company_name"]
    industry_fields = ["Industry"]
    sector_fields = ["Sector"]

    # Collect all fieldnames from all sources
    for result in mf_results:
        all_fieldnames_set.update(result.keys())
    for result in am_results:
        all_fieldnames_set.update(result.keys())
    for row in mf_rows:
        all_fieldnames_set.update(row.keys())
    for row in am_rows:
        all_fieldnames_set.update(row.keys())

    # Fields that should ONLY come from their specific strategy (not mixed)
    MF_EXCLUSIVE_FIELDS = {
        "earnings_yield_rank",
        "return_on_capital_rank",
        "magic_formula_score",
    }
    AM_EXCLUSIVE_FIELDS = {
        "acquirers_multiple_rank",
    }

    # Helper function to get best value (prefer non-empty from input CSVs)
    def get_best_value(
        key: str, mf_input: dict, am_input: dict, mf_result: dict, am_result: dict
    ) -> Any:
        """Get best value, preferring input CSV values over calculated.

        IMPORTANT: Rank/score fields must come from their respective strategies:
        - MF ranks/score → ONLY from mf_result
        - AM rank → ONLY from am_result
        """
        # Strategy-exclusive fields must come from specific strategy only
        if key in MF_EXCLUSIVE_FIELDS:
            return mf_result.get(key)
        if key in AM_EXCLUSIVE_FIELDS:
            return am_result.get(key)

        # For other fields: Priority: MF input > AM input > Calculated MF > Calculated AM
        if key in mf_input:
            val = mf_input[key]
            if val and str(val).strip() and str(val).strip().lower() not in ("none", "null", ""):
                return val
        if key in am_input:
            val = am_input[key]
            if val and str(val).strip() and str(val).strip().lower() not in ("none", "null", ""):
                return val
        if key in mf_result:
            return mf_result[key]
        if key in am_result:
            return am_result[key]
        return None

    # Build merged rows
    # Create a copy of fieldnames to iterate over (since we may add new ones)
    fields_to_process = set(all_fieldnames_set)
    new_fields: set[str] = set()

    for symbol in sorted(all_symbols):
        merged_row: dict[str, Any] = {}
        mf_input = mf_lookup.get(symbol, {})
        am_input = am_lookup.get(symbol, {})
        mf_result = mf_results_lookup.get(symbol, {})
        am_result = am_results_lookup.get(symbol, {})

        # For each field, get the best value
        for key in fields_to_process:
            if key == symbol_column:
                merged_row[key] = symbol
                continue

            # Skip AM input metrics that are duplicates (we'll handle separately)
            if key.endswith("_from_am_input"):
                continue

            # Check if this is a metric that might have AM input version
            is_metric = any(
                metric in key.lower()
                for metric in [
                    "acquirers_multiple",
                    "enterprise_value",
                    "ev",
                    "ebit",
                    "operating income",
                    "oi",
                    "earnings_yield",
                    "return_on_capital",
                ]
            )

            # Get best value
            value = get_best_value(key, mf_input, am_input, mf_result, am_result)

            if value is not None:
                merged_row[key] = value

            # If AM input has this metric and it's different, add _from_am_input version
            if (
                is_metric
                and key in am_input
                and (am_val := am_input[key])
                and str(am_val).strip()
                and str(am_val).strip().lower() not in ("none", "null", "")
                and value != am_val
            ):
                new_field = f"{key}_from_am_input"
                merged_row[new_field] = am_val
                new_fields.add(new_field)
                all_fieldnames_set.add(new_field)

        # Ensure symbol is set
        merged_row[symbol_column] = symbol

        # =====================================================================
        # CONSOLIDATE DUPLICATE FIELDS
        # =====================================================================
        # Helper to get first non-empty value from multiple fields/sources
        def get_consolidated_value(
            field_names: list[str], sources: list[dict[str, str]]
        ) -> str | None:
            for field in field_names:
                for source in sources:
                    val = source.get(field)
                    if (
                        val
                        and str(val).strip()
                        and str(val).strip().lower() not in ("none", "null", "")
                    ):
                        return str(val).strip()
            return None

        # Consolidate name fields into single "name" field
        # Check original inputs FIRST (they have the actual names)
        consolidated_name = get_consolidated_value(
            ["Company Name (in alphabetical order)", "Name", "company_name", "name"],
            [mf_input, am_input, merged_row],
        )
        if consolidated_name:
            merged_row["name"] = consolidated_name

        # Consolidate industry fields into single "industry" field
        consolidated_industry = get_consolidated_value(
            ["Industry", "industry"],
            [mf_input, am_input, merged_row],
        )
        if consolidated_industry:
            merged_row["industry"] = consolidated_industry

        # Consolidate sector fields into single "sector" field
        consolidated_sector = get_consolidated_value(
            ["Sector", "sector"],
            [mf_input, am_input, merged_row],
        )
        if consolidated_sector:
            merged_row["sector"] = consolidated_sector

        # Consolidate price_from field
        price_from = None
        for pf_field in ["Price From", "price_from", "data_timestamp"]:
            for source in [am_input, mf_input, merged_row]:
                if pf_field in source:
                    val = source.get(pf_field)
                    if (
                        val
                        and str(val).strip()
                        and str(val).strip().lower() not in ("none", "null")
                    ):
                        price_from = str(val).strip()
                        break
            if price_from:
                break
        if price_from:
            merged_row["price_from"] = price_from

        # =====================================================================
        # CALCULATE AM-SPECIFIC METRICS FOR ALL TICKERS
        # =====================================================================
        # Get ticker data for calculations
        ticker_data = fetched_lookup.get(symbol)

        # Debt:Equity (%) - total_debt / shareholders_equity * 100
        if "debt_to_equity" not in merged_row or merged_row.get("debt_to_equity") is None:
            # Try from AM input first
            am_de = am_input.get("Debt:Equity (%)")
            if am_de and str(am_de).strip() and str(am_de).strip().lower() not in ("none", "null"):
                try:
                    merged_row["debt_to_equity"] = float(am_de)
                except (ValueError, TypeError):
                    pass  # Invalid value, skip
            # Calculate if not from AM
            if "debt_to_equity" not in merged_row or merged_row.get("debt_to_equity") is None:
                total_debt = merged_row.get("total_debt")
                market_cap = merged_row.get("market_cap")
                if total_debt and market_cap:
                    try:
                        # Approximate: use market_cap as proxy if no equity data
                        merged_row["debt_to_equity"] = round(
                            float(total_debt) / float(market_cap) * 100, 2
                        )
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass  # Calculation failed

        # Dividend Yield (%)
        if "dividend_yield" not in merged_row or merged_row.get("dividend_yield") is None:
            am_div = am_input.get("Div Yield (%)")
            if (
                am_div
                and str(am_div).strip()
                and str(am_div).strip().lower() not in ("none", "null")
            ):
                try:
                    merged_row["dividend_yield"] = float(am_div)
                except (ValueError, TypeError):
                    pass  # Invalid value

        # ROA (%) - from AM input or calculate
        if "roa" not in merged_row or merged_row.get("roa") is None:
            am_roa = am_input.get("ROA (5YrAvg %)")
            if (
                am_roa
                and str(am_roa).strip()
                and str(am_roa).strip().lower() not in ("none", "null")
            ):
                try:
                    merged_row["roa"] = float(am_roa)
                except (ValueError, TypeError):
                    pass  # Invalid value

        # Price Change (%) - derive from price_index_6month
        if "price_change_pct" not in merged_row or merged_row.get("price_change_pct") is None:
            am_change = am_input.get("Change (%)")
            if (
                am_change
                and str(am_change).strip()
                and str(am_change).strip().lower() not in ("none", "null")
            ):
                try:
                    merged_row["price_change_pct"] = float(am_change)
                except (ValueError, TypeError):
                    pass  # Invalid value
            # Calculate from price_index_6month if not from AM
            if "price_change_pct" not in merged_row or merged_row.get("price_change_pct") is None:
                price_idx = merged_row.get("price_index_6month")
                if price_idx:
                    try:
                        # price_index = current / past, so change = (index - 1) * 100
                        merged_row["price_change_pct"] = round((float(price_idx) - 1) * 100, 2)
                    except (ValueError, TypeError):
                        pass  # Calculation failed

        # Buyback Yield (%)
        if "buyback_yield" not in merged_row or merged_row.get("buyback_yield") is None:
            am_bb = am_input.get("BB Yield (%)")
            if am_bb and str(am_bb).strip() and str(am_bb).strip().lower() not in ("none", "null"):
                try:
                    merged_row["buyback_yield"] = float(am_bb)
                except (ValueError, TypeError):
                    pass  # Invalid value

        # Shareholder Yield (%)
        if "shareholder_yield" not in merged_row or merged_row.get("shareholder_yield") is None:
            am_shy = am_input.get("SHYield (%)")
            if (
                am_shy
                and str(am_shy).strip()
                and str(am_shy).strip().lower() not in ("none", "null")
            ):
                try:
                    merged_row["shareholder_yield"] = float(am_shy)
                except (ValueError, TypeError):
                    pass  # Invalid value
            # Calculate if we have components
            if "shareholder_yield" not in merged_row or merged_row.get("shareholder_yield") is None:
                div_yield = merged_row.get("dividend_yield")
                bb_yield = merged_row.get("buyback_yield")
                if div_yield is not None or bb_yield is not None:
                    try:
                        merged_row["shareholder_yield"] = round(
                            (float(div_yield) if div_yield else 0)
                            + (float(bb_yield) if bb_yield else 0),
                            2,
                        )
                    except (ValueError, TypeError):
                        pass  # Calculation failed

        # Remove duplicate/redundant fields that have been consolidated
        fields_to_remove = [
            "Company Name (in alphabetical order)",
            "Name",
            "company_name",
            "Industry",
            "Sector",
            "Price From",
            "Price ($)",
            "Mkt Cap ($M)",
            "Market Cap ($ Millions)",
            "EV ($M)",
            "OI ($M)",
            "FCF Yield (%)",
            "Acquirer's Multiple®",
        ]
        for field in fields_to_remove:
            if field in merged_row and field not in [symbol_column]:
                del merged_row[field]

        merged_rows.append(merged_row)

    # Add any new fields discovered during merging to the fieldnames set
    all_fieldnames_set.update(new_fields)

    # Add calculated AM-specific fields to fieldnames set
    calculated_fields = {
        "name",
        "industry",
        "sector",
        "price_from",
        "debt_to_equity",
        "dividend_yield",
        "roa",
        "price_change_pct",
        "buyback_yield",
        "shareholder_yield",
    }
    all_fieldnames_set.update(calculated_fields)

    # Remove consolidated duplicate field names from fieldnames set
    fields_to_remove_from_set = {
        "Company Name (in alphabetical order)",
        "Name",
        "company_name",
        "Industry",
        "Sector",
        "Price From",
        "Price ($)",
        "Mkt Cap ($M)",
        "Market Cap ($ Millions)",
        "EV ($M)",
        "OI ($M)",
        "FCF Yield (%)",
        "Acquirer's Multiple®",
    }
    all_fieldnames_set -= fields_to_remove_from_set

    # =========================================================================
    # COMPOSITE SCORING: 50% MF + 30% AM + 20% Momentum (6-month price index)
    # Lower composite score = better investment candidate
    # =========================================================================
    logger.info("Calculating composite scores for investment ranking...")

    # Configurable: number of top picks to highlight
    top_n_picks = 6

    # Helper to check if a value is valid (not None, not NaN, numeric)
    def is_valid_numeric(val: Any) -> bool:
        """Check if value is a valid finite number."""
        if val is None:
            return False
        try:
            f = float(val)
            return math.isfinite(f)  # Rejects NaN and Inf
        except (ValueError, TypeError):
            return False

    # First pass: Validate MF and AM components, clear invalid scores
    mf_valid_count = 0
    am_valid_count = 0

    for merged_row_data in merged_rows:
        symbol = merged_row_data.get(symbol_column, "Unknown")

        # =====================================================================
        # MAGIC FORMULA VALIDATION
        # Requires: earnings_yield AND return_on_capital (both must be valid)
        # =====================================================================
        ey = merged_row_data.get("earnings_yield")
        roc = merged_row_data.get("return_on_capital")

        mf_valid = is_valid_numeric(ey) and is_valid_numeric(roc)

        if not mf_valid:
            # Clear MF score and ranks - they're invalid
            merged_row_data["magic_formula_score"] = None
            merged_row_data["earnings_yield_rank"] = None
            merged_row_data["return_on_capital_rank"] = None
        else:
            mf_valid_count += 1

        # =====================================================================
        # ACQUIRER'S MULTIPLE VALIDATION
        # Requires: acquirers_multiple (positive value = valid)
        # Note: We accept AM from input CSVs even if ebit/ev weren't fetched
        # =====================================================================
        am = merged_row_data.get("acquirers_multiple")

        # AM is valid if it's a positive finite number
        # (positive AM means positive EBIT and positive EV)
        am_valid = is_valid_numeric(am) and float(str(am)) > 0

        if not am_valid:
            # Clear AM rank - it's invalid
            merged_row_data["acquirers_multiple_rank"] = None
            # Also clear acquirers_multiple if it's not a valid positive number
            if not is_valid_numeric(am):
                merged_row_data["acquirers_multiple"] = None
        else:
            am_valid_count += 1

    logger.info(
        f"Validation complete: {mf_valid_count} tickers valid for MF, {am_valid_count} valid for AM"
    )

    # =========================================================================
    # RE-RANK ALL VALID TICKERS
    # The strategy results may have incomplete ranks due to rate limiting.
    # Re-rank based on the validated data in merged_rows.
    # =========================================================================

    # Financial sectors to exclude from ranking
    EXCLUDED_SECTORS = {
        "financial services",
        "financial",
        "banks",
        "insurance",
        "real estate",
        "reit",
    }

    def is_excluded_sector(sector: str | None) -> bool:
        """Check if sector is excluded from ranking."""
        if not sector:
            return False
        return sector.lower() in EXCLUDED_SECTORS or any(
            excl in sector.lower() for excl in EXCLUDED_SECTORS
        )

    # Re-rank Magic Formula scores
    # Store (idx, ey, roc) tuples for ranking
    mf_rankable: list[tuple[int, float, float]] = []
    for idx, row in enumerate(merged_rows):
        ey = row.get("earnings_yield")
        roc = row.get("return_on_capital")
        sector = row.get("sector")

        if is_valid_numeric(ey) and is_valid_numeric(roc) and not is_excluded_sector(sector):
            # is_valid_numeric ensures these are convertible to float
            ey_f = float(str(ey))
            roc_f = float(str(roc))
            # MF requires positive EY and ROC
            if ey_f > 0 and roc_f > 0:
                mf_rankable.append((idx, ey_f, roc_f))

    # Rank by EY (higher = better = lower rank)
    mf_rankable.sort(key=lambda x: -x[1])  # Sort descending by EY
    ey_ranks = {item[0]: rank + 1 for rank, item in enumerate(mf_rankable)}

    # Rank by ROC (higher = better = lower rank)
    mf_rankable.sort(key=lambda x: -x[2])  # Sort descending by ROC
    roc_ranks = {item[0]: rank + 1 for rank, item in enumerate(mf_rankable)}

    # Calculate MF score (sum of ranks) and assign
    for row_idx, _, _ in mf_rankable:
        ey_rank = ey_ranks[row_idx]
        roc_rank = roc_ranks[row_idx]
        mf_score = ey_rank + roc_rank
        merged_rows[row_idx]["earnings_yield_rank"] = ey_rank
        merged_rows[row_idx]["return_on_capital_rank"] = roc_rank
        merged_rows[row_idx]["magic_formula_score"] = mf_score

    logger.info(f"Re-ranked {len(mf_rankable)} tickers for Magic Formula")

    # Re-rank Acquirer's Multiple
    am_rankable: list[tuple[int, float]] = []
    for idx, row in enumerate(merged_rows):
        am = row.get("acquirers_multiple")
        sector = row.get("sector")

        if is_valid_numeric(am) and not is_excluded_sector(sector):
            # is_valid_numeric ensures this is convertible to float
            am_f = float(str(am))
            # AM must be positive (EV/EBIT > 0 means positive EBIT)
            if am_f > 0:
                am_rankable.append((idx, am_f))

    # Rank by AM (lower = better = lower rank)
    am_rankable.sort(key=lambda x: x[1])  # Sort ascending by AM
    for rank, (row_idx, _) in enumerate(am_rankable, start=1):
        merged_rows[row_idx]["acquirers_multiple_rank"] = rank

    logger.info(f"Re-ranked {len(am_rankable)} tickers for Acquirer's Multiple")

    # Helper to get valid numeric value or None
    def get_valid_float(val: Any) -> float | None:
        """Return float if value is valid finite number, else None."""
        if val is None:
            return None
        try:
            f = float(val)
            return f if math.isfinite(f) else None
        except (ValueError, TypeError):
            return None

    # Extract values for ranking (only rows with VALID data)
    mf_scores: list[tuple[int, float]] = []  # (row_idx, value)
    am_ranks: list[tuple[int, float]] = []
    momentum_values: list[tuple[int, float]] = []

    for idx, merged_row_data in enumerate(merged_rows):
        # Magic Formula score (lower = better) - must be valid finite number
        mf_float = get_valid_float(merged_row_data.get("magic_formula_score"))
        if mf_float is not None:
            mf_scores.append((idx, mf_float))

        # Acquirer's Multiple rank (lower = better) - must be valid finite number
        am_float = get_valid_float(merged_row_data.get("acquirers_multiple_rank"))
        if am_float is not None:
            am_ranks.append((idx, am_float))

        # 6-month price index (higher = better, we'll invert for ranking)
        mom_float = get_valid_float(merged_row_data.get("price_index_6month"))
        if mom_float is not None:
            momentum_values.append((idx, mom_float))

    # Calculate percentile ranks (0-100, lower = better for all)
    def calculate_percentile_ranks(
        values: list[tuple[int, float]], lower_is_better: bool = True
    ) -> dict[int, float]:
        """Calculate percentile ranks for values.

        Args:
            values: List of (row_idx, value) tuples (already validated as finite).
            lower_is_better: If True, lower values get lower (better) percentiles.

        Returns:
            Dict mapping row_idx to percentile (0-100).
        """
        if not values:
            return {}

        # Sort by value
        if lower_is_better:
            sorted_values = sorted(values, key=lambda x: x[1])
        else:
            # Higher is better → sort descending so highest gets rank 1
            sorted_values = sorted(values, key=lambda x: x[1], reverse=True)

        n = len(sorted_values)
        percentiles: dict[int, float] = {}
        for rank, (idx, _) in enumerate(sorted_values, start=1):
            # Percentile: (rank - 1) / (n - 1) * 100, or 0 if n=1
            if n > 1:
                percentiles[idx] = ((rank - 1) / (n - 1)) * 100
            else:
                percentiles[idx] = 0.0

        return percentiles

    # Calculate percentiles for each factor
    mf_percentiles = calculate_percentile_ranks(mf_scores, lower_is_better=True)
    am_percentiles = calculate_percentile_ranks(am_ranks, lower_is_better=True)
    # Momentum: higher is better, so we pass lower_is_better=False
    mom_percentiles = calculate_percentile_ranks(momentum_values, lower_is_better=False)

    # Weights for composite score
    weight_mf = 0.50
    weight_am = 0.30
    weight_momentum = 0.20

    # Calculate composite scores
    # REQUIRE: MF score AND AM rank (both mandatory)
    # OPTIONAL: Momentum - use neutral 50th percentile if missing (rate limiting)
    NEUTRAL_PERCENTILE = 50.0  # Neutral position when momentum data missing

    composite_scores: list[tuple[int, float]] = []
    for idx, merged_row_data in enumerate(merged_rows):
        mf_pct = mf_percentiles.get(idx)
        am_pct = am_percentiles.get(idx)
        mom_pct = mom_percentiles.get(idx)

        # MF and AM are REQUIRED for composite score
        if mf_pct is not None and am_pct is not None:
            # Use neutral 50th percentile for momentum if missing
            effective_mom_pct = mom_pct if mom_pct is not None else NEUTRAL_PERCENTILE
            composite = (
                (mf_pct * weight_mf) + (am_pct * weight_am) + (effective_mom_pct * weight_momentum)
            )
            composite_scores.append((idx, composite))
            merged_row_data["composite_score"] = round(composite, 2)
        else:
            # Mark as not investable - missing required MF or AM data
            merged_row_data["composite_score"] = None

    # Sort by composite score to determine ranks and top picks
    composite_scores.sort(key=lambda x: x[1])

    # Assign composite ranks and top_pick flags
    for rank, (idx, _) in enumerate(composite_scores, start=1):
        merged_rows[idx]["composite_rank"] = rank
        merged_rows[idx]["top_pick"] = rank <= top_n_picks

    # Mark rows without composite score
    for merged_row_data in merged_rows:
        if "composite_rank" not in merged_row_data:
            merged_row_data["composite_rank"] = None
            merged_row_data["top_pick"] = False

    # Add new fields to fieldnames set
    all_fieldnames_set.add("composite_score")
    all_fieldnames_set.add("composite_rank")
    all_fieldnames_set.add("top_pick")

    # Sort merged_rows by composite_rank (top picks first, then by rank)
    def sort_key_for_output(row: dict[str, Any]) -> tuple[int, int]:
        """Sort key: top picks first, then by composite rank."""
        is_top = row.get("top_pick", False)
        rank = row.get("composite_rank")
        # Top picks first (0), then others (1)
        # Within each group, sort by rank (None goes last)
        return (0 if is_top else 1, rank if rank is not None else 9999)

    merged_rows.sort(key=sort_key_for_output)

    logger.info(f"Composite scoring complete. Top {top_n_picks} picks identified.")

    # =========================================================================
    # BUILD FINAL COLUMN ORDER
    # =========================================================================

    # Build final column order: preferred order first, then remaining fields
    all_fieldnames: list[str] = []
    used_fields: set[str] = set()

    # Create a frozen copy of fieldnames for safe iteration
    all_fieldnames_frozen = set(all_fieldnames_set)

    # Add fields in preferred order
    for field in preferred_order:
        # Try exact match
        if field in all_fieldnames_frozen:
            all_fieldnames.append(field)
            used_fields.add(field)
        else:
            # Try case-insensitive match
            field_lower = field.lower()
            for existing_field in all_fieldnames_frozen:
                if existing_field.lower() == field_lower and existing_field not in used_fields:
                    all_fieldnames.append(existing_field)
                    used_fields.add(existing_field)
                    break

    # Add remaining fields alphabetically (excluding already used)
    remaining = sorted(all_fieldnames_frozen - used_fields)
    all_fieldnames.extend(remaining)

    # Add timestamp to output filename if not already present
    output_path = Path(output_csv)
    if "_" not in output_path.stem or not any(c.isdigit() for c in output_path.stem.split("_")[-1]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path.parent / f"{output_path.stem}_{timestamp}{output_path.suffix}"

    # Clean NaN values from all rows - replace with empty string
    def clean_nan_values(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace NaN, nan, None values with empty string for CSV output."""
        cleaned = []
        for row in rows:
            clean_row = {}
            for key, val in row.items():
                # Check if value should be replaced with empty string
                is_none = val is None
                is_float_invalid = isinstance(val, float) and (math.isnan(val) or math.isinf(val))
                is_str_invalid = isinstance(val, str) and val.lower() in (
                    "nan",
                    "none",
                    "inf",
                    "-inf",
                )
                if is_none or is_float_invalid or is_str_invalid:
                    clean_row[key] = ""
                else:
                    clean_row[key] = val
            cleaned.append(clean_row)
        return cleaned

    merged_rows = clean_nan_values(merged_rows)

    # Write merged CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, all_fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)
        logger.info(f"Merged {len(merged_rows)} rows to {output_path}")
    except PermissionError:
        logger.error(
            f"Permission denied: Cannot write to {output_csv}. "
            "File may be open in another program. Please close it and try again."
        )
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge Magic Formula and Acquirer's Multiple results into a single CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge results from both strategies
  python scripts/merge_strategy_results.py \\
    --magic-formula magic_formula_with_results.csv \\
    --acquirers-multiple acquirers_multiple_with_results.csv \\
    --output combined_results.csv
        """,
    )

    parser.add_argument(
        "--magic-formula",
        type=Path,
        required=True,
        help="Path to Magic Formula results CSV",
    )

    parser.add_argument(
        "--acquirers-multiple",
        type=Path,
        required=True,
        help="Path to Acquirer's Multiple results CSV",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output merged CSV",
    )

    parser.add_argument(
        "--symbol-column",
        type=str,
        default="Ticker",
        help="Name of the symbol column (default: Ticker)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Merging Strategy Results with Full Calculations")
    logger.info("=" * 80)

    merge_csv_results(
        args.magic_formula,
        args.acquirers_multiple,
        args.output,
        args.symbol_column,
    )

    logger.info("=" * 80)
    logger.info("Merge completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
