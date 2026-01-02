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
        with magic_formula_csv.open(encoding="utf-8") as f:
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
        with acquirers_multiple_csv.open(encoding="utf-8") as f:
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

    # Define preferred column order (user-friendly)
    preferred_order = [
        # INVESTMENT DECISION COLUMNS (most important - at the top)
        "top_pick",
        "composite_rank",
        "composite_score",
        # Basic identification
        symbol_column,
        "Company Name (in alphabetical order)",
        "Name",
        "company_name",
        "name",
        # Company info
        "sector",
        "Sector",
        "industry",
        "Industry",
        # Market data
        "market_cap",
        "Market Cap ($ Millions)",
        "Mkt Cap ($M)",
        "Market Cap",
        "Price From",
        "price_from",
        "Most Recent Quarter Data",
        "most_recent_quarter_data",
        "price",
        "Price ($)",
        "Price",
        # Magic Formula metrics
        "earnings_yield",
        "return_on_capital",
        # Magic Formula ranks
        "earnings_yield_rank",
        "return_on_capital_rank",
        # Magic Formula score
        "magic_formula_score",
        # Acquirer's Multiple
        "acquirers_multiple",
        "Acquirer's Multiple®",
        "acquirers_multiple_rank",
        # Core financial metrics
        "enterprise_value",
        "EV ($M)",
        "Enterprise Value",
        "ebit",
        "OI ($M)",
        "Operating Income",
        "total_debt",
        "Total Debt",
        "cash",
        "Cash",
        # Working capital and assets
        "net_working_capital",
        "Net Working Capital",
        "net_fixed_assets",
        "Net Fixed Assets",
        # Additional metrics
        "book_to_market",
        "Book to Market",
        "free_cash_flow_yield",
        "FCF Yield (%)",
        "price_to_sales",
        "Price to Sales",
        "price_index_6month",
        "price_index_12month",
        # Quality and status
        "quality_score",
        "Quality Score",
        "status",
        "Status",
        # Timestamps
        "data_timestamp",
        "Data Timestamp",
    ]

    # Collect all fieldnames from all sources
    for result in mf_results:
        all_fieldnames_set.update(result.keys())
    for result in am_results:
        all_fieldnames_set.update(result.keys())
    for row in mf_rows:
        all_fieldnames_set.update(row.keys())
    for row in am_rows:
        all_fieldnames_set.update(row.keys())

    # Helper function to get best value (prefer non-empty from input CSVs)
    def get_best_value(
        key: str, mf_input: dict, am_input: dict, mf_result: dict, am_result: dict
    ) -> Any:
        """Get best value, preferring input CSV values over calculated."""
        # Priority: MF input > AM input > Calculated MF > Calculated AM
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

        merged_rows.append(merged_row)

    # Add any new fields discovered during merging to the fieldnames set
    all_fieldnames_set.update(new_fields)

    # =========================================================================
    # COMPOSITE SCORING: 50% MF + 30% AM + 20% Momentum (6-month price index)
    # Lower composite score = better investment candidate
    # =========================================================================
    logger.info("Calculating composite scores for investment ranking...")

    # Configurable: number of top picks to highlight
    top_n_picks = 6

    # Extract values for ranking (only rows with valid data)
    mf_scores: list[tuple[int, float | None]] = []  # (row_idx, value)
    am_ranks: list[tuple[int, float | None]] = []
    momentum_values: list[tuple[int, float | None]] = []

    for idx, merged_row_data in enumerate(merged_rows):
        # Magic Formula score (lower = better)
        mf_val = merged_row_data.get("magic_formula_score")
        if mf_val is not None:
            try:
                mf_scores.append((idx, float(mf_val)))
            except (ValueError, TypeError):
                pass  # Skip non-numeric values

        # Acquirer's Multiple rank (lower = better)
        am_rank_val = merged_row_data.get("acquirers_multiple_rank")
        if am_rank_val is not None:
            try:
                am_ranks.append((idx, float(am_rank_val)))
            except (ValueError, TypeError):
                pass  # Skip non-numeric values

        # 6-month price index (higher = better, we'll invert for ranking)
        mom_val = merged_row_data.get("price_index_6month")
        if mom_val is not None:
            try:
                momentum_values.append((idx, float(mom_val)))
            except (ValueError, TypeError):
                pass  # Skip non-numeric values

    # Calculate percentile ranks (0-100, lower = better for all)
    def calculate_percentile_ranks(
        values: list[tuple[int, float | None]], lower_is_better: bool = True
    ) -> dict[int, float]:
        """Calculate percentile ranks for values.

        Args:
            values: List of (row_idx, value) tuples.
            lower_is_better: If True, lower values get lower (better) percentiles.

        Returns:
            Dict mapping row_idx to percentile (0-100).
        """
        if not values:
            return {}

        # Filter out None values
        valid_values = [(idx, v) for idx, v in values if v is not None]
        if not valid_values:
            return {}

        # Sort by value
        if lower_is_better:
            sorted_values = sorted(valid_values, key=lambda x: x[1])
        else:
            # Higher is better → sort descending so highest gets rank 1
            sorted_values = sorted(valid_values, key=lambda x: x[1], reverse=True)

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
    composite_scores: list[tuple[int, float]] = []
    for idx, merged_row_data in enumerate(merged_rows):
        mf_pct = mf_percentiles.get(idx)
        am_pct = am_percentiles.get(idx)
        mom_pct = mom_percentiles.get(idx)

        # Only calculate if we have at least MF or AM score
        if mf_pct is not None or am_pct is not None:
            # Use available percentiles, default missing to 50 (neutral)
            mf_component = (mf_pct if mf_pct is not None else 50.0) * weight_mf
            am_component = (am_pct if am_pct is not None else 50.0) * weight_am
            mom_component = (mom_pct if mom_pct is not None else 50.0) * weight_momentum

            composite = mf_component + am_component + mom_component
            composite_scores.append((idx, composite))
            merged_row_data["composite_score"] = round(composite, 2)
        else:
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
