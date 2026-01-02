#!/usr/bin/env python3
"""Main CLI entry point for running investment strategy analysis."""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings_pydantic import settings
from src.calculations.exceptions import InvalidDataError, MissingDataError
from src.calculations.financial_metrics import (
    calculate_acquirers_multiple,
    calculate_earnings_yield,
    calculate_enterprise_value,
    calculate_return_on_capital,
)
from src.data.csv_reader import CSVReader
from src.data.fetchers.async_fetcher import AsyncYFinanceFetcher
from src.data.fetchers.yfinance_fetcher import YFinanceFetcher
from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus
from src.data.quality import assess_data_quality
from src.output.csv_writer import CSVWriter
from src.output.json_writer import JSONWriter
from src.strategies.acquirers_multiple import AcquirersMultipleStrategy
from src.strategies.dca import DCAStrategy
from src.strategies.magic_formula import MagicFormulaStrategy
from src.utils.logging_config import setup_logging

logger = setup_logging()


def load_symbols(symbols_file: Path) -> list[str]:
    """Load symbols from JSON file.

    Args:
        symbols_file: Path to symbols JSON file.

    Returns:
        List of ticker symbols.
    """
    try:
        with Path(symbols_file).open(encoding="utf-8") as f:
            data: dict[str, list[str]] = json.load(f)
            symbols: list[str] = data.get("symbols", [])
            return symbols
    except FileNotFoundError:
        logger.error(f"Symbols file not found: {symbols_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in symbols file: {e}")
        sys.exit(1)


def get_strategy(
    strategy_name: str,
    **kwargs: str | int | float | bool | date | None,
) -> MagicFormulaStrategy | AcquirersMultipleStrategy | DCAStrategy:
    """Factory function to get strategy instance.

    Args:
        strategy_name: Name of the strategy.
        **kwargs: Additional arguments for strategy initialization.

    Returns:
        Strategy instance.

    Raises:
        ValueError: If strategy name is not supported.
    """
    if strategy_name == settings.strategy_magic_formula:
        return MagicFormulaStrategy()
    if strategy_name == settings.strategy_acquirers_multiple:
        return AcquirersMultipleStrategy()
    if strategy_name == settings.strategy_dca:
        return DCAStrategy(**kwargs)  # type: ignore[arg-type]

    raise ValueError(f"Unsupported strategy: {strategy_name}")


def get_writer(output_path: Path, output_format: str) -> CSVWriter | JSONWriter:
    """Factory function to get output writer.

    Args:
        output_path: Path to output file.
        output_format: Output format ('csv' or 'json').

    Returns:
        Writer instance.

    Raises:
        ValueError: If output format is not supported.
    """
    if output_format.lower() == "csv":
        return CSVWriter(output_path)
    if output_format.lower() == "json":
        return JSONWriter(output_path)

    raise ValueError(f"Unsupported output format: {output_format}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Investment Strategy Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Magic Formula analysis
  python scripts/run_analysis.py --strategy magic_formula

  # Run Acquirer's Multiple analysis
  python scripts/run_analysis.py --strategy acquirers_multiple

  # Run DCA analysis with custom parameters
  python scripts/run_analysis.py --strategy dca --dca-amount 500 --dca-frequency monthly

  # Use CSV input (Magic Formula or Acquirer's Multiple from website)
  python scripts/run_analysis.py --strategy magic_formula --csv-input magic_formula.csv
  python scripts/run_analysis.py --strategy acquirers_multiple --csv-input acquirers_multiple.csv

  # CSV input will append calculated results back to the CSV file
        """,
    )

    parser.add_argument(
        "--strategy",
        choices=settings.supported_strategies,
        default=settings.default_strategy,
        help=f"Investment strategy to run (default: {settings.default_strategy})",
    )

    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help=f"Path to symbols JSON file (default: {settings.symbols_file})",
    )

    parser.add_argument(
        "--csv-input",
        type=Path,
        default=None,
        help="Path to CSV input file (Magic Formula or Acquirer's Multiple format). "
        "If provided, will use CSV data and append calculated results back to CSV.",
    )

    parser.add_argument(
        "--csv-type",
        choices=["auto", "magic_formula", "acquirers_multiple"],
        default="auto",
        help="Type of CSV file (auto-detect if not specified)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: auto-generated in data/outputs/)",
    )

    parser.add_argument(
        "--output-format",
        choices=["csv", "json"],
        default=settings.default_output_format,
        help=f"Output file format (default: {settings.default_output_format})",
    )

    parser.add_argument(
        "--async",
        action="store_true",
        dest="use_async",
        default=True,  # Make async the default - much faster
        help="Use async fetching for better performance (default: True)",
    )
    parser.add_argument(
        "--no-async",
        action="store_false",
        dest="use_async",
        help="Disable async fetching (use sequential fetching)",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=settings.async_max_concurrent,
        help=f"Max concurrent requests for async mode (default: {settings.async_max_concurrent})",
    )

    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=settings.async_requests_per_second,
        help=f"Rate limit for async mode (default: {settings.async_requests_per_second})",
    )

    parser.add_argument(
        "--sleep-time",
        type=float,
        default=settings.default_sleep_time,
        help=f"Sleep time between API requests in seconds (default: {settings.default_sleep_time})",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching",
    )

    # DCA-specific arguments
    parser.add_argument(
        "--dca-amount",
        type=float,
        default=settings.dca_investment_amount,
        help=f"DCA investment amount per period (default: {settings.dca_investment_amount})",
    )

    parser.add_argument(
        "--dca-frequency",
        choices=["daily", "weekly", "monthly"],
        default=settings.dca_frequency,
        help=f"DCA investment frequency (default: {settings.dca_frequency})",
    )

    parser.add_argument(
        "--dca-slippage",
        type=int,
        default=settings.dca_slippage_bps,
        help=f"DCA slippage in basis points (default: {settings.dca_slippage_bps})",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        # Also set level for all child loggers
        logging.getLogger("magicformula").setLevel(logging.DEBUG)

    logger.info("=" * 80)
    logger.info("Investment Strategy Analysis Tool")
    logger.info("=" * 80)

    time_start = time.time()

    # Determine input source: CSV or symbols file
    use_csv_input = args.csv_input is not None and args.csv_input.exists()

    if use_csv_input:
        logger.info(f"Loading data from CSV: {args.csv_input}")
        ticker_data = CSVReader.read_csv(args.csv_input, csv_type=args.csv_type)
        logger.info(f"Loaded {len(ticker_data)} tickers from CSV")

        # Fetch missing data from yfinance to complete the dataset
        logger.info("Fetching missing data from yfinance to complete calculations...")
        symbols_to_fetch = [td.symbol for td in ticker_data]
        if args.use_async:
            fetcher_csv_async: AsyncYFinanceFetcher = AsyncYFinanceFetcher(
                max_concurrent=args.max_concurrent,
                requests_per_second=args.requests_per_second,
                use_cache=not args.no_cache,
            )
            fetched_data = asyncio.run(
                fetcher_csv_async.fetch_multiple_tickers_async(symbols_to_fetch)
            )
        else:
            fetcher_csv_sync: YFinanceFetcher = YFinanceFetcher(
                sleep_time=args.sleep_time,
                use_cache=not args.no_cache,
            )
            fetched_data = fetcher_csv_sync.fetch_multiple_tickers(symbols_to_fetch)

        # Merge CSV data with fetched data (CSV data takes precedence for fields it has)
        fetched_lookup = {td.symbol.upper(): td for td in fetched_data}
        merged_ticker_data = []
        for csv_ticker in ticker_data:
            fetched_ticker = fetched_lookup.get(csv_ticker.symbol.upper())
            if fetched_ticker:
                # Check if fetched ticker has any useful data (not all None)
                has_useful_fetched_data = any(
                    [
                        fetched_ticker.price is not None,
                        fetched_ticker.ebit is not None,
                        fetched_ticker.total_debt is not None,
                        fetched_ticker.cash is not None,
                        fetched_ticker.net_working_capital is not None,
                        fetched_ticker.net_fixed_assets is not None,
                        fetched_ticker.sector,
                        fetched_ticker.industry,
                    ]
                )

                # Merge: use CSV values where available, otherwise use fetched values
                merged = csv_ticker.model_copy(deep=True)
                # Fill in missing fields from fetched data (only if fetched has data)
                if merged.market_cap is None and fetched_ticker.market_cap is not None:
                    merged.market_cap = fetched_ticker.market_cap
                if merged.price is None and fetched_ticker.price is not None:
                    merged.price = fetched_ticker.price
                if merged.ebit is None and fetched_ticker.ebit is not None:
                    merged.ebit = fetched_ticker.ebit
                if merged.enterprise_value is None and fetched_ticker.enterprise_value is not None:
                    merged.enterprise_value = fetched_ticker.enterprise_value
                if (
                    merged.net_working_capital is None
                    and fetched_ticker.net_working_capital is not None
                ):
                    merged.net_working_capital = fetched_ticker.net_working_capital
                if merged.net_fixed_assets is None and fetched_ticker.net_fixed_assets is not None:
                    merged.net_fixed_assets = fetched_ticker.net_fixed_assets
                if merged.total_debt is None and fetched_ticker.total_debt is not None:
                    merged.total_debt = fetched_ticker.total_debt
                if merged.cash is None and fetched_ticker.cash is not None:
                    merged.cash = fetched_ticker.cash
                if not merged.sector and fetched_ticker.sector:
                    merged.sector = fetched_ticker.sector
                if not merged.industry and fetched_ticker.industry:
                    merged.industry = fetched_ticker.industry
                # Merge calculated metrics from fetched data
                if (
                    merged.price_index_6month is None
                    and fetched_ticker.price_index_6month is not None
                ):
                    merged.price_index_6month = fetched_ticker.price_index_6month
                if (
                    merged.price_index_12month is None
                    and fetched_ticker.price_index_12month is not None
                ):
                    merged.price_index_12month = fetched_ticker.price_index_12month
                if merged.book_to_market is None and fetched_ticker.book_to_market is not None:
                    merged.book_to_market = fetched_ticker.book_to_market
                if (
                    merged.free_cash_flow_yield is None
                    and fetched_ticker.free_cash_flow_yield is not None
                ):
                    merged.free_cash_flow_yield = fetched_ticker.free_cash_flow_yield
                if merged.price_to_sales is None and fetched_ticker.price_to_sales is not None:
                    merged.price_to_sales = fetched_ticker.price_to_sales
                # Preserve timestamp from fetched data (it's fresh)
                if merged.data_timestamp is None and fetched_ticker.data_timestamp is not None:
                    merged.data_timestamp = fetched_ticker.data_timestamp

                # Only update status if fetched data has useful information
                # If fetched data is all None, keep CSV status (likely ACTIVE)
                if has_useful_fetched_data:
                    if fetched_ticker.status == TickerStatus.DELISTED and (
                        merged.price is None and not merged.market_cap
                    ):
                        merged.status = TickerStatus.DELISTED
                    elif fetched_ticker.status in (
                        TickerStatus.DATA_UNAVAILABLE,
                        TickerStatus.STALE,
                    ):
                        # Only set to DATA_UNAVAILABLE if we actually tried to fetch and got nothing
                        # If CSV has market_cap, keep it as ACTIVE for now (assess_data_quality will update if needed)
                        if merged.market_cap or merged.ebit:
                            # We have financial data from CSV, so keep ACTIVE status
                            # assess_data_quality will update it appropriately
                            pass
                        else:
                            merged.status = fetched_ticker.status
                # If fetched data is all None, keep CSV status (don't overwrite with DATA_UNAVAILABLE)
                merged_ticker_data.append(merged)
            else:
                # No fetched data - keep CSV data but don't mark as DELISTED if we have financial data
                if csv_ticker.market_cap or csv_ticker.ebit:
                    csv_ticker = csv_ticker.model_copy(
                        update={"status": TickerStatus.DATA_UNAVAILABLE}
                    )
                merged_ticker_data.append(csv_ticker)

        ticker_data = merged_ticker_data
        logger.info(f"Merged CSV data with yfinance data for {len(ticker_data)} tickers")
    else:
        # Use symbols file (original behavior)
        symbols_file = args.symbols_file or settings.symbols_file
        symbols = load_symbols(symbols_file)
        logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")

        # Choose fetcher based on async flag
        if args.use_async:
            logger.info("Using async fetcher for improved performance")
            fetcher_file_async: AsyncYFinanceFetcher = AsyncYFinanceFetcher(
                max_concurrent=args.max_concurrent,
                requests_per_second=args.requests_per_second,
                use_cache=not args.no_cache,
            )
            logger.info(f"Fetching data for {len(symbols)} symbols (async)...")
            ticker_data = asyncio.run(fetcher_file_async.fetch_multiple_tickers_async(symbols))
        else:
            fetcher_file_sync: YFinanceFetcher = YFinanceFetcher(
                sleep_time=args.sleep_time,
                use_cache=not args.no_cache,
            )
            logger.info(f"Fetching data for {len(symbols)} symbols...")
            ticker_data = fetcher_file_sync.fetch_multiple_tickers(symbols)

    logger.info(f"Successfully prepared data for {len(ticker_data)} tickers")

    # Log status breakdown
    from collections import Counter

    status_counts = Counter(td.status for td in ticker_data)
    logger.info(f"Ticker status breakdown: {dict(status_counts)}")

    strategy_kwargs = {}
    if args.strategy == settings.strategy_dca:
        strategy_kwargs = {
            "investment_amount": args.dca_amount,
            "frequency": args.dca_frequency,
            "slippage_bps": args.dca_slippage,
        }

    strategy = get_strategy(args.strategy, **strategy_kwargs)
    logger.info(f"Running {strategy.get_strategy_name()} strategy...")

    # Calculate missing metrics for tickers that have base data but missing calculated metrics
    updated_ticker_data = []
    # Log summary of what data we have
    logger.debug("Data availability summary before calculations:")
    sample_ticker = ticker_data[0] if ticker_data else None
    if sample_ticker:
        logger.debug(
            f"Sample ticker {sample_ticker.symbol}: "
            f"market_cap={sample_ticker.market_cap is not None}, "
            f"ebit={sample_ticker.ebit is not None}, "
            f"enterprise_value={sample_ticker.enterprise_value is not None}, "
            f"nwc={sample_ticker.net_working_capital is not None}, "
            f"nfa={sample_ticker.net_fixed_assets is not None}"
        )
    for ticker in ticker_data:
        # Create a copy to modify (Pydantic models are immutable)
        ticker_dict = ticker.model_dump()

        # Calculate enterprise_value if we have market_cap but not EV
        if ticker_dict.get("market_cap") and not ticker_dict.get("enterprise_value"):
            ticker_dict["enterprise_value"] = calculate_enterprise_value(
                ticker_dict["market_cap"],
                ticker_dict.get("total_debt") or 0.0,
                ticker_dict.get("cash") or 0.0,
            )

        # Calculate earnings_yield if we have EBIT and EV
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
            except Exception as e:
                logger.debug(
                    f"{ticker_dict.get('symbol', 'Unknown')}: "
                    f"Could not calculate earnings_yield: {e}"
                )

        # Calculate return_on_capital - try pre-calculated first, then calculate
        symbol = ticker_dict.get("symbol", "Unknown")
        has_ebit = bool(ticker_dict.get("ebit"))
        has_roc = ticker_dict.get("return_on_capital") is not None
        has_nwc = ticker_dict.get("net_working_capital") is not None
        has_nfa = ticker_dict.get("net_fixed_assets") is not None

        # Step 1: If we don't have ROC, try to get pre-calculated from Alpha Vantage
        if has_ebit and not has_roc:
            logger.debug(f"{symbol}: ROC missing, trying Alpha Vantage for pre-calculated ROC...")
            try:
                from src.data.fetchers.alphavantage_fetcher import (
                    fetch_missing_financial_data_alphavantage,
                )

                av_data = fetch_missing_financial_data_alphavantage(symbol, ["return_on_capital"])

                if "return_on_capital" in av_data and av_data["return_on_capital"] is not None:
                    ticker_dict["return_on_capital"] = av_data["return_on_capital"]
                    logger.info(
                        f"{symbol}: Using pre-calculated ROC from Alpha Vantage: "
                        f"{ticker_dict['return_on_capital']:.4f}"
                    )
                    has_roc = True
            except Exception as e:
                logger.debug(f"{symbol}: Could not fetch ROC from Alpha Vantage: {e}")

        # Step 2: If still no ROC, calculate it using NWC and NFA (which we may have from yfinance or Alpha Vantage)
        if has_ebit and not has_roc and (has_nwc or has_nfa):
            try:
                ticker_dict["return_on_capital"] = calculate_return_on_capital(
                    ticker_dict["ebit"],
                    ticker_dict.get("net_working_capital") or 0.0,
                    ticker_dict.get("net_fixed_assets") or 0.0,
                )
            except (MissingDataError, InvalidDataError, ValueError) as e:
                # InvalidDataError means capital_employed <= 0 (negative NWC or NFA)
                # Try Alpha Vantage as fallback if we got invalid data
                error_msg = str(e)
                if (
                    "Capital Employed must be positive" in error_msg
                    or "must be positive" in error_msg
                ):
                    logger.info(
                        f"{symbol}: ROC calculation failed ({error_msg}), trying Alpha Vantage fallback..."
                    )
                    try:
                        from src.data.fetchers.alphavantage_fetcher import (
                            fetch_missing_financial_data_alphavantage,
                        )

                        missing_fields = ["net_working_capital", "net_fixed_assets"]
                        av_data = fetch_missing_financial_data_alphavantage(symbol, missing_fields)

                        # Update ticker_dict with Alpha Vantage data if we got better values
                        if (
                            "net_working_capital" in av_data
                            and av_data["net_working_capital"] is not None
                        ):
                            ticker_dict["net_working_capital"] = av_data["net_working_capital"]
                        if (
                            "net_fixed_assets" in av_data
                            and av_data["net_fixed_assets"] is not None
                        ):
                            ticker_dict["net_fixed_assets"] = av_data["net_fixed_assets"]

                        # Try calculating ROC again with Alpha Vantage data
                        if (
                            ticker_dict.get("net_working_capital") is not None
                            or ticker_dict.get("net_fixed_assets") is not None
                        ):
                            try:
                                ticker_dict["return_on_capital"] = calculate_return_on_capital(
                                    ticker_dict["ebit"],
                                    ticker_dict.get("net_working_capital") or 0.0,
                                    ticker_dict.get("net_fixed_assets") or 0.0,
                                )
                                logger.info(
                                    f"{symbol}: Successfully calculated ROC using Alpha Vantage data"
                                )
                            except (MissingDataError, InvalidDataError, ValueError) as e2:
                                logger.debug(
                                    f"{symbol}: Still could not calculate ROC after Alpha Vantage: {e2}"
                                )
                    except Exception as av_error:
                        logger.debug(f"{symbol}: Alpha Vantage fallback failed: {av_error}")
                else:
                    logger.debug(
                        f"{symbol}: Could not calculate return_on_capital: {e}",
                    )
        elif has_ebit and not has_roc and not has_nwc and not has_nfa:
            # Try Alpha Vantage fallback for missing NWC/NFA
            logger.info(
                f"{symbol}: Missing NWC/NFA for ROC calculation, trying Alpha Vantage fallback..."
            )
            try:
                from src.data.fetchers.alphavantage_fetcher import (
                    fetch_missing_financial_data_alphavantage,
                )

                missing_fields = ["net_working_capital", "net_fixed_assets"]
                av_data = fetch_missing_financial_data_alphavantage(symbol, missing_fields)

                # Update ticker_dict with Alpha Vantage data
                if "net_working_capital" in av_data and av_data["net_working_capital"] is not None:
                    ticker_dict["net_working_capital"] = av_data["net_working_capital"]
                if "net_fixed_assets" in av_data and av_data["net_fixed_assets"] is not None:
                    ticker_dict["net_fixed_assets"] = av_data["net_fixed_assets"]

                # Try calculating ROC again if we got data
                if (
                    ticker_dict.get("net_working_capital") is not None
                    or ticker_dict.get("net_fixed_assets") is not None
                ):
                    try:
                        ticker_dict["return_on_capital"] = calculate_return_on_capital(
                            ticker_dict["ebit"],
                            ticker_dict.get("net_working_capital") or 0.0,
                            ticker_dict.get("net_fixed_assets") or 0.0,
                        )
                    except (MissingDataError, InvalidDataError, ValueError) as e:
                        logger.debug(
                            f"{symbol}: Could not calculate return_on_capital after Alpha Vantage fetch: {e}"
                        )
            except Exception as e:
                logger.debug(f"{symbol}: Alpha Vantage fallback failed: {e}")

        # Calculate acquirers_multiple if we have EBIT and EV
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
                pass  # Skip if calculation fails

        # Recreate TickerData with updated values
        updated_ticker = TickerData(**ticker_dict)
        # Assess data quality
        updated_ticker = assess_data_quality(updated_ticker)
        updated_ticker_data.append(updated_ticker)

    ticker_data = updated_ticker_data

    results = strategy.calculate(ticker_data)
    logger.info(f"Calculated results for {len(results)} tickers")

    # If CSV input was used, merge results with CSV and write to new output file
    if use_csv_input:
        if args.output:
            output_path = args.output
        else:
            # Generate output filename: {input_name}_with_results_{timestamp}.csv
            input_stem = args.csv_input.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = args.csv_input.parent / f"{input_stem}_with_results_{timestamp}.csv"

        logger.info(f"Merging calculated results with CSV: {args.csv_input} -> {output_path}")
        try:
            writer_csv: CSVWriter = CSVWriter(output_path)
            writer_csv.merge_with_csv(args.csv_input, output_path, results)
        except PermissionError:
            logger.error(
                f"Permission denied: Cannot write to {output_path}. "
                "Please close the file if it's open in another program (e.g., Excel) and try again."
            )
            sys.exit(1)
    else:
        # Original behavior: write to new file
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{args.strategy}_results_{timestamp}.{args.output_format}"
            output_path = settings.output_dir / output_filename
        else:
            output_path = args.output

        try:
            writer: CSVWriter | JSONWriter = get_writer(output_path, args.output_format)
            writer.write(results)
        except PermissionError:
            logger.error(
                f"Permission denied: Cannot write to {output_path}. "
                "Please close the file if it's open in another program (e.g., Excel) and try again."
            )
            sys.exit(1)

    time_end = time.time()
    duration = time_end - time_start

    logger.info("=" * 80)
    logger.info(f"Analysis completed in {duration:.2f} seconds")
    logger.info(f"Results written to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
