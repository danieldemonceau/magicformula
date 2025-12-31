#!/usr/bin/env python3
"""Main CLI entry point for running investment strategy analysis."""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings_pydantic import settings
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
        with open(symbols_file, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("symbols", [])
    except FileNotFoundError:
        logger.error(f"Symbols file not found: {symbols_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in symbols file: {e}")
        sys.exit(1)


def get_strategy(strategy_name: str, **kwargs) -> (
    MagicFormulaStrategy | AcquirersMultipleStrategy | DCAStrategy
):
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
        return DCAStrategy(**kwargs)

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
        help="Use async fetching for better performance",
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
        if args.use_async:
            fetcher = AsyncYFinanceFetcher(
                max_concurrent=args.max_concurrent,
                requests_per_second=args.requests_per_second,
                use_cache=not args.no_cache,
            )
            symbols_to_fetch = [td.symbol for td in ticker_data]
            fetched_data = asyncio.run(fetcher.fetch_multiple_tickers_async(symbols_to_fetch))
        else:
            fetcher = YFinanceFetcher(
                sleep_time=args.sleep_time,
                use_cache=not args.no_cache,
            )
            symbols_to_fetch = [td.symbol for td in ticker_data]
            fetched_data = fetcher.fetch_multiple_tickers(symbols_to_fetch)

        # Merge CSV data with fetched data (CSV data takes precedence for fields it has)
        fetched_lookup = {td.symbol: td for td in fetched_data}
        merged_ticker_data = []
        for csv_ticker in ticker_data:
            fetched_ticker = fetched_lookup.get(csv_ticker.symbol)
            if fetched_ticker:
                # Merge: use CSV values where available, otherwise use fetched values
                merged = csv_ticker.model_copy(deep=True)
                # Fill in missing fields from fetched data
                if merged.market_cap is None:
                    merged.market_cap = fetched_ticker.market_cap
                if merged.price is None:
                    merged.price = fetched_ticker.price
                if merged.ebit is None:
                    merged.ebit = fetched_ticker.ebit
                if merged.enterprise_value is None:
                    merged.enterprise_value = fetched_ticker.enterprise_value
                if merged.net_working_capital is None:
                    merged.net_working_capital = fetched_ticker.net_working_capital
                if merged.net_fixed_assets is None:
                    merged.net_fixed_assets = fetched_ticker.net_fixed_assets
                if merged.total_debt is None:
                    merged.total_debt = fetched_ticker.total_debt
                if merged.cash is None:
                    merged.cash = fetched_ticker.cash
                if not merged.sector:
                    merged.sector = fetched_ticker.sector
                if not merged.industry:
                    merged.industry = fetched_ticker.industry
                merged_ticker_data.append(merged)
            else:
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
            fetcher = AsyncYFinanceFetcher(
                max_concurrent=args.max_concurrent,
                requests_per_second=args.requests_per_second,
                use_cache=not args.no_cache,
            )
            logger.info(f"Fetching data for {len(symbols)} symbols (async)...")
            ticker_data = asyncio.run(fetcher.fetch_multiple_tickers_async(symbols))
        else:
            fetcher = YFinanceFetcher(
                sleep_time=args.sleep_time,
                use_cache=not args.no_cache,
            )
            logger.info(f"Fetching data for {len(symbols)} symbols...")
            ticker_data = fetcher.fetch_multiple_tickers(symbols)

    logger.info(f"Successfully prepared data for {len(ticker_data)} tickers")

    # Log status breakdown
    from collections import Counter
    from src.data.models.ticker_status import TickerStatus

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
            except Exception:
                pass

        # Calculate return_on_capital if we have EBIT and capital components
        if ticker_dict.get("ebit") and not ticker_dict.get("return_on_capital"):
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
                    # InvalidDataError means capital_employed <= 0 (negative NWC or NFA)
                    # This is expected for some companies, so we just skip ROC calculation
                    logger.debug(
                        f"{ticker_dict.get('symbol', 'Unknown')}: "
                        f"Could not calculate return_on_capital: {e}",
                    )
                    pass

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
                pass

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
            # Generate output filename: {input_name}_with_results.csv
            input_stem = args.csv_input.stem
            output_path = args.csv_input.parent / f"{input_stem}_with_results.csv"

        logger.info(f"Merging calculated results with CSV: {args.csv_input} -> {output_path}")
        writer = CSVWriter(output_path)
        writer.merge_with_csv(args.csv_input, output_path, results)
    else:
        # Original behavior: write to new file
        if not args.output:
            output_filename = f"{args.strategy}_results.{args.output_format}"
            output_path = settings.output_dir / output_filename
        else:
            output_path = args.output

        writer = get_writer(output_path, args.output_format)
        writer.write(results)

    time_end = time.time()
    duration = time_end - time_start

    logger.info("=" * 80)
    logger.info(f"Analysis completed in {duration:.2f} seconds")
    logger.info(f"Results written to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

