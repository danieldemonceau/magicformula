"""CSV input reader for Magic Formula and Acquirer's Multiple data."""

import csv
import logging
from pathlib import Path

from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus

logger = logging.getLogger("magicformula")


class CSVReader:
    """Reader for Magic Formula and Acquirer's Multiple CSV files."""

    @staticmethod
    def read_magic_formula_csv(csv_path: Path) -> list[TickerData]:
        """Read Magic Formula CSV from website.

        Expected format:
        Company Name (in alphabetical order),Ticker,Market Cap ($ Millions),Price From,Most Recent Quarter Data

        Args:
            csv_path: Path to Magic Formula CSV file.

        Returns:
            List of TickerData objects.
        """
        ticker_data_list: list[TickerData] = []

        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try different possible column names
                    ticker = (
                        row.get("Ticker", "") or row.get("ticker", "") or row.get("TICKER", "")
                    ).strip()
                    if not ticker:
                        continue

                    # Parse market cap (remove quotes and convert millions to actual value)
                    market_cap_str = (
                        (
                            row.get("Market Cap ($ Millions)", "")
                            or row.get("market cap ($ millions)", "")
                            or row.get("Market Cap", "")
                        )
                        .replace('"', "")
                        .replace(",", "")
                        .strip()
                    )
                    market_cap = None
                    if market_cap_str:
                        try:
                            market_cap = (
                                float(market_cap_str) * 1_000_000
                            )  # Convert millions to actual
                        except ValueError:
                            logger.warning(
                                f"{ticker}: Could not parse market cap: {market_cap_str}"
                            )

                    # Get company name if available
                    ticker_data = TickerData(
                        symbol=ticker,
                        status=TickerStatus.ACTIVE,
                        sector=None,
                        industry=None,
                        price=None,
                        market_cap=market_cap,
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
                    ticker_data_list.append(ticker_data)

            logger.info(
                f"Loaded {len(ticker_data_list)} tickers from Magic Formula CSV: {csv_path}"
            )
            return ticker_data_list

        except FileNotFoundError:
            logger.error(f"Magic Formula CSV file not found: {csv_path}")
            return []
        except Exception as e:
            logger.error(f"Error reading Magic Formula CSV: {e}")
            return []

    @staticmethod
    def read_acquirers_multiple_csv(csv_path: Path) -> list[TickerData]:
        """Read Acquirer's Multiple CSV from website.

        Expected format:
        Ticker,Name,Price ($),Change (%),Industry,Acquirer's Multiple®,FCF Yield (%),E(r) (%),IV/P,ROA (5YrAvg %),Incr. Growth (%),SHYield (%),BB Yield (%),Div Yield (%),Mkt Cap ($M),EV ($M),Debt:Equity (%),OI ($M)

        Args:
            csv_path: Path to Acquirer's Multiple CSV file.

        Returns:
            List of TickerData objects.
        """
        ticker_data_list: list[TickerData] = []

        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ticker = (
                        row.get("Ticker", "") or row.get("ticker", "") or row.get("TICKER", "")
                    ).strip()
                    if not ticker:
                        continue

                    # Parse fields from CSV (try different column name variations)
                    price = None
                    price_str = (
                        (row.get("Price ($)", "") or row.get("Price", "") or row.get("price", ""))
                        .replace('"', "")
                        .replace(",", "")
                        .strip()
                    )
                    if price_str:
                        try:
                            price = float(price_str)
                        except ValueError:
                            # Invalid price format, leave as None
                            pass

                    market_cap = None
                    market_cap_str = (
                        (
                            row.get("Mkt Cap ($M)", "")
                            or row.get("Mkt Cap", "")
                            or row.get("Market Cap", "")
                        )
                        .replace('"', "")
                        .replace(",", "")
                        .strip()
                    )
                    if market_cap_str:
                        try:
                            market_cap = (
                                float(market_cap_str) * 1_000_000
                            )  # Convert millions to actual
                        except ValueError:
                            # Invalid market cap format, leave as None
                            pass

                    enterprise_value = None
                    ev_str = (
                        (
                            row.get("EV ($M)", "")
                            or row.get("EV", "")
                            or row.get("Enterprise Value", "")
                        )
                        .replace('"', "")
                        .replace(",", "")
                        .strip()
                    )
                    if ev_str:
                        try:
                            enterprise_value = (
                                float(ev_str) * 1_000_000
                            )  # Convert millions to actual
                        except ValueError:
                            # Invalid enterprise value format, leave as None
                            pass

                    # OI = Operating Income = EBIT
                    ebit = None
                    oi_str = (
                        (
                            row.get("OI ($M)", "")
                            or row.get("OI", "")
                            or row.get("Operating Income", "")
                        )
                        .replace('"', "")
                        .replace(",", "")
                        .strip()
                    )
                    if oi_str:
                        try:
                            ebit = float(oi_str) * 1_000_000  # Convert millions to actual
                        except ValueError:
                            # Invalid operating income format, leave as None
                            pass

                    # Get industry
                    industry = (row.get("Industry", "") or row.get("industry", "")).strip()

                    ticker_data = TickerData(
                        symbol=ticker,
                        status=TickerStatus.ACTIVE,
                        sector=None,
                        industry=industry if industry else None,
                        price=price,
                        market_cap=market_cap,
                        total_debt=None,
                        cash=None,
                        ebit=ebit,
                        net_working_capital=None,
                        net_fixed_assets=None,
                        enterprise_value=enterprise_value,
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
                    ticker_data_list.append(ticker_data)

            logger.info(
                f"Loaded {len(ticker_data_list)} tickers from Acquirer's Multiple CSV: {csv_path}"
            )
            return ticker_data_list

        except FileNotFoundError:
            logger.error(f"Acquirer's Multiple CSV file not found: {csv_path}")
            return []
        except Exception as e:
            logger.error(f"Error reading Acquirer's Multiple CSV: {e}")
            return []

    @staticmethod
    def read_csv(csv_path: Path, csv_type: str = "auto") -> list[TickerData]:
        """Read CSV file, auto-detecting type or using specified type.

        Args:
            csv_path: Path to CSV file.
            csv_type: Type of CSV ("magic_formula", "acquirers_multiple", or "auto").

        Returns:
            List of TickerData objects.
        """
        if csv_type == "auto":
            # Try to detect type by reading fieldnames
            try:
                with open(csv_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    # Check for Acquirer's Multiple specific columns
                    if any(
                        "Acquirer's Multiple" in col or "Acquirer's Multiple®" in col
                        for col in fieldnames
                    ):
                        csv_type = "acquirers_multiple"
                    elif any("Market Cap ($ Millions)" in col for col in fieldnames):
                        csv_type = "magic_formula"
                    elif any("EV ($M)" in col or "Mkt Cap ($M)" in col for col in fieldnames):
                        csv_type = "acquirers_multiple"
                    else:
                        logger.warning(
                            f"Could not auto-detect CSV type for {csv_path}, defaulting to Magic Formula"
                        )
                        csv_type = "magic_formula"
            except Exception as e:
                logger.warning(f"Error auto-detecting CSV type: {e}, defaulting to Magic Formula")
                csv_type = "magic_formula"

        if csv_type == "acquirers_multiple":
            return CSVReader.read_acquirers_multiple_csv(csv_path)
        else:
            return CSVReader.read_magic_formula_csv(csv_path)
