"""CSV output writer."""

import csv
import logging
from pathlib import Path

from src.data.types import StrategyResultDict

logger = logging.getLogger("magicformula")


class CSVWriter:
    """CSV file writer for strategy results."""

    def __init__(self, output_path: Path | str, append: bool = False) -> None:
        """Initialize CSV writer.

        Args:
            output_path: Path to output CSV file.
            append: If True, append to existing file instead of overwriting.
        """
        self.output_path = Path(output_path)
        self.append = append

    def write(self, data: list[StrategyResultDict]) -> None:
        """Write data to CSV file.

        Args:
            data: List of dictionaries to write.
        """
        if not data:
            logger.warning("No data to write to CSV")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        keys = data[0].keys()

        mode = "a" if self.append else "w"
        file_exists = self.output_path.exists() and self.append

        with open(self.output_path, mode, newline="", encoding="utf-8") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            if not file_exists:
                dict_writer.writeheader()
            dict_writer.writerows(data)

        action = "Appended" if self.append else "Wrote"
        logger.info(f"{action} {len(data)} records to {self.output_path}")

    def merge_with_csv(
        self,
        input_csv_path: Path | str,
        output_csv_path: Path | str,
        data: list[StrategyResultDict],
        symbol_column: str = "Ticker",
    ) -> None:
        """Merge calculated results with existing CSV and write to new output file.

        Matches rows by symbol and adds new columns with calculated values.
        The input CSV remains untouched.

        Args:
            input_csv_path: Path to input CSV file.
            output_csv_path: Path to output CSV file (will be created/overwritten).
            data: List of dictionaries with calculated results (must have 'symbol' key).
            symbol_column: Name of the symbol column in the input CSV.
        """
        input_path = Path(input_csv_path)
        output_path = Path(output_csv_path)

        if not input_path.exists():
            logger.error(f"Input CSV file not found: {input_path}")
            return

        # Read existing CSV
        existing_rows: list[StrategyResultDict] = []
        with open(input_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            existing_rows = list(reader)

        # Create lookup for calculated data by symbol
        calculated_lookup: dict[str, StrategyResultDict] = {}
        for item in data:
            symbol_val = item.get("symbol", "")
            symbol = str(symbol_val).upper() if symbol_val is not None else ""
            if symbol:
                calculated_lookup[symbol] = item

        # Merge calculated data into existing rows
        merged_rows: list[StrategyResultDict] = []
        for row in existing_rows:
            symbol_val = row.get(symbol_column, "")
            symbol = str(symbol_val).upper() if symbol_val is not None else ""
            if symbol in calculated_lookup:
                # Merge calculated fields (excluding 'symbol' to avoid duplication)
                calculated = calculated_lookup[symbol]
                merged_row = {**row}
                for key, value in calculated.items():
                    if key != "symbol":
                        merged_row[key] = value
                merged_rows.append(merged_row)
            else:
                # Keep original row if no calculated data
                merged_rows.append(row)

        # Write merged data to NEW output file
        if merged_rows:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                all_fieldnames = list(fieldnames)
                # Add new calculated columns that aren't already in fieldnames
                for item in data:
                    for key in item:
                        if key != "symbol" and key not in all_fieldnames:
                            all_fieldnames.append(key)

                writer = csv.DictWriter(f, all_fieldnames)
                writer.writeheader()
                writer.writerows(merged_rows)

            logger.info(
                f"Merged calculated results: {input_path} -> {output_path} ({len(merged_rows)} rows)"
            )
