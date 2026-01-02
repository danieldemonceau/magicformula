"""JSON output writer."""

import json
import logging
from pathlib import Path

from src.data.types import StrategyResultDict

logger = logging.getLogger("magicformula")


class JSONWriter:
    """JSON file writer for strategy results."""

    def __init__(self, output_path: Path | str) -> None:
        """Initialize JSON writer.

        Args:
            output_path: Path to output JSON file.
        """
        self.output_path = Path(output_path)

    def write(self, data: list[StrategyResultDict]) -> None:
        """Write data to JSON file.

        Args:
            data: List of dictionaries to write.
        """
        if not data:
            logger.warning("No data to write to JSON")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as output_file:
            json.dump(data, output_file, indent=2, default=str)

        logger.info(f"Wrote {len(data)} records to {self.output_path}")
