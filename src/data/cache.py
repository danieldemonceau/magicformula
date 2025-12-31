"""SQLite caching layer for ticker data."""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings_pydantic import settings
from src.data.models.ticker_data import TickerData

logger = logging.getLogger("magicformula")

# Cache expiration from settings
CACHE_EXPIRATION_HOURS = settings.cache_expiration_hours


class TickerCache:
    """SQLite-based cache for ticker data."""

    def __init__(self, cache_path: Path | str | None = None) -> None:
        """Initialize cache.

        Args:
            cache_path: Path to SQLite database file. If None, uses settings default.
        """
        self.cache_path = Path(cache_path or settings.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ticker_cache (
                    symbol TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON ticker_cache(expires_at)
                """,
            )
            conn.commit()

    def get(self, symbol: str) -> TickerData | None:
        """Get ticker data from cache.

        Args:
            symbol: Ticker symbol.

        Returns:
            TickerData if found and not expired, None otherwise.
        """
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT data, expires_at
                    FROM ticker_cache
                    WHERE symbol = ? AND expires_at > ?
                    """,
                    (symbol.upper(), datetime.now().isoformat()),
                )
                row = cursor.fetchone()

                if row:
                    data_dict = json.loads(row[0])
                    ticker_data = TickerData(**data_dict)
                    logger.debug(f"Cache hit for {symbol}")
                    return ticker_data

                logger.debug(f"Cache miss for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None

    def set(self, symbol: str, ticker_data: TickerData) -> None:
        """Store ticker data in cache.

        Args:
            symbol: Ticker symbol.
            ticker_data: TickerData object to cache.
        """
        try:
            expires_at = datetime.now() + timedelta(hours=CACHE_EXPIRATION_HOURS)

            # Convert to dict, handling datetime serialization
            data_dict = ticker_data.model_dump()
            if data_dict.get("data_timestamp"):
                data_dict["data_timestamp"] = data_dict["data_timestamp"].isoformat()

            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ticker_cache
                    (symbol, data, timestamp, expires_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        symbol.upper(),
                        json.dumps(data_dict),
                        datetime.now().isoformat(),
                        expires_at.isoformat(),
                    ),
                )
                conn.commit()

            logger.debug(f"Cached data for {symbol}")

        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

    def clear(self, symbol: str | None = None) -> None:
        """Clear cache entries.

        Args:
            symbol: Specific symbol to clear. If None, clears all expired entries.
        """
        try:
            with sqlite3.connect(self.cache_path) as conn:
                if symbol:
                    conn.execute(
                        "DELETE FROM ticker_cache WHERE symbol = ?",
                        (symbol.upper(),),
                    )
                else:
                    # Clear expired entries
                    conn.execute(
                        "DELETE FROM ticker_cache WHERE expires_at < ?",
                        (datetime.now().isoformat(),),
                    )
                conn.commit()

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def clear_all(self) -> None:
        """Clear all cache entries."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute("DELETE FROM ticker_cache")
                conn.commit()
            logger.info("Cleared all cache entries")

        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")

