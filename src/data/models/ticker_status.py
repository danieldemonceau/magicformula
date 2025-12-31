"""Ticker status enumeration."""

from enum import Enum


class TickerStatus(str, Enum):
    """Status of a ticker's data availability."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    DELISTED = "DELISTED"
    STALE = "STALE"
