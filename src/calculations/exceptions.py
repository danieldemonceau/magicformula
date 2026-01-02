"""Custom exceptions for calculation errors."""


class CalculationError(Exception):
    """Base exception for calculation errors."""


class MissingDataError(CalculationError):
    """Raised when required data is missing for a calculation."""


class InvalidDataError(CalculationError):
    """Raised when data is invalid for a calculation."""


class DelistedTickerError(CalculationError):
    """Raised when a ticker is delisted or inactive."""


class StaleDataError(CalculationError):
    """Raised when data is stale (too old)."""


class OutlierDataError(CalculationError):
    """Raised when data contains outliers that may be errors."""


class RateLimitError(CalculationError):
    """Raised when API rate limit is exceeded."""
