"""Pydantic Settings for application configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings using Pydantic."""

    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    output_dir: Path = data_dir / "outputs"
    symbols_file: Path = data_dir / "symbols.json"
    cache_dir: Path = data_dir / "cache"
    cache_path: Path = cache_dir / "ticker_cache.db"

    # Default settings
    default_sleep_time: float = 1.0
    default_output_format: str = "csv"
    default_strategy: str = "magic_formula"

    # Strategy names
    strategy_magic_formula: str = "magic_formula"
    strategy_acquirers_multiple: str = "acquirers_multiple"
    strategy_dca: str = "dca"

    # Async fetcher settings (conservative defaults to avoid API throttling)
    async_max_concurrent: int = Field(default=10, ge=1, description="Maximum concurrent async requests")
    async_requests_per_second: float = Field(default=5.0, gt=0.0, description="Rate limit for async requests")
    async_retry_attempts: int = Field(default=3, ge=1, description="Number of retry attempts")
    async_retry_delay: float = Field(default=1.0, gt=0.0, description="Initial retry delay in seconds")

    # Cache settings
    cache_enabled: bool = True
    cache_expiration_hours: int = Field(default=24, ge=0, description="Cache expiration time in hours")

    # Data quality settings
    staleness_threshold_days: int = Field(default=1, ge=0, description="Days before data is considered stale")
    min_quality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum quality score threshold")

    # Outlier thresholds
    outlier_pe_ratio: float = 1000.0
    outlier_price_to_sales: float = 1000.0
    outlier_earnings_yield: float = 1.0
    outlier_return_on_capital: float = 10.0
    outlier_acquirers_multiple: float = 1000.0
    outlier_market_cap: float = 1e15
    outlier_enterprise_value: float = 1e15
    outlier_price: float = 1e6

    # Lookback periods (months)
    lookback_6months: int = 6
    lookback_12months: int = 12

    # DCA defaults
    dca_investment_amount: float = Field(default=1000.0, gt=0.0, description="DCA investment amount per period")
    dca_frequency: str = Field(default="monthly", description="DCA frequency: daily, weekly, or monthly")
    dca_slippage_bps: int = Field(default=10, ge=0, description="DCA slippage in basis points")

    # Alpha Vantage API (optional fallback)
    av_apikey: str | None = None
    av_rate_limit_delay: float = Field(default=12.1, gt=0.0, description="Seconds between Alpha Vantage API calls (5 calls/min)")
    av_timeout_seconds: float = Field(default=10.0, gt=0.0, description="Request timeout in seconds")
    av_cache_path: Path = cache_dir / "alphavantage_cache.db"
    av_cache_expiration_hours: int = Field(default=24, ge=0, description="Alpha Vantage cache expiration in hours")
    ebitda_to_ebit_multiplier: float = Field(default=0.75, gt=0.0, le=1.0, description="Multiplier for EBITDA fallback to EBIT (industry-specific approximation)")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def supported_strategies(self) -> list[str]:
        """Get list of supported strategies."""
        return [
            self.strategy_magic_formula,
            self.strategy_acquirers_multiple,
            self.strategy_dca,
        ]


# Global settings instance
settings = Settings()

