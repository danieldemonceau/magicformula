"""Pydantic Settings for application configuration."""

from pathlib import Path

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
    async_max_concurrent: int = 10
    async_requests_per_second: float = 5.0
    async_retry_attempts: int = 3
    async_retry_delay: float = 1.0

    # Cache settings
    cache_enabled: bool = True
    cache_expiration_hours: int = 24

    # Data quality settings
    staleness_threshold_days: int = 1
    min_quality_score: float = 0.5

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
    dca_investment_amount: float = 1000.0
    dca_frequency: str = "monthly"
    dca_slippage_bps: int = 10

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

