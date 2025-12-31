"""Pydantic models for ticker financial data."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from src.data.models.ticker_status import TickerStatus
from src.data.types import TickerDataDict


class TickerData(BaseModel):
    """Financial data for a single ticker."""

    symbol: str = Field(..., description="Stock ticker symbol")
    status: TickerStatus = Field(
        default=TickerStatus.ACTIVE,
        description="Status of ticker data",
    )
    sector: str | None = Field(None, description="Company sector")
    industry: str | None = Field(None, description="Company industry")
    price: float | None = Field(None, description="Current stock price")
    market_cap: float | None = Field(None, description="Market capitalization")
    total_debt: float | None = Field(None, description="Total debt")
    cash: float | None = Field(None, description="Cash and cash equivalents")
    ebit: float | None = Field(None, description="Earnings Before Interest and Taxes")
    net_working_capital: float | None = Field(None, description="Net Working Capital")
    net_fixed_assets: float | None = Field(None, description="Net Fixed Assets")
    enterprise_value: float | None = Field(None, description="Enterprise Value")
    earnings_yield: float | None = Field(None, description="Earnings Yield (EBIT/EV)")
    return_on_capital: float | None = Field(None, description="Return on Capital")
    acquirers_multiple: float | None = Field(None, description="Acquirer's Multiple (EV/EBIT)")
    price_index_6month: float | None = Field(None, description="Price index (6 months)")
    price_index_12month: float | None = Field(None, description="Price index (12 months)")
    book_to_market: float | None = Field(None, description="Book to Market ratio")
    free_cash_flow_yield: float | None = Field(None, description="Free Cash Flow Yield")
    price_to_sales: float | None = Field(None, description="Price to Sales ratio")
    data_timestamp: datetime | None = Field(
        None,
        description="Timestamp when data was fetched",
    )
    quality_score: float | None = Field(
        None,
        description="Data quality score (0-1, higher is better)",
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol is uppercase."""
        return v.upper().strip()

    @field_validator("price", "market_cap", "total_debt", "cash", "ebit")
    @classmethod
    def validate_positive_or_none(cls, v: float | None) -> float | None:
        """Validate financial metrics are positive or None."""
        if v is not None and v < 0:
            return None
        return v

    @field_validator("data_timestamp", mode="before")
    @classmethod
    def parse_datetime(cls, v: str | datetime | None) -> datetime | None:
        """Parse datetime from string if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            result: datetime = datetime.fromisoformat(v)
            return result
        if isinstance(v, datetime):
            return v
        return None

    def to_dict(self) -> TickerDataDict:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=False)
