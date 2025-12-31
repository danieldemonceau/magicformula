"""Pytest configuration and fixtures."""

import pytest
from src.data.models.ticker_data import TickerData


@pytest.fixture
def sample_ticker_data() -> TickerData:
    """Create sample ticker data for testing."""
    return TickerData(
        symbol="AAPL",
        price=150.0,
        market_cap=2_500_000_000_000,
        total_debt=100_000_000_000,
        cash=50_000_000_000,
        ebit=100_000_000_000,
        net_working_capital=50_000_000_000,
        net_fixed_assets=100_000_000_000,
        enterprise_value=2_550_000_000_000,
        earnings_yield=0.0392,
        return_on_capital=0.6667,
        acquirers_multiple=25.5,
        price_index_6month=1.1,
        price_index_12month=1.2,
    )


@pytest.fixture
def sample_ticker_data_list() -> list[TickerData]:
    """Create list of sample ticker data for testing."""
    return [
        TickerData(
            symbol="AAPL",
            price=150.0,
            market_cap=2_500_000_000_000,
            total_debt=100_000_000_000,
            cash=50_000_000_000,
            ebit=100_000_000_000,
            net_working_capital=50_000_000_000,
            net_fixed_assets=100_000_000_000,
            enterprise_value=2_550_000_000_000,
            earnings_yield=0.05,
            return_on_capital=0.67,
        ),
        TickerData(
            symbol="MSFT",
            price=300.0,
            market_cap=2_200_000_000_000,
            total_debt=80_000_000_000,
            cash=100_000_000_000,
            ebit=80_000_000_000,
            net_working_capital=40_000_000_000,
            net_fixed_assets=80_000_000_000,
            enterprise_value=2_180_000_000_000,
            earnings_yield=0.037,
            return_on_capital=0.67,
        ),
        TickerData(
            symbol="GOOGL",
            price=120.0,
            market_cap=1_500_000_000_000,
            total_debt=50_000_000_000,
            cash=200_000_000_000,
            ebit=60_000_000_000,
            net_working_capital=30_000_000_000,
            net_fixed_assets=60_000_000_000,
            enterprise_value=1_350_000_000_000,
            earnings_yield=0.044,
            return_on_capital=0.67,
        ),
    ]


@pytest.fixture
def invalid_ticker_data() -> TickerData:
    """Create invalid ticker data (missing required fields)."""
    return TickerData(
        symbol="INVALID",
        price=100.0,
        market_cap=1_000_000_000,
        ebit=None,
        earnings_yield=None,
        return_on_capital=None,
    )

