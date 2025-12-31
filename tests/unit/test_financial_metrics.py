"""Unit tests for financial metrics calculations."""

import pytest

from src.calculations.exceptions import InvalidDataError, MissingDataError
from src.calculations.financial_metrics import (
    calculate_acquirers_multiple,
    calculate_earnings_yield,
    calculate_enterprise_value,
    calculate_return_on_capital,
)


class TestEnterpriseValue:
    """Tests for Enterprise Value calculation."""

    def test_calculate_ev_basic(self) -> None:
        """Test basic EV calculation."""
        ev = calculate_enterprise_value(
            market_cap=1_000_000_000,
            total_debt=100_000_000,
            cash_and_equivalents=50_000_000,
        )
        assert ev == 1_050_000_000

    def test_calculate_ev_no_debt(self) -> None:
        """Test EV calculation with no debt."""
        ev = calculate_enterprise_value(
            market_cap=1_000_000_000,
            total_debt=0,
            cash_and_equivalents=50_000_000,
        )
        assert ev == 950_000_000

    def test_calculate_ev_no_cash(self) -> None:
        """Test EV calculation with no cash."""
        ev = calculate_enterprise_value(
            market_cap=1_000_000_000,
            total_debt=100_000_000,
            cash_and_equivalents=0,
        )
        assert ev == 1_100_000_000

    def test_calculate_ev_missing_market_cap(self) -> None:
        """Test EV calculation with missing market cap."""
        with pytest.raises(MissingDataError):
            calculate_enterprise_value(
                market_cap=None,
                total_debt=100_000_000,
                cash_and_equivalents=50_000_000,
            )


class TestEarningsYield:
    """Tests for Earnings Yield calculation."""

    def test_calculate_earnings_yield_basic(self) -> None:
        """Test basic earnings yield calculation."""
        ey = calculate_earnings_yield(ebit=100_000_000, enterprise_value=1_000_000_000)
        assert ey == 0.1

    def test_calculate_earnings_yield_missing_ebit(self) -> None:
        """Test earnings yield with missing EBIT."""
        with pytest.raises(MissingDataError):
            calculate_earnings_yield(ebit=None, enterprise_value=1_000_000_000)

    def test_calculate_earnings_yield_missing_ev(self) -> None:
        """Test earnings yield with missing EV."""
        with pytest.raises(MissingDataError):
            calculate_earnings_yield(ebit=100_000_000, enterprise_value=None)

    def test_calculate_earnings_yield_zero_ev(self) -> None:
        """Test earnings yield with zero EV."""
        with pytest.raises(InvalidDataError):
            calculate_earnings_yield(ebit=100_000_000, enterprise_value=0)

    def test_calculate_earnings_yield_negative_ev(self) -> None:
        """Test earnings yield with negative EV."""
        with pytest.raises(InvalidDataError):
            calculate_earnings_yield(ebit=100_000_000, enterprise_value=-1_000_000_000)


class TestReturnOnCapital:
    """Tests for Return on Capital calculation."""

    def test_calculate_roc_basic(self) -> None:
        """Test basic ROC calculation."""
        roc = calculate_return_on_capital(
            ebit=100_000_000,
            net_working_capital=50_000_000,
            net_fixed_assets=150_000_000,
        )
        assert roc == 0.5

    def test_calculate_roc_missing_ebit(self) -> None:
        """Test ROC with missing EBIT."""
        with pytest.raises(MissingDataError):
            calculate_return_on_capital(
                ebit=None,
                net_working_capital=50_000_000,
                net_fixed_assets=150_000_000,
            )

    def test_calculate_roc_zero_capital(self) -> None:
        """Test ROC with zero capital employed."""
        with pytest.raises(InvalidDataError):
            calculate_return_on_capital(
                ebit=100_000_000,
                net_working_capital=0,
                net_fixed_assets=0,
            )

    def test_calculate_roc_negative_capital(self) -> None:
        """Test ROC with negative capital employed."""
        with pytest.raises(InvalidDataError):
            calculate_return_on_capital(
                ebit=100_000_000,
                net_working_capital=-50_000_000,
                net_fixed_assets=-50_000_000,
            )


class TestAcquirersMultiple:
    """Tests for Acquirer's Multiple calculation."""

    def test_calculate_acquirers_multiple_basic(self) -> None:
        """Test basic Acquirer's Multiple calculation."""
        am = calculate_acquirers_multiple(ebit=100_000_000, enterprise_value=1_000_000_000)
        assert am == 10.0

    def test_calculate_acquirers_multiple_missing_ebit(self) -> None:
        """Test Acquirer's Multiple with missing EBIT."""
        with pytest.raises(MissingDataError):
            calculate_acquirers_multiple(ebit=None, enterprise_value=1_000_000_000)

    def test_calculate_acquirers_multiple_missing_ev(self) -> None:
        """Test Acquirer's Multiple with missing EV."""
        with pytest.raises(MissingDataError):
            calculate_acquirers_multiple(ebit=100_000_000, enterprise_value=None)

    def test_calculate_acquirers_multiple_zero_ebit(self) -> None:
        """Test Acquirer's Multiple with zero EBIT."""
        with pytest.raises(InvalidDataError):
            calculate_acquirers_multiple(ebit=0, enterprise_value=1_000_000_000)

    def test_calculate_acquirers_multiple_negative_ebit(self) -> None:
        """Test Acquirer's Multiple with negative EBIT."""
        with pytest.raises(InvalidDataError):
            calculate_acquirers_multiple(ebit=-100_000_000, enterprise_value=1_000_000_000)
