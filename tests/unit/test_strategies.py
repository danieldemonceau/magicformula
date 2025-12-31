"""Unit tests for investment strategies."""

from src.data.models.ticker_data import TickerData
from src.strategies.acquirers_multiple import AcquirersMultipleStrategy
from src.strategies.magic_formula import MagicFormulaStrategy


class TestMagicFormulaStrategy:
    """Tests for Magic Formula strategy."""

    def test_calculate_magic_formula(self, sample_ticker_data_list: list[TickerData]) -> None:
        """Test Magic Formula calculation."""
        strategy = MagicFormulaStrategy()
        results = strategy.calculate(sample_ticker_data_list)

        assert len(results) == 3
        assert "magic_formula_score" in results[0]
        assert "earnings_yield_rank" in results[0]
        assert "return_on_capital_rank" in results[0]

        scores = [r["magic_formula_score"] for r in results]
        assert scores == sorted(scores)

    def test_calculate_magic_formula_empty(self) -> None:
        """Test Magic Formula with empty list."""
        strategy = MagicFormulaStrategy()
        results = strategy.calculate([])
        assert results == []

    def test_calculate_magic_formula_invalid_data(
        self,
        invalid_ticker_data: TickerData,
    ) -> None:
        """Test Magic Formula with invalid data."""
        strategy = MagicFormulaStrategy()
        results = strategy.calculate([invalid_ticker_data])
        assert results == []

    def test_get_strategy_name(self) -> None:
        """Test strategy name."""
        strategy = MagicFormulaStrategy()
        assert strategy.get_strategy_name() == "Magic Formula"


class TestAcquirersMultipleStrategy:
    """Tests for Acquirer's Multiple strategy."""

    def test_calculate_acquirers_multiple(
        self,
        sample_ticker_data_list: list[TickerData],
    ) -> None:
        """Test Acquirer's Multiple calculation."""
        strategy = AcquirersMultipleStrategy()
        results = strategy.calculate(sample_ticker_data_list)

        assert len(results) == 3
        assert "acquirers_multiple_rank" in results[0]

        ranks = [r["acquirers_multiple_rank"] for r in results]
        assert ranks == sorted(ranks)

    def test_calculate_acquirers_multiple_empty(self) -> None:
        """Test Acquirer's Multiple with empty list."""
        strategy = AcquirersMultipleStrategy()
        results = strategy.calculate([])
        assert results == []

    def test_calculate_acquirers_multiple_invalid_data(
        self,
        invalid_ticker_data: TickerData,
    ) -> None:
        """Test Acquirer's Multiple with invalid data."""
        strategy = AcquirersMultipleStrategy()
        results = strategy.calculate([invalid_ticker_data])
        assert results == []

    def test_get_strategy_name(self) -> None:
        """Test strategy name."""
        strategy = AcquirersMultipleStrategy()
        assert strategy.get_strategy_name() == "Acquirer's Multiple"
