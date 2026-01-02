"""Unit tests for data quality assessment."""

from datetime import datetime, timedelta

from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus
from src.data.quality import (
    assess_data_quality,
    calculate_quality_score,
    check_staleness,
    detect_outliers,
)


class TestOutlierDetection:
    """Tests for outlier detection."""

    def test_detect_outliers_pe_ratio(self) -> None:
        """Test P/E ratio outlier detection."""
        ticker = TickerData(
            symbol="TEST",
            earnings_yield=0.0001,  # P/E = 10,000 (outlier)
        )
        outliers = detect_outliers(ticker)
        assert "pe_ratio" in outliers

    def test_detect_outliers_price_to_sales(self) -> None:
        """Test P/S ratio outlier detection."""
        ticker = TickerData(
            symbol="TEST",
            price_to_sales=2000.0,  # Outlier
        )
        outliers = detect_outliers(ticker)
        assert "price_to_sales" in outliers

    def test_detect_outliers_no_outliers(self) -> None:
        """Test with no outliers."""
        ticker = TickerData(
            symbol="TEST",
            price=100.0,
            earnings_yield=0.05,
            price_to_sales=2.0,
        )
        outliers = detect_outliers(ticker)
        assert len(outliers) == 0


class TestStalenessCheck:
    """Tests for staleness checking."""

    def test_check_staleness_fresh(self) -> None:
        """Test with fresh data."""
        ticker = TickerData(
            symbol="TEST",
            data_timestamp=datetime.now(),
        )
        assert not check_staleness(ticker)

    def test_check_staleness_stale(self) -> None:
        """Test with stale data."""
        ticker = TickerData(
            symbol="TEST",
            data_timestamp=datetime.now() - timedelta(days=2),
        )
        assert check_staleness(ticker)

    def test_check_staleness_no_timestamp(self) -> None:
        """Test with no timestamp."""
        ticker = TickerData(symbol="TEST")
        assert check_staleness(ticker)


class TestQualityScore:
    """Tests for quality score calculation."""

    def test_calculate_quality_score_perfect(self) -> None:
        """Test perfect quality score."""
        ticker = TickerData(
            symbol="TEST",
            status=TickerStatus.ACTIVE,
            price=100.0,
            market_cap=1_000_000_000,
            ebit=100_000_000,
            enterprise_value=1_100_000_000,
            data_timestamp=datetime.now(),
        )
        score = calculate_quality_score(ticker)
        assert score >= 0.8

    def test_calculate_quality_score_missing_data(self) -> None:
        """Test quality score with missing data."""
        ticker = TickerData(
            symbol="TEST",
            status=TickerStatus.ACTIVE,
            price=None,
            market_cap=None,
        )
        score = calculate_quality_score(ticker)
        assert score < 0.5

    def test_calculate_quality_score_inactive(self) -> None:
        """Test quality score for inactive ticker."""
        ticker = TickerData(
            symbol="TEST",
            status=TickerStatus.DELISTED,
            price=100.0,
        )
        score = calculate_quality_score(ticker)
        assert score < 0.7


class TestAssessDataQuality:
    """Tests for data quality assessment."""

    def test_assess_data_quality_delisted(self) -> None:
        """Test assessment for delisted ticker."""
        ticker = TickerData(
            symbol="TEST",
            price=None,
        )
        result = assess_data_quality(ticker)
        assert result.status == TickerStatus.DELISTED
        assert result.quality_score is not None

    def test_assess_data_quality_stale(self) -> None:
        """Test assessment for stale data."""
        ticker = TickerData(
            symbol="TEST",
            price=100.0,
            data_timestamp=datetime.now() - timedelta(days=2),
        )
        result = assess_data_quality(ticker)
        assert result.status == TickerStatus.STALE

    def test_assess_data_quality_sets_timestamp(self) -> None:
        """Test that timestamp is set if missing."""
        ticker = TickerData(symbol="TEST", price=100.0)
        result = assess_data_quality(ticker)
        assert result.data_timestamp is not None
