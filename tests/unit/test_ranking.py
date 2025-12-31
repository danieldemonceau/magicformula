"""Unit tests for ranking utilities."""

from src.calculations.ranking import calculate_combined_rank, rank_series


class TestRankSeries:
    """Tests for rank_series function."""

    def test_rank_series_ascending(self) -> None:
        """Test ranking in ascending order."""
        data = [
            {"symbol": "A", "value": 10},
            {"symbol": "B", "value": 20},
            {"symbol": "C", "value": 30},
        ]
        result = rank_series(data, "value", ascending=True)
        assert result[0]["value_rank"] == 1
        assert result[1]["value_rank"] == 2
        assert result[2]["value_rank"] == 3

    def test_rank_series_descending(self) -> None:
        """Test ranking in descending order."""
        data = [
            {"symbol": "A", "value": 10},
            {"symbol": "B", "value": 20},
            {"symbol": "C", "value": 30},
        ]
        result = rank_series(data, "value", ascending=False)
        assert result[0]["value_rank"] == 3
        assert result[1]["value_rank"] == 2
        assert result[2]["value_rank"] == 1

    def test_rank_series_empty(self) -> None:
        """Test ranking empty list."""
        result = rank_series([], "value")
        assert result == []

    def test_rank_series_ties(self) -> None:
        """Test ranking with tied values."""
        data = [
            {"symbol": "A", "value": 10},
            {"symbol": "B", "value": 10},
            {"symbol": "C", "value": 20},
        ]
        result = rank_series(data, "value", ascending=False, method="min")
        assert result[0]["value_rank"] == 1
        assert result[1]["value_rank"] == 1
        assert result[2]["value_rank"] == 3


class TestCalculateCombinedRank:
    """Tests for calculate_combined_rank function."""

    def test_calculate_combined_rank_basic(self) -> None:
        """Test basic combined rank calculation."""
        data = [
            {"symbol": "A", "rank1": 1, "rank2": 2},
            {"symbol": "B", "rank1": 2, "rank2": 1},
            {"symbol": "C", "rank1": 3, "rank2": 3},
        ]
        result = calculate_combined_rank(data, ["rank1", "rank2"], "combined")
        assert result[0]["combined"] == 3
        assert result[1]["combined"] == 3
        assert result[2]["combined"] == 6

    def test_calculate_combined_rank_empty(self) -> None:
        """Test combined rank with empty list."""
        result = calculate_combined_rank([], ["rank1", "rank2"])
        assert result == []

    def test_calculate_combined_rank_missing_keys(self) -> None:
        """Test combined rank with missing rank keys."""
        data = [
            {"symbol": "A", "rank1": 1},
            {"symbol": "B", "rank2": 2},
        ]
        result = calculate_combined_rank(data, ["rank1", "rank2"], "combined")
        assert result[0]["combined"] == 1
        assert result[1]["combined"] == 2

