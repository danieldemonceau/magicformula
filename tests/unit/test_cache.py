"""Unit tests for caching layer."""

import tempfile
from pathlib import Path

import pytest

from src.data.cache import TickerCache
from src.data.models.ticker_data import TickerData
from src.data.models.ticker_status import TickerStatus


class TestTickerCache:
    """Tests for TickerCache."""

    @pytest.fixture
    def cache(self) -> TickerCache:
        """Create a temporary cache for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.db"
            yield TickerCache(cache_path=str(cache_path))

    def test_get_set(self, cache: TickerCache) -> None:
        """Test basic get/set operations."""
        ticker = TickerData(
            symbol="AAPL",
            price=150.0,
            status=TickerStatus.ACTIVE,
        )

        # Should not be in cache
        assert cache.get("AAPL") is None

        # Set in cache
        cache.set("AAPL", ticker)

        # Should be in cache
        cached = cache.get("AAPL")
        assert cached is not None
        assert cached.symbol == "AAPL"
        assert cached.price == 150.0

    def test_clear_symbol(self, cache: TickerCache) -> None:
        """Test clearing specific symbol."""
        ticker = TickerData(symbol="AAPL", price=150.0)
        cache.set("AAPL", ticker)
        cache.clear("AAPL")
        assert cache.get("AAPL") is None

    def test_clear_all(self, cache: TickerCache) -> None:
        """Test clearing all entries."""
        cache.set("AAPL", TickerData(symbol="AAPL", price=150.0))
        cache.set("MSFT", TickerData(symbol="MSFT", price=300.0))
        cache.clear_all()
        assert cache.get("AAPL") is None
        assert cache.get("MSFT") is None
