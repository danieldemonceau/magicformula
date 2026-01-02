"""Type aliases for data structures."""

from datetime import datetime

from src.data.models.ticker_status import TickerStatus

# Type for yfinance ticker.info dictionary
YFinanceInfoDict = dict[str, str | int | float | bool | None]

# Type for financial data extracted from yfinance
FinancialDataDict = dict[str, float | None | str | bool]

# Type for strategy result dictionaries (contains symbol, metrics, ranks, scores)
# Can include TickerData fields which may have datetime, bool, TickerStatus
StrategyResultDict = dict[str, str | int | float | bool | None | datetime | TickerStatus]

# Type for TickerData.to_dict() output
TickerDataDict = dict[str, str | int | float | bool | None | datetime | TickerStatus]
