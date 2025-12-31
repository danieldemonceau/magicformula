# Magic Formula Investment Strategy Analysis Tool

A comprehensive Python tool for analyzing stocks using three proven investment strategies:

- **Magic Formula** (Joel Greenblatt): Ranks stocks by Earnings Yield (EBIT/EV) and Return on Capital
- **Acquirer's Multiple** (Tobias Carlisle): Ranks stocks by EV/EBIT ratio (lower is better)
- **Dollar Cost Averaging (DCA)**: Simulates periodic investments with dividend reinvestment and slippage

## Features

- ✅ **Corrected Financial Metrics**: Properly implements EBIT/Enterprise Value and Return on Capital
- ✅ **Modular Architecture**: Clean separation of data fetching, calculations, and reporting
- ✅ **Strategy Pattern**: Easy to extend with new investment strategies
- ✅ **Comprehensive Testing**: Unit tests and integration tests included
- ✅ **Type Safety**: Full type hints with mypy validation
- ✅ **CLI Interface**: User-friendly command-line interface
- ✅ **Multiple Output Formats**: CSV and JSON output support
- ✅ **Error Handling**: Robust error handling for missing data and API failures
- ✅ **Async Fetching**: High-performance concurrent data fetching with rate limiting
- ✅ **Data Quality Assessment**: Outlier detection, staleness checks, and quality scoring
- ✅ **SQLite Caching**: Optional caching layer to reduce API calls
- ✅ **Ticker Status Tracking**: ACTIVE, INACTIVE, DELISTED, STALE, DATA_UNAVAILABLE statuses
- ✅ **Pydantic Settings**: Environment-based configuration management
- ✅ **Explicit Data Filtering**: Automatic exclusion of invalid/None data from rankings

## Installation

### Using Conda (Recommended)

**Important**: If you have multiple conda installations (e.g., ESRI and Miniconda), configure conda to use your current conda base first:

```bash
# Setup current conda as primary (keeps ESRI paths for later use)
# This automatically detects your conda base using: conda info --base
make conda-setup-miniconda

# Or manually (replace with your actual conda base path):
CONDA_BASE=$(conda info --base)
conda config --prepend envs_dirs "$CONDA_BASE/envs"
```

Then create the environment:

```bash
# Production environment
conda env create -f environment.yml
conda activate magicformula-env
# Verify installation (optional)
pip list | grep yfinance

# Development environment
conda env create -f environment-dev.yml
conda activate magicformula-env-dev
# Verify installation (optional)
pip list | grep yfinance
```

**Note**: If `yfinance` or other pip dependencies are missing after creating the conda environment, run:

```bash
pip install -r requirements.txt  # For production
# or
pip install -r requirements-dev.txt  # For development
```

**For ESRI Projects**: When working on ESRI projects, switch back with `make conda-setup-esri` (see `CONDA_ENVS_DIRS.md` for details).

### Using Pip

```bash
# Production
pip install -r requirements.txt
pip install -e .

# Development
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

**Note**: After installing with `pip install -e .`, the package will be available in your Python path and you can run scripts directly.

## Quick Start

1. **Prepare your symbols file** (`data/symbols.json`):

```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"]
}
```

2. **Run Magic Formula analysis**:

```bash
python scripts/run_analysis.py --strategy magic_formula
```

3. **View results** in `data/outputs/magic_formula_results.csv`

## Usage

### Command-Line Interface

```bash
# Magic Formula (sync mode)
python scripts/run_analysis.py --strategy magic_formula

# Magic Formula (async mode - faster for large symbol lists)
python scripts/run_analysis.py --strategy magic_formula --async

# Acquirer's Multiple with custom async settings
python scripts/run_analysis.py \
    --strategy acquirers_multiple \
    --async \
    --max-concurrent 10 \
    --requests-per-second 3.0

# Dollar Cost Averaging
python scripts/run_analysis.py --strategy dca --dca-amount 1000 --dca-frequency monthly

# Custom symbols file and output
python scripts/run_analysis.py \
    --strategy magic_formula \
    --symbols-file data/custom_symbols.json \
    --output results.csv \
    --output-format csv

# Disable caching
python scripts/run_analysis.py --strategy magic_formula --no-cache

# Verbose logging
python scripts/run_analysis.py --strategy magic_formula --verbose
```

### Using Makefile

```bash
# Run Magic Formula analysis
make run-analysis

# Run Acquirer's Multiple analysis
make run-acquirers

# Run DCA analysis
make run-dca

# Run all checks (format, lint, type-check)
make check

# Auto-fix issues
make fix

# Run tests
make test

# Run tests with coverage
make test-cov
```

## Project Structure

```
magicformula/
├── src/
│   ├── data/
│   │   ├── fetchers/          # Data fetching (YFinance, etc.)
│   │   ├── models/            # Pydantic data models
│   │   └── validators.py      # Data validation
│   ├── strategies/            # Investment strategies
│   │   ├── magic_formula.py
│   │   ├── acquirers_multiple.py
│   │   └── dca.py
│   ├── calculations/          # Financial metrics
│   │   ├── financial_metrics.py
│   │   └── ranking.py
│   ├── output/                # Output writers (CSV, JSON)
│   └── utils/                 # Utilities (logging, dates, decorators)
├── config/                    # Configuration
├── scripts/                    # CLI entry point
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── data/
│   ├── symbols.json           # Input symbols
│   └── outputs/               # Generated results
├── Makefile                   # Build automation
├── environment.yml             # Conda production environment
├── environment-dev.yml        # Conda development environment
└── README.md
```

## Financial Metrics Explained

### Magic Formula

The Magic Formula ranks stocks by two metrics:

1. **Earnings Yield** = EBIT / Enterprise Value

   - Higher is better
   - Measures how much earnings a company generates relative to its purchase price

2. **Return on Capital** = EBIT / (Net Working Capital + Net Fixed Assets)
   - Higher is better
   - Measures how efficiently a company uses its capital

**Final Score**: Sum of both ranks (lower is better)

### Acquirer's Multiple

The Acquirer's Multiple ranks stocks by:

- **EV/EBIT Ratio** = Enterprise Value / EBIT
  - Lower is better
  - Measures how many years of earnings it would take to pay back the purchase price

### Dollar Cost Averaging

Simulates periodic investments over time:

- Configurable investment amount and frequency
- Dividend reinvestment support
- Slippage modeling (configurable basis points)
- Returns total invested, total value, and return percentage

## Data Quality & Performance

### Data Quality Features

- **Outlier Detection**: Automatically flags suspicious metrics (e.g., P/E > 1000, impossible market caps)
- **Staleness Checks**: Warns when data is older than 1 day (configurable)
- **Quality Scoring**: Each ticker gets a quality score (0-1) based on:
  - Completeness of required fields
  - Presence of outliers
  - Data freshness
  - Ticker status
- **Status Tracking**: Tickers are classified as:
  - `ACTIVE`: Valid data available
  - `INACTIVE`: Ticker exists but data incomplete
  - `DELISTED`: Ticker no longer trades
  - `STALE`: Data is too old
  - `DATA_UNAVAILABLE`: Critical data missing

### Performance Optimizations

- **Async Fetching**: Concurrent API calls with configurable rate limiting
  - Default: 5 concurrent requests, 2 requests/second
  - Can be tuned via `--max-concurrent` and `--requests-per-second`
  - 3-5x faster than sequential fetching for large symbol lists
- **SQLite Caching**: Optional caching layer reduces redundant API calls
  - Cache expiration: 24 hours (configurable)
  - Automatic cache cleanup of expired entries
  - Disable with `--no-cache` flag

## Data Sources

### Current Implementation

- **Yahoo Finance (yfinance)**: Primary data source for:
  - Stock prices and historical data
  - Market capitalization
  - Financial statements (EBIT, debt, cash)
  - Fundamental ratios

### Manual Data Downloads

**Note**: For production use, you may need to download Magic Formula and Acquirer's Multiple data manually from their respective websites:

- **Magic Formula**: Data available from [magicformulainvesting.com](https://www.magicformulainvesting.com/)
- **Acquirer's Multiple**: Data available from [acquirersmultiple.com](https://www.acquirersmultiple.com/)

The current implementation uses yfinance as a proxy, but for maximum accuracy, consider integrating manually downloaded data.

### Alternative APIs (Future Enhancements)

For production use, consider these APIs:

- **Alpha Vantage**: Free tier available, requires API key
- **Polygon.io**: Real-time and historical data, WebSocket support
- **Nasdaq Data Link (Quandl)**: Comprehensive financial datasets
- **Financial Modeling Prep**: Free tier with financial statements

## Configuration

Configuration is managed via Pydantic Settings, which supports:

1. **Environment Variables**: Set variables in `.env` file or environment
2. **Default Values**: Sensible defaults in `config/settings_pydantic.py`

### Key Configuration Options

Create a `.env` file (see `.env.example`) to customize:

```bash
# Performance
ASYNC_MAX_CONCURRENT=5          # Max concurrent async requests
ASYNC_REQUESTS_PER_SECOND=2.0   # Rate limit

# Caching
CACHE_ENABLED=true               # Enable SQLite caching
CACHE_EXPIRATION_HOURS=24        # Cache expiration

# Data Quality
STALENESS_THRESHOLD_DAYS=1       # Max age for fresh data
MIN_QUALITY_SCORE=0.5            # Minimum quality for rankings

# Outlier Detection
OUTLIER_PE_RATIO=1000.0          # Max P/E ratio before flagging
OUTLIER_PRICE_TO_SALES=1000.0    # Max P/S ratio before flagging
```

All settings can be overridden via environment variables or CLI arguments.

## Development

### Running Tests

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Fix linting issues
make lint-fix

# Type checking
make type-check

# Run all checks
make check

# Auto-fix all issues
make fix
```

### Adding a New Strategy

1. Create a new strategy class in `src/strategies/`:

```python
from src.strategies.base_strategy import BaseStrategy
from src.data.models.ticker_data import TickerData

class MyNewStrategy(BaseStrategy):
    def calculate(self, ticker_data: list[TickerData]) -> list[dict]:
        # Your calculation logic
        pass

    def get_strategy_name(self) -> str:
        return "My New Strategy"
```

2. Register it in `config/settings.py`:

```python
STRATEGY_MY_NEW = "my_new"
SUPPORTED_STRATEGIES.append(STRATEGY_MY_NEW)
```

3. Add factory method in `scripts/run_analysis.py`:

```python
if strategy_name == STRATEGY_MY_NEW:
    return MyNewStrategy()
```

4. Write tests in `tests/unit/test_strategies.py`

## Limitations & Known Issues

1. **Data Availability**: yfinance may not have complete financial data for all tickers, especially:

   - EBIT (may fall back to EBITDA)
   - Net Working Capital
   - Net Fixed Assets

2. **Rate Limiting**: YFinance has rate limits. The tool includes sleep delays, but for large symbol lists, consider:

   - Using paid APIs
   - Implementing caching
   - Using async/concurrent fetching (future enhancement)

3. **Manual Data**: For production accuracy, Magic Formula and Acquirer's Multiple data should be downloaded manually from their official websites.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make check` to ensure code quality
5. Write tests for new functionality
6. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

## References

- **The Little Book That Beats the Market** by Joel Greenblatt
- **Deep Value** by Tobias Carlisle
- **The Acquirer's Multiple** by Tobias Carlisle

## Acknowledgments

- Joel Greenblatt for the Magic Formula methodology
- Tobias Carlisle for the Acquirer's Multiple methodology
- The yfinance library maintainers
