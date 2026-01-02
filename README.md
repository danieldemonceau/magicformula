# Magic Formula Investment Strategy Analysis Tool

A comprehensive Python tool for analyzing stocks using three proven investment strategies:

- **Magic Formula** (Joel Greenblatt): Ranks stocks by Earnings Yield (EBIT/EV) and Return on Capital
- **Acquirer's Multiple** (Tobias Carlisle): Ranks stocks by EV/EBIT ratio (lower is better)
- **Dollar Cost Averaging (DCA)**: Simulates periodic investments with dividend reinvestment and slippage

## Table of Contents

- [Quick Start](#quick-start)
- [Quick Setup Guide](#quick-setup-guide)
- [Installation](#installation)
  - [Using Conda (Recommended)](#using-conda-recommended)
  - [Using Pip](#using-pip)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Using Makefile](#using-makefile)
  - [CSV Input Support](#csv-input-support)
  - [Merging Multiple Strategies](#merging-multiple-strategies)
- [Financial Metrics Explained](#financial-metrics-explained)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Data Sources](#data-sources)
- [Data Quality & Performance](#data-quality--performance)
- [Development](#development)
- [Limitations & Known Issues](#limitations--known-issues)
- [References](#references)

## Quick Start

1. **Install dependencies** (see [Quick Setup Guide](#quick-setup-guide) below)

2. **Prepare symbols file** (`data/symbols.json`):

```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"]
}
```

3. **Run analysis**:

```bash
python scripts/run_analysis.py --strategy magic_formula
```

4. **View results** in `data/outputs/magic_formula_results.csv`

## Quick Setup Guide

### Using Conda (Recommended)

**Step 1: Configure conda** (if you have multiple conda installations):

```bash
# Setup current conda as primary (detects conda base automatically)
make conda-setup-miniconda
```

**Step 2: Create environment**:

```bash
# Production
conda env create -f environment.yml
conda activate magicformula-env

# Development
conda env create -f environment-dev.yml
conda activate magicformula-env-dev
```

**Step 3: Install pip dependencies** (if needed):

```bash
pip install -r requirements.txt  # Production
# or
pip install -r requirements-dev.txt  # Development
```

**Note**: For ESRI projects, switch back with `make conda-setup-esri`.

### Using Pip

```bash
# Production
pip install -r requirements.txt
pip install -e .

# Development
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

## Installation

### Using Conda (Recommended)

Conda is recommended for managing Python environments and dependencies. The project includes two environment files:

- `environment.yml` - Production dependencies
- `environment-dev.yml` - Development dependencies (includes testing, linting, type checking)

**Important**: If you have multiple conda installations (e.g., ESRI and Miniconda), configure conda to use your current conda base first:

```bash
# Automatically detects and configures your conda base
make conda-setup-miniconda

# Or manually
CONDA_BASE=$(conda info --base)
conda config --prepend envs_dirs "$CONDA_BASE/envs"
```

Then create the environment:

```bash
conda env create -f environment.yml
conda activate magicformula-env
```

**Note**: If `yfinance` or other pip dependencies are missing after creating the conda environment, run:

```bash
pip install -r requirements.txt
```

### Using Pip

```bash
pip install -r requirements.txt
pip install -e .
```

After installing with `pip install -e .`, the package is available in your Python path.

## Usage

### Command-Line Interface

**Basic usage**:

```bash
# Magic Formula
python scripts/run_analysis.py --strategy magic_formula

# Acquirer's Multiple
python scripts/run_analysis.py --strategy acquirers_multiple

# DCA with custom parameters
python scripts/run_analysis.py --strategy dca --dca-amount 1000 --dca-frequency monthly
```

**Async mode** (faster for large symbol lists):

```bash
python scripts/run_analysis.py --strategy magic_formula --async \
    --max-concurrent 10 \
    --requests-per-second 3.0
```

**Custom options**:

```bash
python scripts/run_analysis.py \
    --strategy magic_formula \
    --symbols-file data/custom_symbols.json \
    --output results.csv \
    --output-format csv \
    --verbose
```

### Using Makefile

```bash
make run-analysis          # Magic Formula
make run-acquirers         # Acquirer's Multiple
make run-dca              # DCA
make check                # Run all checks (format, lint, type-check)
make fix                  # Auto-fix issues
make test                 # Run tests
make test-cov             # Tests with coverage
```

### CSV Input Support

The tool supports reading CSV files from Magic Formula and Acquirer's Multiple websites, using that data for calculations, and appending results back to the CSV.

**Magic Formula CSV**:

```bash
python scripts/run_analysis.py \
    --strategy magic_formula \
    --csv-input magic_formula.csv
```

**Acquirer's Multiple CSV**:

```bash
python scripts/run_analysis.py \
    --strategy acquirers_multiple \
    --csv-input acquirers_multiple.csv
```

**Auto-detection**: The script automatically detects CSV type by examining column names. Force type with `--csv-type magic_formula` or `--csv-type acquirers_multiple`.

**What gets added**:

- **Magic Formula**: `earnings_yield`, `earnings_yield_rank`, `return_on_capital`, `return_on_capital_rank`, `magic_formula_score`, `quality_score`
- **Acquirer's Multiple**: `acquirers_multiple`, `acquirers_multiple_rank`, `quality_score`

**Note**: The original CSV remains untouched. A new output file is created: `{input_filename}_with_results.csv`.

### Merging Multiple Strategies

Run both strategies and merge results into a single CSV:

```bash
# Using Make (with default paths)
make run-both-strategies

# Or with custom paths
make run-both-strategies \
  MF_CSV=/path/to/magic_formula.csv \
  AM_CSV=/path/to/acquirers_multiple.csv \
  MERGED_OUTPUT=/path/to/combined_results.csv
```

**Manual merge**:

```bash
# Step 1: Run Magic Formula
python scripts/run_analysis.py --strategy magic_formula --csv-input mf.csv --output mf_results.csv

# Step 2: Run Acquirer's Multiple
python scripts/run_analysis.py --strategy acquirers_multiple --csv-input am.csv --output am_results.csv

# Step 3: Merge
python scripts/merge_strategy_results.py \
    --magic-formula mf_results.csv \
    --acquirers-multiple am_results.csv \
    --output combined_results.csv
```

## Financial Metrics Explained

### Magic Formula

Ranks stocks by two metrics:

1. **Earnings Yield** = EBIT / Enterprise Value (higher is better)
2. **Return on Capital** = EBIT / (Net Working Capital + Net Fixed Assets) (higher is better)

**Final Score**: Sum of both ranks (lower is better)

### Acquirer's Multiple

Ranks stocks by **EV/EBIT Ratio** = Enterprise Value / EBIT (lower is better)

### Dollar Cost Averaging

Simulates periodic investments with:

- Configurable investment amount and frequency
- Dividend reinvestment support
- Slippage modeling (configurable basis points)
- Returns total invested, total value, and return percentage

## Project Structure

```
magicformula/
├── src/
│   ├── data/              # Data fetching, models, validation
│   ├── strategies/        # Investment strategies
│   ├── calculations/      # Financial metrics
│   ├── output/            # Output writers (CSV, JSON)
│   └── utils/             # Utilities
├── config/                # Configuration
├── scripts/                # CLI entry points
├── tests/                  # Test suite
├── data/                   # Input symbols and outputs
├── Makefile               # Build automation
├── environment.yml        # Conda production environment
└── environment-dev.yml    # Conda development environment
```

## Configuration

Configuration is managed via Pydantic Settings. Create a `.env` file to customize:

```bash
# Performance
ASYNC_MAX_CONCURRENT=5
ASYNC_REQUESTS_PER_SECOND=2.0

# Caching
CACHE_ENABLED=true
CACHE_EXPIRATION_HOURS=24

# Data Quality
STALENESS_THRESHOLD_DAYS=1
MIN_QUALITY_SCORE=0.5

# Alpha Vantage (optional fallback)
AV_APIKEY=your_api_key_here
```

All settings can be overridden via environment variables or CLI arguments.

## Data Sources

### Primary: Yahoo Finance (yfinance)

- Stock prices and historical data
- Market capitalization
- Financial statements (EBIT, debt, cash)
- Fundamental ratios

### Fallback: Alpha Vantage

Optional fallback for missing financial data (requires API key in `.env`):

- Balance sheet data (for Net Working Capital, Net Fixed Assets)
- Income statement data (for EBIT)
- Pre-calculated ratios (Return on Capital)

### Manual Data Downloads

For production use, consider downloading Magic Formula and Acquirer's Multiple data manually:

- **Magic Formula**: [magicformulainvesting.com](https://www.magicformulainvesting.com/)
- **Acquirer's Multiple**: [acquirersmultiple.com](https://www.acquirersmultiple.com/)

The tool supports CSV input from these sources (see [CSV Input Support](#csv-input-support)).

## Data Quality & Performance

### Data Quality Features

- **Outlier Detection**: Flags suspicious metrics (e.g., P/E > 1000)
- **Staleness Checks**: Warns when data is older than 1 day
- **Quality Scoring**: Each ticker gets a score (0-1) based on completeness, outliers, freshness, and status
- **Status Tracking**: ACTIVE, INACTIVE, DELISTED, STALE, DATA_UNAVAILABLE

### Performance Optimizations

- **Async Fetching**: Concurrent API calls (default: 5 concurrent, 2 req/sec) - 3-5x faster than sequential
- **SQLite Caching**: Optional caching layer reduces redundant API calls (24-hour expiration, configurable)
- **Batch Price Fetching**: Uses `yf.download()` for efficient bulk price data retrieval

## Development

### Running Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/unit/ -v               # Unit tests only
pytest tests/ --cov=src --cov-report=html  # With coverage
```

### Code Quality

```bash
make format      # Format code
make lint        # Lint code
make lint-fix    # Fix linting issues
make type-check  # Type checking
make check       # Run all checks
make fix         # Auto-fix all issues
```

### Adding a New Strategy

1. Create strategy class in `src/strategies/` extending `BaseStrategy`
2. Register in `config/settings_pydantic.py`
3. Add factory method in `scripts/run_analysis.py`
4. Write tests in `tests/unit/test_strategies.py`

## Limitations & Known Issues

1. **Data Availability**: yfinance may not have complete financial data for all tickers (EBIT may fall back to EBITDA; NWC/NFA may be missing)
2. **Rate Limiting**: YFinance has rate limits. Use async mode and caching for large symbol lists
3. **Manual Data**: For production accuracy, consider using manually downloaded CSV data from official websites

## References

- **The Little Book That Beats the Market** by Joel Greenblatt
- **Deep Value** by Tobias Carlisle
- **The Acquirer's Multiple** by Tobias Carlisle

## License

See [LICENSE](LICENSE) file for details.
