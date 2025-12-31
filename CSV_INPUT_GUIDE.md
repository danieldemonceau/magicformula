# CSV Input Guide

## Overview

The script now supports reading CSV files from the Magic Formula and Acquirer's Multiple websites, using that data for calculations, and appending the results back to the CSV.

## Supported CSV Formats

### 1. Magic Formula CSV

**Expected columns:**
- `Company Name (in alphabetical order)` or `Company Name` or `Name`
- `Ticker` (required)
- `Market Cap ($ Millions)` or `Market Cap`

**Example:**
```csv
Company Name (in alphabetical order),Ticker,Market Cap ($ Millions),Price From,Most Recent Quarter Data
Altria Group Inc,MO,"97,004.61",30-Dec,30-Sep
AMC Networks Inc,AMCX,422.82,30-Dec,30-Sep
```

### 2. Acquirer's Multiple CSV

**Expected columns:**
- `Ticker` (required)
- `Price ($)` or `Price`
- `Mkt Cap ($M)` or `Mkt Cap` or `Market Cap`
- `EV ($M)` or `EV` or `Enterprise Value`
- `OI ($M)` or `OI` or `Operating Income` (EBIT)
- `Industry`

**Example:**
```csv
"Ticker","Name","Price ($)","Change (%)","Industry","Acquirer's Multiple®","FCF Yield (%)","E(r) (%)","IV/P","ROA (5YrAvg %)","Incr. Growth (%)","SHYield (%)","BB Yield (%)","Div Yield (%)","Mkt Cap ($M)","EV ($M)","Debt:Equity (%)","OI ($M)"
"SYF","Synchrony Financial","84.17","-0.25","Specialty Finance","2.80","32.16","17.75","1.60","9","0","8.51","6.86","1.65","30,313.00","29,721.00","-5.98","10,800.00"
```

## Usage

### Basic Usage

```bash
# Magic Formula CSV input
python scripts/run_analysis.py --strategy magic_formula --csv-input magic_formula.csv

# Acquirer's Multiple CSV input
python scripts/run_analysis.py --strategy acquirers_multiple --csv-input acquirers_multiple.csv
```

### Auto-Detection

The script automatically detects the CSV type by examining column names:

```bash
# Auto-detect CSV type
python scripts/run_analysis.py --strategy magic_formula --csv-input data.csv --csv-type auto
```

### Manual Type Specification

If auto-detection fails, specify the type manually:

```bash
# Force Magic Formula format
python scripts/run_analysis.py --strategy magic_formula --csv-input data.csv --csv-type magic_formula

# Force Acquirer's Multiple format
python scripts/run_analysis.py --strategy acquirers_multiple --csv-input data.csv --csv-type acquirers_multiple
```

## How It Works

1. **Read CSV**: Parses the CSV file and extracts available data (ticker, market cap, EBIT, EV, etc.)

2. **Fetch Missing Data**: For fields not in the CSV, fetches from yfinance API to complete the dataset

3. **Calculate Metrics**: 
   - Calculates missing metrics (earnings_yield, return_on_capital, acquirers_multiple)
   - Runs the selected strategy (Magic Formula or Acquirer's Multiple)

4. **Create Output**: Merges calculated results with input CSV and writes to a **new output file** (input CSV remains untouched)

## What Gets Added to CSV

The script adds the following calculated columns to your CSV:

### For Magic Formula:
- `earnings_yield` - EBIT / Enterprise Value
- `earnings_yield_rank` - Rank (lower is better)
- `return_on_capital` - EBIT / (Net Working Capital + Net Fixed Assets)
- `return_on_capital_rank` - Rank (lower is better)
- `magic_formula_score` - Combined rank (lower is better)
- `quality_score` - Data quality score (0-1)

### For Acquirer's Multiple:
- `acquirers_multiple` - EV / EBIT (if not already in CSV)
- `acquirers_multiple_rank` - Rank (lower is better)
- `quality_score` - Data quality score (0-1)

## Data Merging Strategy

- **CSV data takes precedence**: If a field exists in both CSV and yfinance, CSV value is used
- **yfinance fills gaps**: Missing fields are fetched from yfinance
- **Calculations**: Missing calculated metrics are computed from available base data

## Example Workflow

1. Download CSV from website (Magic Formula or Acquirer's Multiple)
2. Save to your project directory (e.g., `data/magic_formula.csv`)
3. Run script:
   ```bash
   python scripts/run_analysis.py --strategy magic_formula --csv-input data/magic_formula.csv
   ```
4. A new file is created: `data/magic_formula_with_results.csv` with all original columns plus calculated results

### Custom Output Path

You can specify a custom output path:

```bash
python scripts/run_analysis.py --strategy magic_formula --csv-input data/magic_formula.csv --output data/enhanced_results.csv
```

## Notes

- The **original CSV file remains untouched** - a new output file is created
- Default output filename: `{input_filename}_with_results.csv`
- CSV data is preferred over yfinance data when both are available
- Missing fields are automatically fetched from yfinance
- All original CSV columns are preserved in the output

