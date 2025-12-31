from xml.dom import INVALID_MODIFICATION_ERR
import requests
from requests import exceptions
from datetime import date, timedelta, datetime, timezone
import time
import os
from dotenv import load_dotenv
import sys
import time
from dateutil.relativedelta import relativedelta
import finviz
import json
import csv
from operator import itemgetter
import yfinance as yf

headers = [
    "Symbol",
    "Price",
    "Price index (6month)",
    "Price index (12month)",
    "Book/Market",
    "Free Cash Flow Yield (12months)",
    "Price/Sales",
    "Earnings Yield (12months)",
    "ROA",
    "Earnings Yield (12months) rank",
    "ROA rank",
    "Magic Formula Score",
]


def quotes_get(symbols, period_name, source_name):
    quote_data = []
    time_sleep = 1

    print("--- ---")
    print("--- QUOTE EXTRACTION -- START ---")
    print("--- ---")

    for symbol in symbols:
        print(f"--- PROCESSING {symbol} -- START ---")
        count = 0

        quote_price = 0.0
        quote_price_index_6month = 0.0
        quote_price_index_12month = 0.0
        quote_bm = 0.0
        quote_fcfy_12months = 0.0
        quote_ps = 0.0
        quote_ey_12months = 0.0
        quote_roa = 0.0

        ticker = yf.Ticker(symbol)
        today = date.today()
        weekday_last = today - timedelta(days=1)
        while weekday_last.weekday() > 4:
            weekday_last -= timedelta(days=1)
        weekday_minus_6months = weekday_last + relativedelta(months=-6)
        while weekday_minus_6months.weekday() > 4:
            weekday_minus_6months -= timedelta(days=1)
        weekday_minus_12months = weekday_last + relativedelta(months=-12)
        while weekday_minus_12months.weekday() > 4:
            weekday_minus_12months -= timedelta(days=1)

        try:
            hist = ticker.history(
                start=weekday_minus_12months.strftime("%Y-%m-%d"),
                end=(weekday_last + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            last_price_row = hist.loc[hist.index.date == weekday_last]
            if not last_price_row.empty:
                quote_price = float(last_price_row["Close"].iloc[0])
            else:
                quote_price = -1.0
            price_6m_row = hist.loc[hist.index.date == weekday_minus_6months]
            if not price_6m_row.empty:
                quote_price_6months = float(price_6m_row["Close"].iloc[0])
            else:
                quote_price_6months = -1.0
            price_12m_row = hist.loc[hist.index.date == weekday_minus_12months]
            if not price_12m_row.empty:
                quote_price_12months = float(price_12m_row["Close"].iloc[0])
            else:
                quote_price_12months = float(hist["Close"].iloc[0])
            quote_price_index_6month = (
                float(quote_price) / float(quote_price_6months)
                if quote_price_6months > 0
                else 0
            )
            quote_price_index_12month = (
                float(quote_price) / float(quote_price_12months)
                if quote_price_12months > 0
                else 0
            )
        except Exception as e:
            print(f"Error fetching price data for {symbol}: {e}")
            quote_price = 0.0
            quote_price_index_6month = 0.0
            quote_price_index_12month = 0.0

        try:
            info = ticker.info
            quote_bm = (
                1.0 / float(info["priceToBook"])
                if info.get("priceToBook") and info["priceToBook"] > 0
                else 0.0
            )
            if (
                info.get("freeCashflow")
                and info.get("marketCap")
                and info["marketCap"] > 0
            ):
                quote_fcfy_12months = float(info["freeCashflow"]) / float(
                    info["marketCap"]
                )
            else:
                quote_fcfy_12months = 0.0
            quote_ps = (
                float(info["priceToSalesTrailing12Months"])
                if info.get("priceToSalesTrailing12Months")
                else 0.0
            )
            quote_ey_12months = (
                1.0 / float(info["trailingPE"])
                if info.get("trailingPE") and info["trailingPE"] > 0
                else 0.0
            )
            quote_roa = (
                float(info["returnOnAssets"]) * 100
                if info.get("returnOnAssets")
                else 0.0
            )
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
            quote_bm = 0.0
            quote_fcfy_12months = 0.0
            quote_ps = 0.0
            quote_ey_12months = 0.0
            quote_roa = 0.0

        quote_data.append(
            {
                "Symbol": symbol,
                "Price": quote_price,
                "Price index (6month)": quote_price_index_6month,
                "Price index (12month)": quote_price_index_12month,
                "Book/Market": quote_bm,
                "Free Cash Flow Yield (12months)": quote_fcfy_12months,
                "Price/Sales": quote_ps,
                "Earnings Yield (12months)": quote_ey_12months,
                "ROA": quote_roa,
            }
        )
        print(f"--- PROCESSING {symbol} -- END ---")
        if not symbol == symbols[-1]:
            time.sleep(time_sleep)

    print(quote_data)
    print("--- ---")
    print("--- QUOTE EXTRACTION -- END ---")
    print("--- ---")
    print("--- Rank based on Earnings Yield (12months) -- START ---")
    quote_data = sorted(
        quote_data, key=itemgetter("Earnings Yield (12months)"), reverse=True
    )
    for idx, quote_data_item in enumerate(quote_data):
        quote_data[idx]["Earnings Yield (12months) rank"] = idx + 1

    print("--- Rank based on Earnings Yield (12months) -- END ---")

    print("--- Rank based on ROA -- START ---")
    quote_data = sorted(quote_data, key=itemgetter("ROA"), reverse=True)
    for idx, quote_data_item in enumerate(quote_data):
        quote_data[idx]["ROA rank"] = idx + 1
    print("--- Rank based on ROA -- END ---")

    print(
        "--- Calculate the Magic Formula score and Rank based on this score -- START ---"
    )
    for idx, quote_data_item in enumerate(quote_data):
        quote_data[idx]["Magic Formula Score"] = (
            quote_data[idx]["Earnings Yield (12months) rank"]
            + quote_data[idx]["ROA rank"]
        )
    quote_data = sorted(quote_data, key=itemgetter("Magic Formula Score"))
    print(
        "--- Calculate the Magic Formula score and Rank based on this score -- END ---"
    )

    keys = quote_data[0].keys()

    print("--- Writing to csv -- START ---")
    with open("quotes.csv", "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(quote_data)
    print("--- Writing to csv -- END ---")


if __name__ == "__main__":
    time_start = time.time()
    print(
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
    )
    print(" - - - FUNCTION START - - - ")
    print(" - - - REFRESH QUOTES - - - ")
    print(
        " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )

    with open("symbols.json") as json_data_file:
        symbols_json = json.load(json_data_file)

    symbols = symbols_json["symbols"]

    period_name = "1day"
    source_name = "alphavantage"
    if len(sys.argv) > 1:
        symbols = sys.argv[1]
    if len(sys.argv) > 2:
        period_name = sys.argv[2]
    if len(sys.argv) > 3:
        source_name = sys.argv[3]
    quotes_get(symbols, period_name, source_name)
    time_end = time.time()
    print(
        "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
    )
    print(f" - - - Duration: {time_end - time_start} - - - ")
    print(" - - - FUNCTION FINISHED - - - ")
    print(" - - - REFRESH QUOTES - - - ")
    print(
        " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
