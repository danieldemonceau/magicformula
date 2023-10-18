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
    # Get configs
    load_dotenv()
    av_apikey = os.getenv("AV_APIKEY")
    quote_data = []

    # This is because of the limitation of 5 requests / minute, thus 12 seconds wait between API Calls
    time_sleep = 5
    # print("\t".join(headers))

    print("--- ---")
    print("--- QUOTE EXTRACTION -- START ---")
    print("--- ---")

    shouldContinue = True

    for symbol in symbols:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={av_apikey}"
        print(f"--- PROCESSING {symbol} -- START ---")
        print(f"--- PROCESSING {url} -- START ---")
        count = 0

        quote_price = 0.0
        quote_price_index_6month = 0.0
        quote_price_index_12month = 0.0
        quote_bm = 0.0
        quote_fcfy_12months = 0.0
        quote_ps = 0.0
        quote_ey_12months = 0.0
        quote_roa = 0.0

        if not shouldContinue:
            break
        while count < 3:
            try:
                count += 1
                response = requests.get(url)
                quote = response.json()
                if "Error Message" in quote:
                    quote_price = 0.0
                    quote_price_index_6month = 0.0
                    quote_price_index_12month = 0.0
                    print("error message in quote", quote["Error Message"])
                    break
                elif "Meta Data" in quote:
                    ###################################################
                    # Get last price
                    weekday_last = date.today()
                    weekday_last -= timedelta(days=1)
                    while weekday_last.weekday() > 4:  # Mon-Fri are 0-4
                        weekday_last -= timedelta(days=1)

                    weekday_last_year = f"{weekday_last.year:02}"
                    weekday_last_month = f"{weekday_last.month:02}"
                    weekday_last_day = f"{weekday_last.day:02}"
                    # weekday_last_year = "2022"
                    # weekday_last_month = "10"
                    # weekday_last_day = "28"
                    if (
                        f"{weekday_last_year}-{weekday_last_month}-{weekday_last_day}"
                        in quote[f"Time Series (Daily)"]
                    ):
                        quote_price = float(
                            quote[f"Time Series (Daily)"][
                                f"{weekday_last_year}-{weekday_last_month}-{weekday_last_day}"
                            ][f"4. close"]
                        )
                    else:
                        quote_price = -1.0
                    weekday_minus_6months = weekday_last
                    weekday_minus_6months = weekday_minus_6months + relativedelta(
                        months=-6
                    )
                    while weekday_minus_6months.weekday() > 4:  # Mon-Fri are 0-4
                        weekday_minus_6months -= timedelta(days=1)

                    ###################################################
                    # Get Price momentum 6 months
                    weekday_minus_6months_year = f"{weekday_minus_6months.year:02}"
                    weekday_minus_6months_month = f"{weekday_minus_6months.month:02}"
                    weekday_minus_6months_day = f"{weekday_minus_6months.day:02}"

                    if (
                        f"{weekday_minus_6months_year}-{weekday_minus_6months_month}-{weekday_minus_6months_day}"
                        in quote[f"Time Series (Daily)"]
                    ):
                        quote_price_6months = quote[f"Time Series (Daily)"][
                            f"{weekday_minus_6months_year}-{weekday_minus_6months_month}-{weekday_minus_6months_day}"
                        ][f"4. close"]
                    else:
                        quote_price_6months = -1.0
                    weekday_minus_12months = weekday_last
                    weekday_minus_12months = weekday_minus_12months + relativedelta(
                        months=-12
                    )
                    while weekday_minus_12months.weekday() > 4:  # Mon-Fri are 0-4
                        weekday_minus_12months -= timedelta(days=1)

                    ###################################################
                    # Get Price momentum 12 months
                    weekday_minus_12months_year = f"{weekday_minus_12months.year:02}"
                    weekday_minus_12months_month = f"{weekday_minus_12months.month:02}"
                    weekday_minus_12months_day = f"{weekday_minus_12months.day:02}"

                    if (
                        f"{weekday_minus_12months_year}-{weekday_minus_12months_month}-{weekday_minus_12months_day}"
                        in quote[f"Time Series (Daily)"]
                    ):
                        quote_price_12months = quote[f"Time Series (Daily)"][
                            f"{weekday_minus_12months_year}-{weekday_minus_12months_month}-{weekday_minus_12months_day}"
                        ][f"4. close"]
                    else:
                        quote_price_12months = -1

                    # Calculated Price Momentum/Index
                    quote_price_index_6month = float(quote_price) / float(
                        quote_price_6months
                    )
                    if quote_price_index_6month < 0:
                        quote_price_index_6month = 0
                    quote_price_index_12month = float(quote_price) / float(
                        quote_price_12months
                    )
                    if quote_price_index_12month < 0:
                        quote_price_index_12month = 0
                    break
                elif "Information" in quote:
                    if (
                        quote["Information"]
                        == "Thank you for using Alpha Vantage! You have reached the 100 requests/day limit for your free API key. Please subscribe to any of the premium plans at https://www.alphavantage.co/premium/ to instantly remove all daily rate limits."
                    ):
                        print("--- Alpha Vantage Daily limit reached ---")
                        print(f"--- Trying again in {time_sleep} seconds")
                        time.sleep(time_sleep)
                        break
                elif "Note" in quote:
                    if (
                        quote["Note"]
                        == "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 100 calls per day. Please visit https://www.alphavantage.co/premium/ if you would like to target a higher API call frequency."
                    ):
                        print("--- Alpha Vantage Daily limit reached ---")
                        print(f"--- Trying again in {time_sleep} seconds")
                        time.sleep(time_sleep)
                        break
            except requests.exceptions.HTTPError as err:
                quote_price = 0.0
                quote_price_index_6month = 0.0
                quote_price_index_12month = 0.0
                print("error", err)
                print(f"--- Trying again in {time_sleep} seconds")
                time.sleep(time_sleep)
                break
            except:
                print("--- Failed to decode JSON")
                print("--- This is the returned result")
                print(response)
                print(f"--- Trying again in {time_sleep} seconds")
                time.sleep(time_sleep)

        ###################################################
        # Get Finviz data
        # quote = json.dumps(finviz.get_stock(f'{symbol}'), indent=4, sort_keys=True)

        while True:
            try:
                quote = finviz.get_stock(f"{symbol}")
                # Get Book to Market
                quote_bm = 0.0
                if "P/B" in quote:
                    quote_raw_pb = quote["P/B"]
                    if not quote_raw_pb == "-":
                        quote_bm = 1.0 / float(quote_raw_pb)

                # Get Free Cash Flow Yield 12 months
                quote_fcfy_12months = 0.0
                if "P/FCF" in quote:
                    quote_pfcf = quote["P/FCF"]
                    if not quote_pfcf == "-":
                        quote_fcfy_12months = 1.0 / float(quote_pfcf)

                # Get P/S
                quote_ps = 0.0
                if "P/S" in quote:
                    quote_raw_ps = quote["P/S"]
                    if not quote_raw_ps == "-":
                        quote_ps = float(quote_raw_ps)

                # Get Earning Yield 12 months
                quote_ey_12months = 0.0
                if "P/E" in quote:
                    quote_raw_pe = quote["P/E"]
                    if not quote_raw_pe == "-":
                        quote_ey_12months = 1.0 / float(quote_raw_pe)

                # Get Return On Asset
                quote_roa = 0.0
                if "ROA" in quote:
                    quote_raw_roa = quote["ROA"]
                    if not quote_raw_roa == "-":
                        quote_roa = float(quote_raw_roa[:-1])

                break

            except requests.exceptions.HTTPError as err:
                quote_bm = 0.0
                quote_ey_12months = 0.0
                quote_roa = 0.0
                quote_fcfy_12months = 0.0
                quote_ps = 0.0
                break

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
                "ROA": quote_roa
                # "Earnings Yield (12months) rank": ,
                # "ROA rank": ,
                # "Magic Formula Score": ,
            }
        )

        # print(str("\t".join([str(i) for i in list(quote_data[-1].values())])))

        print(f"--- PROCESSING {symbol} -- END ---")

        if not symbol == symbols[-1]:
            time.sleep(time_sleep)

    print(quote_data)
    print("--- ---")
    print("--- QUOTE EXTRACTION -- END ---")
    print("--- ---")
    # Rank based on Earnings Yield (12months)
    print("--- Rank based on Earnings Yield (12months) -- START ---")
    quote_data = sorted(
        quote_data, key=itemgetter("Earnings Yield (12months)"), reverse=True
    )
    for idx, quote_data_item in enumerate(quote_data):
        quote_data[idx]["Earnings Yield (12months) rank"] = idx + 1

    print("--- Rank based on Earnings Yield (12months) -- END ---")

    # Rank based on ROA
    print("--- Rank based on ROA -- START ---")
    quote_data = sorted(quote_data, key=itemgetter("ROA"), reverse=True)
    for idx, quote_data_item in enumerate(quote_data):
        quote_data[idx]["ROA rank"] = idx + 1
    print("--- Rank based on ROA -- END ---")

    # Calculate the Magic Formula score and Rank based on this score
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


# for idx, quote_data_item in enumerate(quote_data):
#     print(headers)
#     print(quote_data_item)


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
