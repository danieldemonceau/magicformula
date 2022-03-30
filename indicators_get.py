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

def indicators_get(symbols, period_name, source_name):
    # Get configs
    load_dotenv()
    av_apikey = os.getenv('AV_APIKEY')

    # This is because of the limitation of 5 requests / minute, thus 12 seconds wait between API Calls
    time_sleep = 4
    print(f'--- [symbol]: quote_price\t\tquote_price_index_6month\tquote_price_index_12month\tquote_bTom\tquote_ey_12months\tquote_fcfy_12months\tquote_pToS ---')

    for symbol in symbols:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={av_apikey}'
        while True:
            try:
                response = requests.get(url)
                quote = response.json()
                if 'Error Message' in quote:
                    quote_price = None
                    quote_price_index_6month = None
                    quote_price_index_12month = None
                    break
                elif 'Meta Data' in quote:
                    ###################################################
                    # Get last price
                    weekday_last = date.today()
                    weekday_last -= timedelta(days = 1)
                    while weekday_last.weekday() > 4: # Mon-Fri are 0-4
                        weekday_last -= timedelta(days = 1)

                    weekday_last_year = f'{weekday_last.year:02}'
                    weekday_last_month = f'{weekday_last.month:02}'
                    weekday_last_day = f'{weekday_last.day:02}'
                    if f'{weekday_last_year}-{weekday_last_month}-{weekday_last_day}' in quote[f'Time Series (Daily)']:
                        quote_price = quote[f'Time Series (Daily)'][f'{weekday_last_year}-{weekday_last_month}-{weekday_last_day}'][f'4. close']
                    else:
                        quote_price = -1
                    weekday_minus_6months = weekday_last
                    weekday_minus_6months = weekday_minus_6months + relativedelta(months =- 6)
                    while weekday_minus_6months.weekday() > 4: # Mon-Fri are 0-4
                        weekday_minus_6months -= timedelta(days = 1)

                    ###################################################
                    # Get Price momentum 6 months
                    weekday_minus_6months_year = f'{weekday_minus_6months.year:02}'
                    weekday_minus_6months_month = f'{weekday_minus_6months.month:02}'
                    weekday_minus_6months_day = f'{weekday_minus_6months.day:02}'
                    
                    if f'{weekday_minus_6months_year}-{weekday_minus_6months_month}-{weekday_minus_6months_day}' in quote[f'Time Series (Daily)']:
                        quote_price_6months = quote[f'Time Series (Daily)'][f'{weekday_minus_6months_year}-{weekday_minus_6months_month}-{weekday_minus_6months_day}'][f'4. close']
                    else:
                        quote_price_6months = -1
                    weekday_minus_12months = weekday_last
                    weekday_minus_12months = weekday_minus_12months + relativedelta(months =- 12)
                    while weekday_minus_12months.weekday() > 4: # Mon-Fri are 0-4
                        weekday_minus_12months -= timedelta(days = 1)

                    ###################################################
                    # Get Price momentum 12 months
                    weekday_minus_12months_year = f'{weekday_minus_12months.year:02}'
                    weekday_minus_12months_month = f'{weekday_minus_12months.month:02}'
                    weekday_minus_12months_day = f'{weekday_minus_12months.day:02}'
                    
                    if f'{weekday_minus_12months_year}-{weekday_minus_12months_month}-{weekday_minus_12months_day}' in quote[f'Time Series (Daily)']:
                        quote_price_12months = quote[f'Time Series (Daily)'][f'{weekday_minus_12months_year}-{weekday_minus_12months_month}-{weekday_minus_12months_day}'][f'4. close']
                    else:
                        quote_price_12months = -1
                    
                    # Calculated Price Momentum/Index
                    quote_price_index_6month = float(quote_price) / float(quote_price_6months)
                    quote_price_index_12month = float(quote_price) / float(quote_price_12months)
                    break
            except requests.exceptions.HTTPError as err:
                quote_price = None
                quote_price_index_6month = None
                quote_price_index_12month = None
                break
            except:
                print('--- Failed to decode JSON')
                print('--- This is the returned result')
                print(response)
                print(f'--- Trying again in {time_sleep} seconds')
                time.sleep(time_sleep)

        ###################################################
        # Get Finviz data
        # quote = json.dumps(finviz.get_stock(f'{symbol}'), indent=4, sort_keys=True)
        
        while True:
            try:
                quote = finviz.get_stock(f'{symbol}')
                # Get Book to Market
                if 'P/B' in quote:
                    quote_pTob = quote['P/B']
                    if not quote_pTob == '-':
                        quote_bTom = 1. / float(quote_pTob)
                    else:
                        quote_bTom = None
                else:
                    quote_bTom = None

                # Get Earning Yield 12 months
                if 'P/E' in quote:
                    quote_pToe = quote['P/E']
                    if not quote_pToe == '-':
                        quote_ey_12months = 1. / float(quote_pToe)
                    else:
                        quote_ey_12months = None
                else:
                    quote_ey_12months = None
                
                # Get Free Cash Flow Yield 12 months
                if 'P/FCF' in quote:
                    quote_pTofcf = quote['P/FCF']
                    if not quote_pTofcf == '-':
                        quote_fcfy_12months = 1. / float(quote_pTofcf)
                    else:
                        quote_fcfy_12months = None
                else:
                    quote_fcfy_12months = None
                
                # Get P/S
                if 'P/S' in quote:
                    quote_pTos = quote['P/S']
                    if not quote_pTos == '-':
                        quote_pToS = float(quote_pTos)
                    break

            except requests.exceptions.HTTPError as err:
                quote_bTom = None
                quote_ey_12months = None
                quote_fcfy_12months = None
                quote_pToS = None
                break

        print(f'--- [{symbol}]: {quote_price}\t\t{quote_price_index_6month}\t{quote_price_index_12month}\t{quote_bTom}\t{quote_ey_12months}\t{quote_fcfy_12months}\t{quote_pToS} ---')

        if not symbol == symbols[-1]:
            time.sleep(time_sleep)


if __name__ == '__main__':
    time_start = time.time()
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print(' - - - FUNCTION START - - - ')
    print(' - - - REFRESH QUOTES - - - ')
    print(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

    with open("symbols.json") as json_data_file:
        data = json.load(json_data_file)

    symbols = data['symbols']
    
    period_name = '1day'
    source_name = 'alphavantage'
    if len(sys.argv) > 1:
        symbols = sys.argv[1]
    if len(sys.argv) > 2:
        period_name = sys.argv[2]
    if len(sys.argv) > 3:
        source_name = sys.argv[3]
    indicators_get(symbols, period_name, source_name)
    time_end = time.time()
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print(f' - - - Duration: {time_end - time_start} - - - ')
    print(' - - - FUNCTION FINISHED - - - ')
    print(' - - - REFRESH QUOTES - - - ')
    print(' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
