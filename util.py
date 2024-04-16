import pandas as pd
from datetime import timedelta
from pandas_datareader import data as pdr
import yfinance as yfin
import os

yfin.pdr_override()

def get_tickers(ticker_list_file):
  with open(f'data/{ticker_list_file}', 'r') as f:
    tickers = f.readlines()
    # replace space to dash '-'
    tickers = [ticker.replace(' ', '-').strip() for ticker in tickers]
    return tickers
