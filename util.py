import pandas as pd
from datetime import timedelta
from pandas_datareader import data as pdr
from safeRegressors import SafeRandomForestRegressor, SafeSVR
import yfinance as yfin
import os
import time
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
yfin.pdr_override()

NUMBER_RECENT_SECONDS = 72000
MIN_TOTAL_DATA_PER_STOCK = 1000
MIN_TRAINING_DATA_PER_STOCK = 500
MIN_TEST_DATA_PER_STOCK = 300
TIMEOUT = 120


def get_tickers(ticker_list_file):
  with open(f'data/{ticker_list_file}', 'r') as f:
    tickers = f.readlines()
    # replace space to dash '-'
    tickers = [ticker.replace(' ', '-').strip() for ticker in tickers]
    return tickers

def nth_weekday_of_month(year, month, index, weekday):
  """
  Find the nth occurrence of a specific weekday in a given month.

  :param year: The year as an integer.
  :param month: The month as an integer (1-12).
  :param index: The index of the occurrence (1st, 2nd, 3rd, etc.)
  :param weekday: The day of the week as an integer where Monday is 1 and Sunday is 7.
  :return: The date of the nth weekday.
  """

  weekday = weekday - 1
  # Start at the beginning of the month
  first_day_of_month = pd.Timestamp(year=year, month=month, day=1)
  # Find the first occurrence of the specific weekday
  first_weekday = first_day_of_month + timedelta(days=((weekday - first_day_of_month.weekday()) + 7) % 7)
  
  # Add (index - 1) weeks to the first occurrence of the weekday
  nth_weekday = first_weekday + timedelta(weeks=index-1)
  return nth_weekday.day

def is_file_downloaded_recently(file_path, seconds=NUMBER_RECENT_SECONDS):
  if not os.path.exists(file_path):
    return False
  file_age = time.time() - os.path.getmtime(file_path)
  return file_age <= seconds

def get_table_by_id_fred(id, path, n_features, 
                         start='1950-01-01', end="2024-01-01", if_log=True):
  feature_columns = []
  if path is None:
    path = 'data/fred'

  file_path = os.path.join(path, f'{id}.csv')
  if not is_file_downloaded_recently(file_path):
    print(f'Metric: {id} need to be refreshed...')
    df = pdr.get_data_fred(id, start='1950-01-01', end=None)
    df.to_csv(f'{path}/{id}.csv')

  df = pd.read_csv(os.path.join(path, f'{id}.csv'), index_col='DATE', parse_dates=True)
  df = df[start:end]

  if if_log:
    df[f'log_{id}'] = np.log(df[id])

  n_days = [int(2**n) for n in range(n_features)]
  for n in n_days:
    if if_log:
      name = f'log_{id}_diff_{n}'
      df[name] = df[f'log_{id}'] - df[f'log_{id}'].shift(n)
    else:
      name = f'{id}_diff_{n}'
      df[name] = df[id] - df[id].shift(n)
    feature_columns.append(name)
  return df, feature_columns

def merge_fred(df, id, n_features, start, end, release_week_index, release_week_day, if_log=True):
  path = 'data/fred'
  df_new, columns = get_table_by_id_fred(id, path, n_features, start=start, end=end, if_log=if_log)
  

  def get_last_metric_date(row, release_week_index, release_week_day):
    year = row.name.year
    month = row.name.month
    day = row.name.day

    release_date = nth_weekday_of_month(year, month, release_week_index, release_week_day)
    if day <= release_date:
      if month == 1:
        year -= 1
        month = 11
      elif month == 2:
        year -= 1
        month = 12
      else:
        month -= 2
    else:
      if month == 1:
        year -= 1
        month = 12
      else:
        month -= 1
    
    return pd.to_datetime(f"{year}-{month}-01")
  
  df['LAST_METRIC_DATE'] = df.apply(get_last_metric_date, axis=1, 
                                    args=(release_week_index, release_week_day))
  
  df = pd.merge_asof(df, df_new[columns], left_on='LAST_METRIC_DATE', right_index=True)
  # delete the column 'LAST_METRIC_DATE'
  df = df.drop(columns=['LAST_METRIC_DATE'])
  return df, columns

def remove_nan(df, type='top'):
  if type == 'top':
    for i in range(len(df)):
      if df.iloc[i].isnull().any() == False:
        break
    df_top = df[:i]
    df = df[i:]

    return df, df_top
  
  elif type == 'bottom':
    for i in range(1, len(df)):
      if df.iloc[-i].isnull().any() == False:
        break
    df_tail = df[-i:]
    df = df[:-i]
    return df, df_tail
  

def add_features(df, n_features):
  feature_columns = []
  for i in range(n_features):
    n_days = 2**i

    df[f'log_price_diff_{n_days}'] = np.log(df['Adj Close']/df['Adj Close'].shift(n_days))
    #df[f'price_diff_{n_days}'] = pd.to_numeric(df[f'price_diff_{n_days}'], errors='coerce')
    log_volume = np.log(df['Volume']+1e-8)
    df[f'log_volume_diff_{n_days}'] = log_volume - log_volume.shift(n_days)
    feature_columns.append(f'log_price_diff_{n_days}')
    feature_columns.append(f'log_volume_diff_{n_days}')
    #feature_columns.append(f'volume_diff_{n_days}')
  return df, feature_columns


# Map the stock suffixes to their base currencies
currency_mapping = {
  '.ST': 'SEK',
  '.DE': 'EUR',
  '.L': 'GBP',
  '.SS': 'RMB'
}

# Map currency pairs to directions
conversion_mapping = {
  ('SEK', 'USD'): ('DEXSDUS', True, 'SEK=X'),
  ('EUR', 'USD'): ('DEXUSEU', False, 'EURUSD=X'),
  ('GBP', 'USD'): ('DEXUSUK', False, 'GBPUSD=X'),
  ('RMB', 'USD'): ('DEXCHUS', True, 'CNY=X')
}

def get_currency_pair(stock_suffix, base_currency):
    stock_base_currency = currency_mapping.get(stock_suffix, 'USD')
    if base_currency == stock_base_currency:
        return None, None, None  # No conversion needed
    else:
        return conversion_mapping.get((stock_base_currency, base_currency))


def read_and_filter_exchange_rates(exchange_name, exchange_name_yahoo, path='data/fred'):
  return read_and_filter(exchange_name, exchange_name_yahoo, path)

def read_and_filter(exchange_name, exchange_name_yahoo, path):
  filepath = f'{path}/{exchange_name}.csv'
  if not is_file_downloaded_recently(filepath):
    print(f'Metric: {exchange_name} need to be refreshed...')
    df = pdr.get_data_fred(exchange_name, start='1950-01-01', end=None)

    # fred data is out of date, we use yahoo data as complement
    df2 = pdr.get_data_yahoo(exchange_name_yahoo, start='1950-01-01', end=None)

    index_name = df.index.name
    df = df.join(df2[['Adj Close']], how='outer')
    df.index.name = index_name

    df[exchange_name].fillna(df['Adj Close'], inplace=True)
    df.drop('Adj Close', axis=1, inplace=True)
    df.dropna(inplace=True)
    df.to_csv(f'{path}/{exchange_name}.csv')

  df = pd.read_csv(filepath, index_col='DATE', parse_dates=True)
  return df

def convert(df, exchange_name, inversion, exchange_name_yahoo):
  df_rate = read_and_filter_exchange_rates(exchange_name, exchange_name_yahoo)
  start = max(df.index[0], df_rate.index[0])
  df = df[df.index >= start]
  df_rate = df_rate[df_rate.index >= start]

  df_rate = df_rate[[exchange_name]]
  if inversion:
    df_rate[exchange_name] = 1/df_rate[exchange_name]
  df_merged = pd.merge_asof(df, df_rate, left_index=True, right_index=True, direction='nearest')
  df_merged['Adj Close'] = df_merged['Adj Close'] * df_merged[exchange_name]
  return df_merged[['Adj Close', 'Volume']]

def load_latest_price_data(stock_name, start='1950-01-01', end=None):
  file_path = f'data/prices/{stock_name}.csv'
  if end is not None:
    # get the number of seconds from end to now
    now = pd.Timestamp.now()
    seconds = (now - pd.to_datetime(end)).total_seconds()
  else:
    seconds = NUMBER_RECENT_SECONDS

  if not is_file_downloaded_recently(file_path, seconds=seconds):
    print('Preparing downloading:', stock_name)
    data = pdr.get_data_yahoo(stock_name, start=start, end=None, timeout=40)
    
    if len(data) > 100:
      data.to_csv(file_path)
    else:
      print(f'Cannot download {stock_name}, using old data...')

  df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
  if len(df) < 100:
    return None
  
  stock_suffix = '.' + stock_name.split('.')[-1]
  exchange_name, needs_inversion, exchange_name_yahoo = get_currency_pair(stock_suffix, 'USD')
  if exchange_name is not None:
    df = convert(df, exchange_name, needs_inversion, exchange_name_yahoo)
  return df


def   get_X_y_by_stock(stock_name, period, start, end, split_date):
  try:
    df = load_latest_price_data(stock_name, start=start, end=end)
  except FileNotFoundError:
    print(f'Cannot find data for: {stock_name}')
    return None, None, None, None
  
  print(f'processing {stock_name}...')
  if df is None or len(df) < MIN_TOTAL_DATA_PER_STOCK:
    print(f'Cannot find enough data for: {stock_name}')
    return None, None, None, None
    
  if len(df) == 0:
    print(f'empty table...')
    return None, None, None, None

  df, feature_columns = add_features(df, 10)
  
  # the predict is the log return of period days.
  df['log_predict'] = np.log(df['Adj Close'].shift(-period) / df['Adj Close'])
  

  timestamp = df.index[0]
  earliest_date = timestamp.strftime('%Y-%m-%d')
  start = earliest_date
  end = None



  df, columns = merge_fred(df, 'M2SL', 6, start, end, 4, 2, if_log=True)
  feature_columns += columns

  
  df, columns = merge_fred(df, 'UNRATE', 6, start, end, 1, 5, if_log=False)
  feature_columns += columns

  df, columns = merge_fred(df, 'FEDFUNDS', 6, start, end, 1, 5, if_log=False)
  feature_columns += columns

  df, _ = remove_nan(df, type='top')
  if len(df) < MIN_TOTAL_DATA_PER_STOCK:
    print(f'Cannot find enough data for: {stock_name} after removing nan from the top')
    return None, None, None, None
  df, _ = remove_nan(df, type='bottom')
  if len(df) < MIN_TOTAL_DATA_PER_STOCK:
    print(f'Cannot find enough data for: {stock_name} after removing nan from the bottom')
    return None, None, None, None
  
  df = df[feature_columns + ['log_predict']]
  df.dropna(inplace=True)
  
  if len(df) < MIN_TOTAL_DATA_PER_STOCK:
    print(f'Cannot find enough data for: {stock_name}')
    return None, None, None, None
  
  df_test = df[df.index >= split_date]
  df_train = df[df.index < split_date]

  if len(df_train) < MIN_TRAINING_DATA_PER_STOCK:
    print(f'Cannot find enough training data for: {stock_name}')
    return None, None, None, None
  if len(df_test) < MIN_TEST_DATA_PER_STOCK:
    print(f'Cannot find enough test data for: {stock_name}')
    return None, None, None, None
  df_train_X = df_train[feature_columns]
  df_train_y = df_train[['log_predict']]
  df_test_X = df_test[feature_columns]
  df_test_y = df_test[['log_predict']]

  return df_train_X, df_train_y, df_test_X, df_test_y

def save_pkl(object, file):
  with open(file, 'wb') as f:
    pickle.dump(object, f)


def load_pkl(file):
  with open(file, 'rb') as f:
    return pickle.load(f)
  

class TruncationTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, k):
    self.k = k

  def fit(self, X, y=None):
    # Since this transformer is stateless, the fit method only needs to pass
    return self

  def transform(self, X, y=None):
    return X[:, :self.k]

def get_pipline_rf(params):
  pipeline = Pipeline([
          ('truncate', TruncationTransformer(k=params['k'])), # Adjust 'k' as needed
          ('regress', SafeRandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            bootstrap=params['bootstrap'],
            max_leaf_nodes=params['max_leaf_nodes'],
            timeout=TIMEOUT
      ))])
  
  return pipeline


def get_pipline_svr(params):
  pipeline = Pipeline([
          ('scaler', StandardScaler()),  # Add scaler here
          ('truncate', TruncationTransformer(k=params['k'])), # Adjust 'k' as needed
          ('regress', SafeSVR(
            C=params['C'], 
            epsilon=params['epsilon'], kernel=params['kernel'],
            gamma=params['gamma'], timeout=TIMEOUT
      ))])
  
  return pipeline
