import uuid
from util import get_tickers, get_currency_pair, load_latest_price_data, convert, add_features, load_pkl, merge_fred, remove_nan
from util import save_pkl
from pandas_datareader import data as pdr
import random
import pandas as pd
import os
import numpy as np
from datetime import timedelta
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from safeRegressors import SafeRandomForestRegressor, SafeSVR, TimeoutException
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
import optuna
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression

random.seed(42)

# how many days to predict
PREDICT_PERIOD=128

# the minimal number of data entries for a stock
MIN_TOTAL_DATA_PER_STOCK = 1000
MIN_TRAINING_DATA_PER_STOCK = 500
MIN_TEST_DATA_PER_STOCK = 300
TIMEOUT = 120
# number of trials to do the hyper-parameter optimization
n_trails = 30

tickers = get_tickers('dax_40.txt') + get_tickers('ftse_100.txt') + get_tickers('sp_500.txt') + get_tickers('omx_30.txt')
selected_tickers = tickers
#selected_tickers = random.sample(tickers, 10)

def get_X_y_by_stock(stock_name, period, start, end, split_date='2018-01-01'):
  print(f'processing {stock_name}...')
  try:
    df = load_latest_price_data(stock_name, start=start, end=end)
  except FileNotFoundError:
    print(f'Cannot find data for: {stock_name}')
    return None, None, None, None
  
  if len(df) < MIN_TOTAL_DATA_PER_STOCK:
    print(f'Cannot find enough data for: {stock_name}')
    return None, None, None, None

  stock_suffix = '.' + stock_name.split('.')[-1]
  exchange_name, needs_inversion = get_currency_pair(stock_suffix, 'USD')
  if exchange_name is not None:
    df = convert(df, exchange_name, needs_inversion)
    
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
    print(f'Cannot find enough data for: {stock_name}')
    return None, None, None, None
  df, _ = remove_nan(df, type='bottom')
  if len(df) < MIN_TOTAL_DATA_PER_STOCK:
    print(f'Cannot find enough data for: {stock_name}')
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


def get_mse_from_hist_average(df_X, df_y, window_size):
  return ((df_X['log_price_diff_128'].rolling(window=window_size).mean()[window_size:] - df_y['log_predict'][window_size:])**2).mean()


def get_X_y(selected_tickers, period, start, end):
  df_train_X_all = []
  df_train_y_all = []
  df_test_X_all = []
  df_test_y_all = []
  mean_square_errors_1 = []
  mean_square_errors_3 = []
  mean_square_errors_5 = []
  valid_tickers = []
  for stock_name in selected_tickers:
    df_train_X, df_train_y, df_test_X, df_test_y = get_X_y_by_stock(stock_name, period, start, end)
    if df_train_X is None:
      continue 
    

    valid_tickers.append(stock_name)
    df_train_X_all.append(df_train_X)
    df_train_y_all.append(df_train_y)
    df_test_X_all.append(df_test_X)
    df_test_y_all.append(df_test_y)
    
    mse_1 = get_mse_from_hist_average(df_test_X, df_test_y, 1)
    mse_3 = get_mse_from_hist_average(df_test_X, df_test_y, 3)
    mse_5 = get_mse_from_hist_average(df_test_X, df_test_y, 5)

    mean_square_errors_1.append(mse_1)
    mean_square_errors_3.append(mse_3)
    mean_square_errors_5.append(mse_5)

  return valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all, {'mse_1': mean_square_errors_1, 'mse_3': mean_square_errors_3, 'mse_5': mean_square_errors_5}

def save_data(data_dir, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all):
  # save the valid_tickers in a text file under the directory
  with open(f'{data_dir}/valid_tickers.txt', 'w') as f:
      for ticker in valid_tickers:
          f.write("%s\n" % ticker)

  save_pkl(df_train_X_all, f'{data_dir}/df_train_X_all.pkl')
  save_pkl(df_train_y_all, f'{data_dir}/df_train_y_all.pkl')
  save_pkl(df_test_X_all, f'{data_dir}/df_test_X_all.pkl')
  save_pkl(df_test_y_all, f'{data_dir}/df_test_y_all.pkl')







def objective_random_forest(trial):
  # Define the hyperparameter configuration space
  k = trial.suggest_int('k', 5, len(df_train_X_all[0].columns))
  n_estimators = trial.suggest_int('n_estimators', 20, 160)
  max_depth = trial.suggest_int('max_depth', 10, 50)
  min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
  min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
  max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
  bootstrap = trial.suggest_categorical('bootstrap', [True, False])
  max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 6, 200)


  # Model setup
  model = SafeRandomForestRegressor(
      n_estimators=n_estimators,
      max_depth=max_depth,
      min_samples_split=min_samples_split,
      min_samples_leaf=min_samples_leaf,
      max_features=max_features,
      bootstrap=bootstrap,
      max_leaf_nodes=max_leaf_nodes,
      timeout=TIMEOUT
  )

  pipeline = Pipeline([
      ('truncate', SelectKBest(f_regression, k=k)), # Adjust 'k' as needed
      ('regress', model),
  ])

  total_mses = 0
  try:
    for i in range(len(valid_tickers)):
      if i % 100 == 0:
        print(f'Processing {i}th stock...')
      df_train_X = df_train_X_all[i]
      df_train_y = df_train_y_all[i]

      X_train = df_train_X.copy().values
      y_train = df_train_y.copy().values.ravel()

      predictions = cross_val_predict(pipeline, X_train, y_train, cv=5, n_jobs=5)

      mse = mean_squared_error(y_train, predictions)
      total_mses += mse
    
    return total_mses/len(valid_tickers)
  except TimeoutException:
      print("A timeout has occurred during model fitting.")
      # Return a large MSE value to penalize this result
      return float('inf')

def objective_svm(trial):
  # Define the hyperparameter configuration space
  k = trial.suggest_int('k', 5, len(df_train_X_all[0].columns))
  C = trial.suggest_float('C', 1e-3, 1e2,log=True)
  epsilon = trial.suggest_float('epsilon', 1e-3, 1e1, log=True)
  kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
  gamma = trial.suggest_float('gamma', 1e-4, 1e1, log=True)

  # Model setup
  model = SafeSVR(C=C,  kernel=kernel, gamma=gamma, epsilon=epsilon, timeout=TIMEOUT)

  pipeline = Pipeline([
      ('truncate', SelectKBest(f_regression, k=k)), # Adjust 'k' as needed
      ('svr', model),
  ])

  total_mses = 0
  try:
    for i in range(len(valid_tickers)):
      if i % 100 == 0:
        print(f'Processing {i}th stock...')
      df_train_X = df_train_X_all[i]
      df_train_y = df_train_y_all[i]

      X_train = df_train_X.copy().values
      y_train = df_train_y.copy().values.ravel()

      predictions = cross_val_predict(pipeline, X_train, y_train, cv=5, n_jobs=-1)

      mse = mean_squared_error(y_train, predictions)
      total_mses += mse
    
    return total_mses / len(valid_tickers)
  except TimeoutException:
      print("A timeout has occurred during model fitting.")
      # Return a large MSE value to penalize this result
      return float('inf')

def save_rf(best_params, model_dir):
  # get current date
  os.makedirs(model_dir, exist_ok=True)

  best_pipeline = Pipeline([
          ('truncate', SelectKBest(f_regression, k=best_params['k'])), # Adjust 'k' as needed
          ('regress', SafeRandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features'],
            bootstrap=best_params['bootstrap'],
            max_leaf_nodes=best_params['max_leaf_nodes'],
            timeout=TIMEOUT
      ))])
  # Save to file in the current working directory
  with open(f"{model_dir}/best_pipeline_rf.pkl", "wb") as file:  
      pickle.dump(best_pipeline, file)
  return best_pipeline

def save_svm(best_params, model_dir):
  # get current date
  os.makedirs(model_dir, exist_ok=True)

  best_pipeline = Pipeline([
          ('truncate', SelectKBest(f_regression, k=best_params['k'])), # Adjust 'k' as needed
          ('regress', SafeSVR(
            C=best_params['C'], 
            epsilon=best_params['epsilon'], kernel=best_params['kernel'],
            gamma=best_params['gamma'], timeout=TIMEOUT
      ))])
  # Save to file in the current working directory
  with open(f"{model_dir}/best_pipeline_svm.pkl", "wb") as file:  
      pickle.dump(best_pipeline, file)
  return best_pipeline





def test_naive(valid_tickers, df_test_X_all, df_test_y_all):
  naive_mses_1 = []
  naive_mses_8 = []
  naive_mses_16 = []
  for i in range(len(valid_tickers)):
    stock_name = valid_tickers[i]
    df_test_X = df_test_X_all[i]
    df_test_y = df_test_y_all[i]
    naive_mse_1 = get_mse_from_hist_average(df_test_X, df_test_y, 1)
    #print(f'naive MSE_1 of {stock_name}: {naive_mse_1}')
    naive_mses_1.append(naive_mse_1)

    naive_mse_8 = get_mse_from_hist_average(df_test_X, df_test_y, 8)
    #print(f'naive MSE_8 of {stock_name}: {naive_mse_8}')
    naive_mses_8.append(naive_mse_8)

    naive_mse_16 = get_mse_from_hist_average(df_test_X, df_test_y, 16)
    #print(f'naive MSE_16 of {stock_name}: {naive_mse_16}')
    naive_mses_16.append(naive_mse_16)

  print(f'The MSE of averaging past 1 * {PREDICT_PERIOD} days: {np.mean(naive_mses_1)}, std: {np.std(naive_mses_1)}')
  print(f'The MSE of averaging past 8 * {PREDICT_PERIOD} days: {np.mean(naive_mses_8)}, std: {np.std(naive_mses_8)}')
  print(f'The MSE of averaging past 16 * {PREDICT_PERIOD} days: {np.mean(naive_mses_16)}, std: {np.std(naive_mses_16)}')


def test_rf(best_pipeline, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all):
  mse_rf = []
  all_errors = None

  for i in range(len(valid_tickers)):
    stock_name = valid_tickers[i]
    print(f'Starting test on {stock_name}...')
    df_train_X = df_train_X_all[i]
    df_train_y = df_train_y_all[i]
    df_test_X = df_test_X_all[i]
    df_test_y = df_test_y_all[i]

    X_train = df_train_X.copy().values
    y_train = df_train_y.copy().values.ravel()
    X_test = df_test_X.copy().values
    y_test = df_test_y.copy().values.ravel()

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    X_test_tranformed = X_test

    for name, step in best_pipeline.steps[:-1]:
      X_test_tranformed = step.transform(X_test_tranformed)
    # Get predictions from each individual tree
    model = best_pipeline.named_steps['regress']
    predictions = np.array([tree.predict(X_test_tranformed) for tree in model.rf.estimators_])

    df_std_preditions = pd.DataFrame(np.std(predictions, axis=0), index=df_test_y.index, columns=[stock_name])
    # reuse the index of df_test_y and the value of y_pred to create a new dataframe
    df_error = pd.DataFrame(y_pred - y_test, index=df_test_y.index, columns=[stock_name])

    # calculate the correlation between std_predictions and errors, this should be a single value
    correlation = df_std_preditions.corrwith(np.abs(df_error), axis=0).values[0]
    print(f'Correlation between std_predictions and errors: {correlation}')

    if all_errors is None:
      all_errors = df_error
    else:
      # concatenate the new dataframe to the existing one, column wise, use outer approach
      all_errors = pd.concat([all_errors, df_error], axis=1, join='outer')

    mse = mean_squared_error(y_test, y_pred)
    print(f'RF: MSE of {stock_name}: {mse}')
    mse_rf.append(mse)

  print('The average MSE of RF: ', np.mean(mse_rf))
  print('The STD of MSE of RF: ', np.std(mse_rf))


def test_svm(best_pipeline, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all):
  mses = []
  for i in range(len(valid_tickers)):
    stock_name = valid_tickers[i]
    df_train_X = df_train_X_all[i]
    df_train_y = df_train_y_all[i]
    df_test_X = df_test_X_all[i]
    df_test_y = df_test_y_all[i]

    X_train = df_train_X.copy().values
    y_train = df_train_y.copy().values.ravel()
    X_test = df_test_X.copy().values
    y_test = df_test_y.copy().values.ravel()

    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'SVM: Test {valid_tickers[i]} MSE: {mse}')
    mses.append(mse)

  print('The average MSE of SVM:', np.mean(mses))
  print('The STD of MSE of SVM:', np.std(mses))


def if_data_exists(data_dir):
# check if all the data files exist
  if not os.path.exists(data_dir):
    return False
  if not os.path.exists(f'{data_dir}/valid_tickers.txt'):
    return False
  if not os.path.exists(f'{data_dir}/df_train_X_all.pkl'):
    return False
  if not os.path.exists(f'{data_dir}/df_train_y_all.pkl'):
    return False
  if not os.path.exists(f'{data_dir}/df_test_X_all.pkl'):
    return False
  if not os.path.exists(f'{data_dir}/df_test_y_all.pkl'):
    return False
  return True


# parse the input parameters and get the reload_data
reload_data = False
for i in range(1, len(os.sys.argv)):
  if os.sys.argv[i] == '--reload_data':
    reload_data = True

data_dir = './processed_data'
if reload_data or not if_data_exists(data_dir):
  print(f'Preparing data from {len(selected_tickers)} assets...')
  start = '1970-01-01'
  end = '2024-01-01'
  valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all, mses = get_X_y(selected_tickers, PREDICT_PERIOD, start, end)
  save_data(data_dir, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all)
  print(f'Data saved to {data_dir}...')
else:
  df_train_X_all = load_pkl(f'{data_dir}/df_train_X_all.pkl')
  df_train_y_all = load_pkl(f'{data_dir}/df_train_y_all.pkl')
  df_test_X_all = load_pkl(f'{data_dir}/df_test_X_all.pkl')
  df_test_y_all = load_pkl(f'{data_dir}/df_test_y_all.pkl')
  with open(f'{data_dir}/valid_tickers.txt', 'r') as f:
    valid_tickers = f.read().splitlines()
  print(f'Data loaded from {data_dir}...')



print(f'Data preparation finished, found {len(valid_tickers)} assets with enough data.')
print('Starting finding optimized hyper-parameters for the random forest...')


np.random.seed(42)


mysql_url = "mysql://root@192.168.2.34:3306/mysql"
n_columns = len(df_train_X_all[0].columns)

study_rf_name = f'study_rf_columns_{n_columns}_stocks_{len(valid_tickers)}'
study_rf = optuna.create_study(study_name=study_rf_name, storage=mysql_url, load_if_exists=True)
study_rf.optimize(objective_random_forest, n_trials=n_trails) # Adjust the number of trials

study_svm_name = f'study_svm_columns_{n_columns}'
study_svm = optuna.create_study(study_name=study_svm_name, storage=mysql_url, load_if_exists=True)
study_svm.optimize(objective_svm, n_trials=n_trails) # Adjust the number of trials

model_id = uuid.uuid4()
cur_date = datetime.now().strftime('%m%d%y')
model_dir = f'models/model_{cur_date}_{model_id}_stocks_{len(valid_tickers)}'
best_pipeline_rf = save_rf(study_rf.best_params, model_dir)
best_pipeline_svm = save_svm(study_svm.best_params, model_dir)

print(f'Model saved to {model_dir}...')
print(f'Starting test')

test_naive(valid_tickers, df_test_X_all, df_test_y_all)
test_rf(best_pipeline_rf, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all)
test_svm(best_pipeline_svm, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all)