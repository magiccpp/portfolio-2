from concurrent.futures import ThreadPoolExecutor
from functools import partial
import uuid
from util import create_if_not_exist, generate_features, generate_features_individual_stock, get_X_y_by_stock, get_tickers, get_currency_pair, load_latest_price_data, convert, add_features, load_pkl, merge_fred, remove_nan
from util import save_pkl, get_pipline_svr, get_pipline_rf
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
from sklearn.preprocessing import StandardScaler
import logging
import getopt
import sys
from sklearn.preprocessing import FunctionTransformer
from util import load_pkl
from optuna.storages import RDBStorage
from sqlalchemy.pool import NullPool

logger = logging.getLogger('training')
logger.setLevel(logging.DEBUG)  # Set the logging level

# Create a file handler
file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.DEBUG)  # Set the logging level for the file handler

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the logging level for the console handler

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

random.seed(42)

# the minimal number of data entries for a stock

TIMEOUT = 120
# number of trials to do the hyper-parameter optimization
N_TRIALS = 50

# get_tickers('ssec.txt')
tickers = get_tickers('dax_40.txt') + get_tickers('ftse_100.txt') + get_tickers('sp_500.txt') + get_tickers('omx_30.txt') + get_tickers('cac_40.txt') \
  + get_tickers('etf.txt') + get_tickers('nasdaq-100.txt')

# remove duplicated tickers
tickers = list(set(tickers))

selected_tickers = tickers


#selected_tickers = random.sample(tickers, 10)


def get_mse_from_hist_average(df_X, df_y, window_size, period):
  return ((df_X[f'log_price_diff_{period}'].rolling(window=window_size).mean()[window_size:] - df_y['log_predict'][window_size:])**2).mean()


def get_X_y(selected_tickers, period, start, end, split_date, force_download):
  df_train_X_all = []
  df_train_y_all = []
  df_test_X_all = []
  df_test_y_all = []
  mean_square_errors_1 = []
  mean_square_errors_3 = []
  mean_square_errors_5 = []
  valid_tickers = []
  for stock_name in selected_tickers:
    df_train_X, df_train_y, df_test_X, df_test_y = get_X_y_by_stock(stock_name, period, start, end, split_date, force_download)
    if df_train_X is None:
      continue 
    

    valid_tickers.append(stock_name)
    df_train_X_all.append(df_train_X)
    df_train_y_all.append(df_train_y)
    df_test_X_all.append(df_test_X)
    df_test_y_all.append(df_test_y)
    
    mse_1 = get_mse_from_hist_average(df_test_X, df_test_y, 1, period)
    mse_3 = get_mse_from_hist_average(df_test_X, df_test_y, 3, period)
    mse_5 = get_mse_from_hist_average(df_test_X, df_test_y, 5, period)

    mean_square_errors_1.append(mse_1)
    mean_square_errors_3.append(mse_3)
    mean_square_errors_5.append(mse_5)

  return valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all, {'mse_1': mean_square_errors_1, 'mse_3': mean_square_errors_3, 'mse_5': mean_square_errors_5}

def save_data(data_dir, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all):
  # create data_dir if it does not exist
  os.makedirs(data_dir, exist_ok=True)

  # save the valid_tickers in a text file under the directory
  with open(f'{data_dir}/valid_tickers.txt', 'w') as f:
      for ticker in valid_tickers:
          f.write("%s\n" % ticker)

  save_pkl(df_train_X_all, f'{data_dir}/df_train_X_all.pkl')
  save_pkl(df_train_y_all, f'{data_dir}/df_train_y_all.pkl')
  save_pkl(df_test_X_all, f'{data_dir}/df_test_X_all.pkl')
  save_pkl(df_test_y_all, f'{data_dir}/df_test_y_all.pkl')


def objective_random_forest(trial, df_train_X, df_train_y):
  # Define the hyperparameter configuration space
  k = trial.suggest_int('k', 5, len(df_train_X.columns))
  n_estimators = trial.suggest_int('n_estimators', 20, 160)
  max_depth = trial.suggest_int('max_depth', 10, 50)
  min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
  min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
  max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
  bootstrap = trial.suggest_categorical('bootstrap', [True, False])
  max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 6, 200)

  params = {"n_estimators": n_estimators, "max_depth": max_depth, \
            "min_samples_split": min_samples_split, \
              "min_samples_leaf": min_samples_leaf, "max_features": max_features, \
                "bootstrap": bootstrap, "max_leaf_nodes": max_leaf_nodes, "k": k}
  
  pipeline = get_pipline_rf(params)

  total_mses = 0
  try:
    X_train = df_train_X.copy().values
    y_train = df_train_y.copy().values.ravel()
    predictions = cross_val_predict(pipeline, X_train, y_train, cv=5, n_jobs=5)
    mse = mean_squared_error(y_train, predictions)
    
    return mse
  except TimeoutException:
      logger.error("A timeout has occurred during model fitting.")
      # Return a large MSE value to penalize this result
      return float('inf')



def objective_svm(trial, df_train_X, df_train_y):
  # Define the hyperparameter configuration space
  k = trial.suggest_int('k', 5, len(df_train_X.columns))
  C = trial.suggest_float('C', 1e-3, 1e2,log=True)
  epsilon = trial.suggest_float('epsilon', 1e-3, 1e1, log=True)
  kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
  gamma = trial.suggest_float('gamma', 1e-4, 1e1, log=True)

  # Model setup
  params = {"C": C, "kernel": kernel, "gamma": gamma, "epsilon": epsilon, "k": k}
  
  print('starting trying parameters: ', params)
  # model = SafeSVR(C=C,  kernel=kernel, gamma=gamma, epsilon=epsilon, timeout=TIMEOUT)

  # pipeline = Pipeline([
  #     ('scaler', StandardScaler()),  # Add scaler here
  #     ('truncate', TruncationTransformer(k=k)),
  #     ('svr', model),
  # ])
  pipeline = get_pipline_svr(params)

  try:
    X_train = df_train_X.copy().values
    y_train = df_train_y.copy().values.ravel()

    predictions = cross_val_predict(pipeline, X_train, y_train, cv=5, n_jobs=-1)

    mse = mean_squared_error(y_train, predictions)
    
    return mse
  except TimeoutException:
      logger.error("A timeout has occurred during model fitting.")
      # Return a large MSE value to penalize this result
      return float('inf')



def test_naive(valid_tickers, df_test_X_all, df_test_y_all, period):
  naive_mses_1 = []
  naive_mses_8 = []
  naive_mses_16 = []
  naive_mses_0 = []  # naive mse of using 0 as prediction
  naive_mses_negation = []
  naive_mses_avg_512 = []
  for i in range(len(valid_tickers)):
    df_test_X = df_test_X_all[i]
    df_test_y = df_test_y_all[i]
    naive_mse_1 = get_mse_from_hist_average(df_test_X, df_test_y, 1, period)
    #print(f'naive MSE_1 of {stock_name}: {naive_mse_1}')
    naive_mses_1.append(naive_mse_1)

    naive_mse_8 = get_mse_from_hist_average(df_test_X, df_test_y, 8, period)
    #print(f'naive MSE_8 of {stock_name}: {naive_mse_8}')
    naive_mses_8.append(naive_mse_8)

    naive_mse_16 = get_mse_from_hist_average(df_test_X, df_test_y, 16, period)
    #print(f'naive MSE_16 of {stock_name}: {naive_mse_16}')
    naive_mses_16.append(naive_mse_16)

    naive_mse_0 = (df_test_y['log_predict']**2).mean()
    naive_mses_0.append(naive_mse_0)

    naive_mse_negation = mean_squared_error(df_test_X[f'log_price_diff_{period}'], df_test_y['log_predict'])
    naive_mses_negation.append(naive_mse_negation)


    divisor = 512/period
    naive_mse_avg_512 = ((df_test_X[f'log_price_diff_512'].rolling(window=512).mean()[512:]/divisor - df_test_y['log_predict'][512:])**2).mean()
    naive_mses_avg_512.append(naive_mse_avg_512)



  logger.info(f'The MSE of averaging past 1 * {period} days: {np.mean(naive_mses_1)}, std: {np.std(naive_mses_1)}')
  logger.info(f'The MSE of averaging past 8 * {period} days: {np.mean(naive_mses_8)}, std: {np.std(naive_mses_8)}')
  logger.info(f'The MSE of averaging past 16 * {period} days: {np.mean(naive_mses_16)}, std: {np.std(naive_mses_16)}')
  logger.info(f'The MSE of using 0 as prediction: {np.mean(naive_mses_0)}, std: {np.std(naive_mses_0)}')
  logger.info(f'The MSE of using inverse of last period as prediction: {np.mean(naive_mses_negation)}, std: {np.std(naive_mses_negation)}')
  logger.info(f'The MSE of using average of 512 days: {np.mean(naive_mses_avg_512)}, std: {np.std(naive_mses_avg_512)}')

def test_all(data_dir, feature_dir, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all, period, storage):
  mses_rf = []
  mses_svm = []
  mses_naive = []
  mses = []
  all_errors = None
  for i in range(len(valid_tickers)):
    stock_name = valid_tickers[i]
    logger.info(f'Starting test on {stock_name}...')
    df_train_X = df_train_X_all[i]
    df_train_y = df_train_y_all[i]
    df_test_X = df_test_X_all[i]
    df_test_y = df_test_y_all[i]

    X_train = df_train_X.copy().values
    y_train = df_train_y.copy().values.ravel()
    X_test = df_test_X.copy().values
    y_test = df_test_y.copy().values.ravel()

    feature_file_name = f'{feature_dir}/sorted_features_{stock_name}.txt'
    with open(feature_file_name, 'r') as file:
      sorted_features = np.array(file.read().split('\n'))
      sorted_features = sorted_features[sorted_features != '']
      
    df_train_X = df_train_X[sorted_features]
    
    study_name_rf = f'study_rf_columns_{len(sorted_features)}_stocks_{stock_name}_period_{period}'
    
    study_rf = optuna.create_study(study_name=study_name_rf, storage=storage, load_if_exists=True)
    best_pipeline_rf = get_pipline_rf(study_rf.best_params)
    best_pipeline_rf.fit(X_train, y_train)
    y_pred_rf = best_pipeline_rf.predict(X_test)
    # save the model to a file
    with open(f'{data_dir}/models/{stock_name}_rf.pkl', 'wb') as f:
      pickle.dump(best_pipeline_rf, f)
      
    study_name_svm = f'study_svm_columns_{len(sorted_features)}_stocks_{stock_name}_period_{period}'
    
    study_svm = optuna.create_study(study_name=study_name_svm, storage=storage, load_if_exists=True)
    best_pipeline_svm = get_pipline_svr(study_svm.best_params)
    best_pipeline_svm.fit(X_train, y_train)
    y_pred_svm = best_pipeline_svm.predict(X_test)
    # save the model to a file
    with open(f'{data_dir}/models/{stock_name}_svm.pkl', 'wb') as f:
      pickle.dump(best_pipeline_svm, f)

    # compute the naive prediction
    divisor = 512 / period
    df_test_X_naive = pd.concat((df_train_X[-512:], df_test_X))
    y_pred_naive = df_test_X_naive[f'log_price_diff_512'].rolling(window=512).mean()[512:] / divisor

    # y_pred is the average of the two predictions
    y_pred = (y_pred_svm + y_pred_rf + y_pred_naive) / 3

    df_error = pd.DataFrame(y_pred - y_test, index=df_test_y.index, columns=[stock_name])
    if all_errors is None:
      all_errors = df_error
    else:
      # concatenate the new dataframe to the existing one, column wise, use outer approach
      all_errors = pd.concat([all_errors, df_error], axis=1, join='outer')

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mse_svm = mean_squared_error(y_test, y_pred_svm)
    mse_naive = mean_squared_error(y_test, y_pred_naive)
    mse = mean_squared_error(y_test, y_pred)

    logger.debug(f'{stock_name} MSE RF: {mse_rf}, MSE SVM: {mse_svm}, MSE Naive: {mse_naive}, MSE Ensemble: {mse}')

    mses_rf.append(mse_rf)
    mses_svm.append(mse_svm)
    mses_naive.append(mse_naive)
    mses.append(mse)

  logger.info(f'MSE of RF: avg: {np.mean(mses_rf)}, std: {np.std(mses_rf)}')
  logger.info(f'MSE of SVM: avg: {np.mean(mses_svm)}, std: {np.std(mses_svm)}')
  logger.info(f'MSE of Naive (512 days average): avg: {np.mean(mses_naive)}, std: {np.std(mses_naive)}')
  logger.info(f'Ensemble: avg: {np.mean(mses)}, std: {np.std(mses)}')
  

def test_rf(best_pipeline, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all):
  mse_rf = []
  all_errors = None

  for i in range(len(valid_tickers)):
    stock_name = valid_tickers[i]
    logger.info(f'Starting test on {stock_name}...')
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
    logger.debug(f'Correlation between std_predictions and errors: {correlation}')

    if all_errors is None:
      all_errors = df_error
    else:
      # concatenate the new dataframe to the existing one, column wise, use outer approach
      all_errors = pd.concat([all_errors, df_error], axis=1, join='outer')

    mse = mean_squared_error(y_test, y_pred)
    logger.debug(f'RF: MSE of {stock_name}: {mse}')
    mse_rf.append(mse)

  logger.info('The average MSE of RF: ', np.mean(mse_rf))
  logger.info('The STD of MSE of RF: ', np.std(mse_rf))


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
    logger.info(f'SVM: Test {valid_tickers[i]} MSE: {mse}')
    mses.append(mse)

  logger.info(f'The average MSE of SVM: {np.mean(mses)}')
  logger.info(f'The STD of MSE of SVM: {np.std(mses)}')


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


def optimize(algo_name, study_name, storage, iterations, objective, get_pipline, df_train_X, df_train_y):
  print(f'Starting finding optimized hyper-parameters for the {algo_name}...')
  study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)
  # check if study_svm contains best value.
  if len(study.get_trials()) > 0:
    best_value = study.best_trial.value
  else:
    best_value = None
  logger.info(f'Current best value: {best_value}')

  study.optimize(lambda trial: objective(trial, df_train_X, df_train_y), 
                    n_trials=iterations)
  best_value_new = study.best_value

  if best_value_new is not None and (best_value is None or best_value_new < best_value):
    best_pipeline = get_pipline(study.best_params)
    logger.info(f'new best value found: {best_value_new}')
  else:
    best_pipeline = get_pipline(study.best_params)
  return best_pipeline

def main(argv):
  logger.info('training started...')
  period = None
  iterations = 10
  try:
      # delete: delete the previous study
      opts, args = getopt.getopt(argv, "", ["period=", "svr_iter=", "rf_iter=", "postgres_ip=", "reload", "delete_svr", "delete_rf", "skip_test", "generate_feature_file"])
  except getopt.GetoptError:
    logger.error('usage: python train_model_v2025.py --period <days> --svr_iter <iterations> --rf_iter <iteration> --postgres_ip <postgres_ip>  --reload --delete_svr --delete_rf --skip_test --generate_feature_file')
    sys.exit(2)

  reload_data = False
  delete_svr_study = False
  delete_rf_study = False
  skip_test = False
  generate_feature_file = False
  for opt, arg in opts:
    if opt in ("-p", "--period"):
        period = int(arg)
    elif opt in ("-s", "--svr_iter"):
        svr_iterations = int(arg)
    elif opt in ("-f", "--rf_iter"):
        rf_iterations = int(arg)
    elif opt in ("-r", "--reload"):
        reload_data = True
    elif opt in ("-d", "--delete_svr"):
        delete_svr_study = True
    elif opt in ("-d", "--delete_rf"):
        delete_rf_study = True
    elif opt in ("--skip_test"):
        skip_test = True
    elif opt in ("--generate_feature_file"):
        generate_feature_file = True
    elif opt in ("--postgres_ip"):
        postgres_ip = arg


  if postgres_ip is None:
    postgres_ip = "127.0.0.1"

  postgres_url = f"postgresql+psycopg2://postgres:example@{postgres_ip}:5432/app_db"

# Build a single shared storage once (before your loops)
  storage = RDBStorage(
      url=postgres_url,
      engine_kwargs={
          # Either disable pooling entirely:
          "poolclass": NullPool,
      },
  )

  if period is None:
    logger.error('python train_model_v2025.py --period <days> --svr_iter <iterations> --rf_iter <iteration> --reload --delete_svr --delete_rf --skip_test --generate_feature_file')
    sys.exit(2)

  if svr_iterations is None:
    svr_iterations = N_TRIALS

  if rf_iterations is None:
    rf_iterations = N_TRIALS
  data_dir = f'./processed_data_2025_{period}'
  if reload_data or not if_data_exists(data_dir):
    logger.info(f'Preparing data from {len(selected_tickers)} assets...')
    start = '1970-01-01'
    end = '2025-01-01'
    valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all, mses = \
      get_X_y(selected_tickers, period, start, end, split_date='2022-01-01', force_download=False)
    save_data(data_dir, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all)
    logger.info(f'Data saved to {data_dir}...')
  else:
    df_train_X_all = load_pkl(f'{data_dir}/df_train_X_all.pkl')
    df_train_y_all = load_pkl(f'{data_dir}/df_train_y_all.pkl')
    df_test_X_all = load_pkl(f'{data_dir}/df_test_X_all.pkl')
    df_test_y_all = load_pkl(f'{data_dir}/df_test_y_all.pkl')
    with open(f'{data_dir}/valid_tickers.txt', 'r') as f:
      valid_tickers = f.read().splitlines()
    logger.info(f'Data loaded from {data_dir}...')

  logger.info(f'Data preparation finished, found {len(valid_tickers)} assets with enough data.')
  logger.info("Finding the importances of features...")


  logger.info('Starting finding optimized hyper-parameters for the random forest...')
  np.random.seed(42)
 

  n_columns = len(df_train_X_all[0].columns)
  feature_dir = f'{data_dir}/features'
  for ticker in valid_tickers:
    logger.info(f'Processing {ticker}...')
    df_train_X, df_train_y = df_train_X_all[0], df_train_y_all[0]

    # create the feature directory if it does not exist
    create_if_not_exist(feature_dir)
    logger.info(f'Generating features for {ticker}...')
    sorted_features = generate_features_individual_stock(df_train_X, df_train_y, ticker, feature_dir)
    df_train_X = df_train_X[sorted_features]
    
    logger.info(f'Optimizing Random forest hyper-parameters for {ticker}...')
    study_rf_name = f'study_rf_columns_{n_columns}_stocks_{ticker}_period_{period}'
    if delete_rf_study:
      try:
        optuna.delete_study(study_rf_name, storage=storage)
        logger.info(f'Study {study_rf_name} deleted...')
      except:
        # pass if the study does not exist
        pass

    if rf_iterations > 0:
      best_pipeline_rf = optimize('Random Forest', study_rf_name, storage, rf_iterations, objective_random_forest, get_pipline_rf,
                                          df_train_X, df_train_y)

    logger.info(f'Optimizing SVM hyper-parameters for {ticker}...')
    study_svm_name = f'study_svm_columns_{n_columns}_stocks_{ticker}_period_{period}'
    if delete_svr_study:
      try:
        optuna.delete_study(study_svm_name, storage=storage)
        logger.info(f'Study {study_svm_name} deleted...')
      except:
        # pass if the study does not exist
        pass

    if svr_iterations > 0:
      best_pipeline_svm = optimize('SVM', study_svm_name, storage, svr_iterations, objective_svm, get_pipline_svr,
                                            df_train_X, df_train_y)
    
  
  if skip_test:
    print('skiping test...')
    return
  
  logger.info(f'Starting test')
  test_naive(valid_tickers, df_test_X_all, df_test_y_all, period)
  #test_rf(best_pipeline_rf, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all)
  test_all(data_dir, feature_dir, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all, period, storage=storage)
  #test_svm(best_pipeline_svm, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all)

  try:
    storage._engine.dispose()
  except Exception:
    pass

if __name__ == "__main__":
    main(sys.argv[1:])