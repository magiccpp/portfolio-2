import uuid
from util import get_X_y_by_stock, get_tickers, get_currency_pair, load_latest_price_data, convert, add_features, load_pkl, merge_fred, remove_nan
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
from sklearn.base import BaseEstimator, TransformerMixin

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

tickers = get_tickers('dax_40.txt') + get_tickers('ftse_100.txt') + get_tickers('sp_500.txt') + get_tickers('omx_30.txt')
selected_tickers = tickers


#selected_tickers = random.sample(tickers, 10)


def get_mse_from_hist_average(df_X, df_y, window_size, period):
  return ((df_X[f'log_price_diff_{period}'].rolling(window=window_size).mean()[window_size:] - df_y['log_predict'][window_size:])**2).mean()


def get_X_y(selected_tickers, period, start, end, split_date):
  df_train_X_all = []
  df_train_y_all = []
  df_test_X_all = []
  df_test_y_all = []
  mean_square_errors_1 = []
  mean_square_errors_3 = []
  mean_square_errors_5 = []
  valid_tickers = []
  for stock_name in selected_tickers:
    df_train_X, df_train_y, df_test_X, df_test_y = get_X_y_by_stock(stock_name, period, start, end, split_date)
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


def objective_random_forest(trial, valid_tickers, df_train_X_all, df_train_y_all):
  # Define the hyperparameter configuration space
  k = trial.suggest_int('k', 5, len(df_train_X_all[0].columns))
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
    for i in range(len(valid_tickers)):
      if i % 100 == 0:
        logger.info(f'Processing {i}th stock...')
      df_train_X = df_train_X_all[i]
      df_train_y = df_train_y_all[i]

      X_train = df_train_X.copy().values
      y_train = df_train_y.copy().values.ravel()
      predictions = cross_val_predict(pipeline, X_train, y_train, cv=5, n_jobs=5)
      mse = mean_squared_error(y_train, predictions)
      total_mses += mse
    
    return total_mses/len(valid_tickers)
  except TimeoutException:
      logger.error("A timeout has occurred during model fitting.")
      # Return a large MSE value to penalize this result
      return float('inf')



def objective_svm(trial, valid_tickers, df_train_X_all, df_train_y_all):
  # Define the hyperparameter configuration space
  k = trial.suggest_int('k', 5, len(df_train_X_all[0].columns))
  C = trial.suggest_float('C', 1e-3, 1e2,log=True)
  epsilon = trial.suggest_float('epsilon', 1e-3, 1e1, log=True)
  kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
  gamma = trial.suggest_float('gamma', 1e-4, 1e1, log=True)

  # Model setup
  params = {"C": C, "kernel": kernel, "gamma": gamma, "epsilon": epsilon, "k": k}
   

  # model = SafeSVR(C=C,  kernel=kernel, gamma=gamma, epsilon=epsilon, timeout=TIMEOUT)

  # pipeline = Pipeline([
  #     ('scaler', StandardScaler()),  # Add scaler here
  #     ('truncate', TruncationTransformer(k=k)),
  #     ('svr', model),
  # ])
  pipeline = get_pipline_svr(params)

  total_mses = 0
  try:
    for i in range(len(valid_tickers)):
      if i % 100 == 0:
        logger.info(f'Processing {i}th stock...')
      df_train_X = df_train_X_all[i]
      df_train_y = df_train_y_all[i]

      X_train = df_train_X.copy().values
      y_train = df_train_y.copy().values.ravel()

      predictions = cross_val_predict(pipeline, X_train, y_train, cv=5, n_jobs=-1)

      mse = mean_squared_error(y_train, predictions)
      total_mses += mse
    
    return total_mses / len(valid_tickers)
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
    stock_name = valid_tickers[i]
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

def test_all(best_pipeline_svm, best_pipeline_rf, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all):
  mses_rf = []
  mses_svm = []
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

    best_pipeline_svm.fit(X_train, y_train)
    y_pred_svm = best_pipeline_svm.predict(X_test)

    best_pipeline_rf.fit(X_train, y_train)
    y_pred_rf = best_pipeline_rf.predict(X_test)

    # y_pred is the average of the two predictions
    y_pred = (y_pred_svm + y_pred_rf) / 2

    df_error = pd.DataFrame(y_pred - y_test, index=df_test_y.index, columns=[stock_name])
    if all_errors is None:
      all_errors = df_error
    else:
      # concatenate the new dataframe to the existing one, column wise, use outer approach
      all_errors = pd.concat([all_errors, df_error], axis=1, join='outer')

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mse_svm = mean_squared_error(y_test, y_pred_svm)
    mse = mean_squared_error(y_test, y_pred)

    logger.debug(f'{stock_name} MSE RF: {mse_rf}, MSE SVM: {mse_svm}, MSE Ensemble: {mse}')

    mses_rf.append(mse_rf)
    mses_svm.append(mse_svm)
    mses.append(mse)

  logger.info(f'MSE of RF: avg: {np.mean(mses_rf)}, std: {np.std(mses_rf)}')
  logger.info(f'MSE of SVM: avg: {np.mean(mses_svm)}, std: {np.std(mses_svm)}')
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

# get the sorted feature in descending order of importance.
def get_sorted_features(df_train_X_all, df_train_y_all):
  tot_scores = None
  for i in range(len(df_train_X_all)):
    # Filter the training and testing data for the current stock
    df_train_X_stock = df_train_X_all[i]
    df_train_y_stock = df_train_y_all[i]

    # Apply feature selection using SelectKBest
    selector = SelectKBest(f_regression, k=5)  # Select 5 best features
    selector.fit(df_train_X_stock, df_train_y_stock['log_predict'])
    scores = selector.scores_

    # make element-wise addition for tot_scores
    if tot_scores is None:
        tot_scores = scores/len(df_train_X_all)
    else:
        tot_scores += scores/len(df_train_X_all)
  # Get the scores in descending order
  feature_names = df_train_X_all[0].columns
  sorted_scores = scores.argsort()[::-1]  # Sort indices in descending order of scores
  sorted_features = feature_names[sorted_scores]
  return sorted_features

def main(argv):
  logger.info('training started...')
  period = None
  iterations = 10
  try:
      # delete: delete the previous study
      opts, args = getopt.getopt(argv, "p:i:rd", ["period=", "iter=", "reload", "delete"])
  except getopt.GetoptError:
    logger.error('usage: train_model.py --period <days> --iter <iterations> --reload --delete')
    sys.exit(2)

  reload_data = False
  delete_study = False
  for opt, arg in opts:
    if opt in ("-p", "--period"):
        period = int(arg)
    elif opt in ("-i", "--iter"):
        iterations = int(arg)
    elif opt in ("-r", "--reload"):
        reload_data = True
    elif opt in ("-d", "--delete"):
        delete_study = True

  if period is None:
    logger.error('usage: script.py --period <days> --iter <iterations>')
    sys.exit(2)

  if iterations is None:
    iterations = N_TRIALS

  data_dir = f'./processed_data_{period}'
  if reload_data or not if_data_exists(data_dir):
    logger.info(f'Preparing data from {len(selected_tickers)} assets...')
    start = '1970-01-01'
    end = '2024-01-01'
    valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all, mses = \
      get_X_y(selected_tickers, period, start, end, split_date='2019-01-01')
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
  sorted_features = get_sorted_features(df_train_X_all, df_train_y_all)
  logger.info(f'The ordered features are: {sorted_features}')

  # iterate all tickers, reorder the features based on the scores by descending order
  for i in range(len(valid_tickers)):
    df_train_X_all[i] = df_train_X_all[i][sorted_features]
    df_test_X_all[i] = df_test_X_all[i][sorted_features]

  logger.info('Starting finding optimized hyper-parameters for the random forest...')
  np.random.seed(42)
 
  mysql_url = "mysql://root@192.168.2.34:3306/mysql"
  n_columns = len(df_train_X_all[0].columns)

  print('Starting finding optimized hyper-parameters for the RF...')
  study_rf_name = f'study_rf_columns_{n_columns}_stocks_{len(valid_tickers)}_period_{period}'
  if delete_study:
    try:
      optuna.delete_study(study_rf_name, storage=mysql_url)
      logger.info(f'Study {study_rf_name} deleted...')
    except:
      # pass if the study does not exist
      pass

  study_rf = optuna.create_study(study_name=study_rf_name, storage=mysql_url, load_if_exists=True)
  # check if study_svm contains best value.
  if len(study_rf.get_trials()) > 0:
    best_value_rf = study_rf.best_trial.value
  else:
    best_value_rf = None

  study_rf.optimize(lambda trial: objective_random_forest(trial, valid_tickers, df_train_X_all, df_train_y_all), 
                    n_trials=iterations)
  best_value_rf_new = study_rf.best_value

  if best_value_rf_new is not None and (best_value_rf is None or best_value_rf_new < best_value_rf):
    best_pipeline_rf = get_pipline_rf(study_rf.best_params)
    logger.info(f'new best value found: {best_value_rf_new}')
  else:
    logger.info('No better model found...')
    best_pipeline_rf = get_pipline_rf(study_rf.best_params)

  print('Starting finding optimized hyper-parameters for SVM...')
  study_svm_name = f'study_svm_columns_{n_columns}_stocks_{len(valid_tickers)}_period_{period}'
  if delete_study:
    try:
      optuna.delete_study(study_svm_name, storage=mysql_url)
      logger.info(f'Study {study_svm_name} deleted...')
    except:
      # pass if the study does not exist
      pass

  study_svm = optuna.create_study(study_name=study_svm_name, storage=mysql_url, load_if_exists=True)

  # check if study_svm contains best value.
  if len(study_svm.get_trials()) > 0:
    best_value_svm = study_rf.best_trial.value
  else:
    best_value_svm = None

  study_svm.optimize(lambda trial: objective_svm(trial, valid_tickers, df_train_X_all, df_train_y_all), n_trials=iterations)
  
  best_value_svm_new = study_svm.best_value
  if best_value_svm_new is not None and (best_value_svm is None or best_value_svm_new < best_value_svm):
    logger.info(f'new best value found: {best_value_svm_new}')
    best_pipeline_svm = get_pipline_svr(study_svm.best_params)
  else:
    logger.info(f'No better model found')
    best_pipeline_svm = get_pipline_svr(study_svm.best_params)

  logger.info(f'Starting test')

  test_naive(valid_tickers, df_test_X_all, df_test_y_all, period)
  #test_rf(best_pipeline_rf, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all)
  test_all(best_pipeline_rf, best_pipeline_svm, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all)
  #test_svm(best_pipeline_svm, valid_tickers, df_train_X_all, df_train_y_all, df_test_X_all, df_test_y_all)
  
if __name__ == "__main__":
    main(sys.argv[1:])