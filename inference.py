from util import do_optimization, generate_features, get_doubled_matrix, get_errors_mu_short, load_pkl, load_latest_price_data, add_features, merge_fred, portfolio_log_return, portfolio_return, portfolio_volatility, portfolio_volatility_log_return, remove_nan, save_json_to_dir, update_stock_operation_and_weight
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import logging
import getopt
import sys
import optuna
from util import save_pkl, get_pipline_svr, get_pipline_rf
import json
import torch
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

logger = logging.getLogger('inference')
logger.setLevel(logging.DEBUG)  # Set the logging level

# Create a file handler
file_handler = logging.FileHandler('inference.log')
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

INTEREST_RATE = 0.0497    # Current interest rate accessible for USD
ANNUAL_TRADING_DAYS = 252
MAX_RISK = 0.08



def get_predict_X(stock_name, sorted_features, start='1950-01-01', max_rows=1000):        
  df = load_latest_price_data(stock_name, start, end=None, save=True)


  df, feature_columns = add_features(df, 10)
  # timestamp = df.index[0]
  # earliest_date = timestamp.strftime('%Y-%m-%d')
  # start = earliest_date
  end = None

  df, columns = merge_fred(df, 'M2SL', 6, start, end, 4, 2, if_log=True)
  
  feature_columns += columns
  df, columns = merge_fred(df, 'UNRATE', 6, start, end, 1, 5, if_log=False)
  feature_columns += columns

  df, columns = merge_fred(df, 'FEDFUNDS', 6, start, end, 1, 5, if_log=False)
  feature_columns += columns
  df, _ = remove_nan(df, type='top')
  df_predict_X = df[feature_columns]
  
  return df_predict_X[sorted_features].iloc[-max_rows:]


def min_func_sharpe(weights, returns, covariance, risk_free_rate):
    portfolio_ret = portfolio_log_return(weights, returns)
    portfolio_vol = portfolio_volatility_log_return(weights, covariance)
    sharpe_ratio = (portfolio_ret - risk_free_rate) / portfolio_vol
    return -sharpe_ratio # Negate Sharpe ratio because we minimize the function



def get_shrinkage_covariance(df):
    lw = LedoitWolf(store_precision=False, assume_centered=True)
    lw.fit(df)
    # Convert the ndarray back to a DataFrame and use the column and index from the original DataFrame
    shrink_cov = pd.DataFrame(lw.covariance_, index=df.columns, columns=df.columns)
    return shrink_cov


def main(argv):
  logger.info('training started...')
  period = None
  iterations = 50
  update_covariance = True
  allow_short = True
  retrain = True
  try:
      opts, args = getopt.getopt(argv, "p:u", ["period=", "no-update-covariance", "no-retrain"])
  except getopt.GetoptError:
    logger.error('usage: python inference.py --period=<days> --no-update-covariance --no-retrain')
    sys.exit(2)
  for opt, arg in opts:
    if opt in ("-p", "--period"):
      period = int(arg)
    elif opt in ("-u", "--no-update-covariance"):
      update_covariance = False
    elif opt in ("--no-retrain"):
      retrain = False


  if period is None:
    logger.error('usage: script.py --period <days>')
    sys.exit(2)

  data_dir = f'processed_data_{period}'
  # old model: models/model_051524_facd9a57-6cfc-4f0a-90db-6db0e065649d_stocks_611
  df_train_X_all = load_pkl(f"{data_dir}/df_train_X_all.pkl")
  df_train_y_all = load_pkl(f"{data_dir}/df_train_y_all.pkl")
  df_test_X_all = load_pkl(f"{data_dir}/df_test_X_all.pkl")
  df_test_y_all = load_pkl(f"{data_dir}/df_test_y_all.pkl")

  # load the valid tickers.
  with open(f'{data_dir}/valid_tickers.txt', 'r') as f:
      valid_tickers = f.readlines()

  logger.info(f'Data preparation finished, found {len(valid_tickers)} assets with enough data.')
  feature_file_path = f'./processed_data_{period}/sorted_features.txt'

  if not os.path.exists(feature_file_path):
    print('generating features...')
    generate_features(period, df_train_X_all, df_train_y_all, valid_tickers)
  else:
    # Read the text file and convert the contents back to a numpy array
    # remove the empty string at the end
    with open(feature_file_path, 'r') as file:
      sorted_features = np.array(file.read().split('\n'))

  sorted_features = sorted_features[sorted_features != '']
  
  # iterate all tickers, reorder the features based on the scores by descending order
  for i in range(len(valid_tickers)):
    df_train_X_all[i] = df_train_X_all[i][sorted_features]  
    df_test_X_all[i] = df_test_X_all[i][sorted_features]

  # You can load it back into memory with the following code
  mysql_url = "mysql://root@192.168.2.34:3306/mysql"
  n_columns = len(df_train_X_all[0].columns)
  study_rf_name = f'study_rf_columns_{n_columns}_stocks_{len(valid_tickers)}_period_{period}'
  study_rf = optuna.create_study(study_name=study_rf_name, storage=mysql_url, load_if_exists=True)
  if study_rf.best_trial is None:
    logger.error('No best trial found')
    sys.exit(2)

  study_svm_name = f'study_svm_columns_{n_columns}_stocks_{len(valid_tickers)}_period_{period}'
  study_svm = optuna.create_study(study_name=study_svm_name, storage=mysql_url, load_if_exists=True)  
  if study_svm.best_trial is None:
    logger.error('No best trial found')
    sys.exit(2)


  best_pipeline_rf = get_pipline_rf(study_rf.best_params)
  best_pipeline_svr = get_pipline_svr(study_svm.best_params)

  mse_rf = []
  all_errors = None

  # the mean of standard deviation of predictions, mean_var_predictions[0] is a number indicating the mean of variance of predictions of the first stock
  # the multiplier which is cov(var_predictions, mse)/var(std_predictions)
  # during inferencing, the conditional expected error is calculated with:
  # E[errors|std_predictions=std] = mean_errors[i] + multiplier*(std-mean_std_predictions[i]) when 
  exp_profits = []
  final_tickers = []

  divisor = 512 / period
  n_days_errors = {}

  for i in range(len(valid_tickers)):
    stock_name = valid_tickers[i].strip()
    df_train_X = df_train_X_all[i]
    df_train_y = df_train_y_all[i]
    df_test_X = df_test_X_all[i]
    df_test_y = df_test_y_all[i]

    X_train = df_train_X.copy().values
    y_train = df_train_y.copy().values.ravel()
    X_test = df_test_X.copy().values
    y_test = df_test_y.copy().values.ravel()

    # make the prediction
    try:
      # check if the model exists
      # create the model path if not exists
      if not os.path.exists(f'{data_dir}/models'):
        os.makedirs(f'{data_dir}/models')

      rf_model_path = f'{data_dir}/models/{stock_name}_rf.pkl'
      svr_model_path = f'{data_dir}/models/{stock_name}_svr.pkl'

      if os.path.exists(rf_model_path) and not retrain:
        best_pipeline_rf = load_pkl(rf_model_path)
      else:
        print(f"retraining rf for stock: {stock_name}")
        best_pipeline_rf = get_pipline_rf(study_rf.best_params)
        best_pipeline_rf.fit(X_train, y_train)
        save_pkl(best_pipeline_rf, rf_model_path)

      if os.path.exists(svr_model_path) and not retrain:
        best_pipeline_svr = load_pkl(svr_model_path)
      else:
        print(f"retraining svr for stock: {stock_name}")
        best_pipeline_svr = get_pipline_svr(study_svm.best_params)
        best_pipeline_svr.fit(X_train, y_train)
        save_pkl(best_pipeline_svr, svr_model_path)

      y_pred_rf = best_pipeline_rf.predict(X_test)
      y_pred_svr = best_pipeline_svr.predict(X_test)
      # compute the naive prediction
      df_test_X_naive = pd.concat((df_train_X[-512:], df_test_X))
      y_pred_naive = (df_test_X_naive[f'log_price_diff_512'].rolling(window=512).mean()[512:] / divisor).to_numpy()

      # calculate the mean squared error
      err_rf = mean_squared_error(y_test, y_pred_rf)
      err_svr = mean_squared_error(y_test, y_pred_svr)
      err_naive = mean_squared_error(y_test, y_pred_naive)

      n_days = len(X_train)
      n_days_errors[stock_name] = (n_days, err_rf, err_svr, err_naive)
      logging.info(f"inference {period} stock: {stock_name}, n_days: {n_days}, err_rf: {err_rf}, err_svr: {err_svr}, err_naive: {err_naive}")

      logging.info(f"inference {period} length: y_pred_naive: {len(y_pred_naive)}, y_pred_rf: {len(y_pred_rf)}, y_pred_svr: {len(y_pred_svr)}")
      y_pred = (y_pred_rf + y_pred_svr + y_pred_naive) / 3

      df_predict_X = get_predict_X(stock_name, sorted_features)
      
      X_predict = df_predict_X.copy().values

      y_pred_2_rf = best_pipeline_rf.predict(X_predict)[512:]
      y_pred_2_svr = best_pipeline_svr.predict(X_predict)[512:]
      y_pred_2_naive = df_predict_X[f'log_price_diff_512'].rolling(window=512).mean()[512:] / divisor
      y_pred_2 = (y_pred_2_rf + y_pred_2_svr + y_pred_2_naive) / 3
      logging.info(f"inference {period} length: y_pred_2_naive: {len(y_pred_2_naive)}, y_pred_2_rf: {len(y_pred_2_rf)}, y_pred_2_svr: {len(y_pred_2_svr)}")
    except Exception as e:
      logger.error(f'Error in predicting {stock_name}: {e}')
      continue
    
    # save the dict n_days_errors
    with open(f'{data_dir}/n_days_errors.json', 'w') as f:
      json.dump(n_days_errors, f)
    

    df_error = pd.DataFrame(y_pred - y_test, index=df_test_y.index, columns=[stock_name])
    if all_errors is None:
      all_errors = df_error
    else:
      # concatenate the new dataframe to the existing one, column wise, use outer approach
      all_errors = pd.concat([all_errors, df_error], axis=1, join='outer')

    mse = mean_squared_error(y_test, y_pred)

    # the last prediction
    profit = y_pred_2.iloc[-1]
    exp_profits.append(profit)

    logger.info(f'inference {period} {stock_name}: exp profit={profit}, MSE={mse}')


    # mse_rf.append(mse)
    final_tickers.append(stock_name)

  if update_covariance:
    all_errors = all_errors.fillna(method='ffill').fillna(method='bfill')
    # Check for remaining NaNs
    nan_counts = all_errors.isna().sum()

    # Display columns with NaNs
    print(nan_counts[nan_counts > 0])

    # Display rows with NaNs
    print(all_errors[all_errors.isna().any(axis=1)])

    #df_error_0 = pd.DataFrame(0, index=all_errors.index, columns=['USD'])
    #all_errors = pd.concat([all_errors, df_error_0], axis=1, join='outer')
    #exp_profits.append(0)
    #final_tickers.append('USD')

    S = get_shrinkage_covariance(all_errors)
    S.to_pickle(f'{data_dir}/S.pkl')
  else:
    S = load_pkl(f'{data_dir}/S.pkl')
  #S = all_errors.cov()
  #S = CovarianceShrinkage(all_errors).ledoit_wolf()
  mu = exp_profits

  # to save S and mu

  # save the result
  cur_date = pd.Timestamp.now().strftime('%Y%m%d')
  np.save(f'{data_dir}/mu.npy', np.array(mu))

  # save the all errors

  all_errors.to_csv(f'{data_dir}/all_errors.csv')

  # save final tickers:
  with open(f'{data_dir}/final_tickers.txt', 'w') as f:
    for ticker in final_tickers:
      f.write(f'{ticker}\n')

  ticket_to_buy = do_optimization(mu, S, final_tickers, period, 0.02)

  save_json_to_dir(ticket_to_buy, f'{data_dir}/computed_portfolios')

  # allow short operations.
  if allow_short:
    all_errors_short, mu_positive = get_errors_mu_short(all_errors, mu)
    S_short = get_shrinkage_covariance(all_errors_short)

    ticket_to_buy_short = do_optimization(mu_positive, S_short, final_tickers, period, 0.02)
    # add the operation direction.

    weight_short = 0
    weight_long = 0

    for stock in ticket_to_buy_short:
        index = final_tickers.index(stock['id'])
        updated_weight =  update_stock_operation_and_weight(stock, index, mu)
        if updated_weight < 0:
            weight_short += updated_weight
        else:
            weight_long += updated_weight

    print(f"Short weight: {weight_short}, Long weight: {weight_long}")
    save_json_to_dir(ticket_to_buy_short, f'{data_dir}/computed_portfolios_short')

if __name__ == "__main__":
    main(sys.argv[1:])
