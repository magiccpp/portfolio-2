from util import load_pkl, load_latest_price_data, add_features, merge_fred, remove_nan
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def get_predict_X(stock_name, start='2018-01-01'):        
  df = load_latest_price_data(stock_name)
  df, feature_columns = add_features(df, 10)
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
  df_predict_X = df[feature_columns]
  
  return df_predict_X

def portfolio_volatility_log_return(weights, covariance):
    return np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))

def portfolio_log_return(weights, returns):
    return np.sum(returns*weights)

def portfolio_volatility(weights, covariance_log_returns):
    covariance_returns = np.exp(covariance_log_returns) - 1
    return np.sqrt(np.dot(weights.T, np.dot(covariance_returns, weights)))

def portfolio_return(weights, log_returns):
    returns = np.exp(log_returns) - 1
    return np.sum(returns*weights)

def min_func_sharpe(weights, returns, covariance):
    return -portfolio_log_return(weights, returns) / portfolio_volatility_log_return(weights, covariance)

def optimize_portfolio(returns, covariance):
    num_assets = len(returns)
    args = (returns, covariance)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,0.15)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(min_func_sharpe, num_assets*[1./num_assets,], args=args,
                                method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def get_shrinkage_covariance(df):
    lw = LedoitWolf(store_precision=False, assume_centered=True)
    lw.fit(df)
    # Convert the ndarray back to a DataFrame and use the column and index from the original DataFrame
    shrink_cov = pd.DataFrame(lw.covariance_, index=df.columns, columns=df.columns)
    return shrink_cov

def adjust_weights(weights, threshold=0.05):
    new_weights = np.array(weights)
    # Identify weights below the threshold
    below_threshold = weights < threshold
    # Set these weights to 0
    new_weights[below_threshold] = 0
    # Compute the deficit (i.e., how much we are currently missing to get to a total of 1)
    deficit = 1 - np.sum(new_weights)
    # Spread this deficit equally among the remaining stocks (i.e., the ones with weights > 0.05)
    new_weights[~below_threshold] += deficit / np.sum(~below_threshold)
    return new_weights

data_dir = 'processed_data'
model_dir = 'models/model_051524_facd9a57-6cfc-4f0a-90db-6db0e065649d_stocks_611'
# You can load it back into memory with the following code
best_pipeline = load_pkl(f"{model_dir}/best_pipeline.pkl")
df_train_X_all = load_pkl(f"{data_dir}/df_train_X_all.pkl")
df_train_y_all = load_pkl(f"{data_dir}/df_train_y_all.pkl")
df_test_X_all = load_pkl(f"{data_dir}/df_test_X_all.pkl")
df_test_y_all = load_pkl(f"{data_dir}/df_test_y_all.pkl")


# load the valid tickers.
with open(f'{data_dir}/valid_tickers.txt', 'r') as f:
    valid_tickers = f.readlines()




mse_rf = []
all_errors = None


# the mean of standard deviation of predictions, mean_var_predictions[0] is a number indicating the mean of variance of predictions of the first stock
mean_var_predictions = []
# the multiplier which is cov(var_predictions, mse)/var(std_predictions)
# during inferencing, the conditional expected error is calculated with:
# E[errors|std_predictions=std] = mean_errors[i] + multiplier*(std-mean_std_predictions[i]) when 
multiplier = []
exp_profits = []
predict_stds = []
final_tickers = []
for i in range(len(valid_tickers)):
#for i in range(10):

  stock_name = valid_tickers[i]
  print(f'Starting Computing {stock_name}')
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
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    df_predict_X = get_predict_X(stock_name)
    
    X_predict = df_predict_X.copy().values

    y_pred_2 = best_pipeline.predict(X_predict)
  except Exception as e:
    print(f'Error in predicting {stock_name}: {e}')
    continue
  
  df_error = pd.DataFrame(y_pred - y_test, index=df_test_y.index, columns=[stock_name])
  if all_errors is None:
    all_errors = df_error
  else:
    # concatenate the new dataframe to the existing one, column wise, use outer approach
    all_errors = pd.concat([all_errors, df_error], axis=1, join='outer')

  mse = mean_squared_error(y_test, y_pred)
  print(f'{valid_tickers[i]} MSE: {mse}')

  # the last prediction
  profit = y_pred_2[-1]
  exp_profits.append(profit)
  print(f'Computing the latest expected profit on {stock_name}, profit={profit}')
  mse_rf.append(mse)
  final_tickers.append(stock_name)

S = get_shrinkage_covariance(all_errors.fillna(method='ffill').fillna(method='bfill'))
#S = all_errors.cov()
#S = CovarianceShrinkage(all_errors).ledoit_wolf()
mu = exp_profits
raw_weights = optimize_portfolio(mu, S)

adjusted_weights = adjust_weights(raw_weights.x)
tickers_to_buy = []

for index, ticker_name in enumerate(final_tickers):
   adjusted_weight = adjusted_weights[index]
   if adjusted_weight > 0:
      print(f'index: {index} {ticker_name}: weight {adjusted_weight} exp profit: {exp_profits[index]}, variance: {S[ticker_name][ticker_name]}')
      tickers_to_buy.append(ticker_name)

print(f'expected return in 128 trading days:', portfolio_return(adjusted_weights, mu))
print(f'volatility of the return in 128 trading days:', portfolio_volatility(adjusted_weights, S))
