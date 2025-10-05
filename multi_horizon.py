
import getopt
import json
import logging
import sys
import numpy as np
import pandas as pd
import os

from util import do_optimization, get_errors_mu_short, get_shrinkage_covariance, save_json_to_dir, update_stock_operation_and_weight

BASE_LINE_HORIZON = 256
RISK_FREE_RATE = 0.05


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # reconfigure even if handlers already exist
)

logger = logging.getLogger('inference')
logger.setLevel(logging.DEBUG)  # Set the logging level

def get_all_errors_and_mu(periods, data_dir_prefix):
  all_errors = None
  all_mu = None
  for time_horizon in periods:
    data_dir = f'./{data_dir_prefix}_{time_horizon}'
    df = pd.read_csv(os.path.join(data_dir, 'all_errors.csv'),  index_col=0, parse_dates=True)
    # compute the mean of the errors
    avg_err = df.mean(axis=1).mean()
    df = df * np.sqrt(BASE_LINE_HORIZON / time_horizon)
    logger.info(f'Average error for time horizon {time_horizon}: {avg_err}')

    df = df.add_suffix(f'_{time_horizon}')
    mu = np.load(os.path.join(data_dir, 'mu.npy'))
    mu = mu * (BASE_LINE_HORIZON / time_horizon) - RISK_FREE_RATE

    if all_errors is None:
      all_errors = df
    else:
      all_errors = pd.concat([all_errors, df], axis=1, join='outer')

    if all_mu is None:
      all_mu = mu
    else:
      all_mu = np.concatenate([all_mu, mu])

  return all_errors, all_mu


def get_remove_cols(all_mu, all_S):
  assert(len(all_mu) == all_S.shape[0])
  idx = np.where(all_mu < 0)[0]
  remove_columns = []
  for asset in all_S.columns[idx]:
    remove_columns.append(asset)

  # find the indices of the columns to remove
  #indices = [all_S.columns.get_loc(col) for col in remove_columns]
  return remove_columns, idx

def remove_col_and_optimize(mu, S, max_weight, remove_negative=True):
  mu_copy = mu.copy()
  S_copy = S.copy()
  if remove_negative:
    print('Before removing negative gain stocks:')
    print('mu:', len(mu_copy))
    print('S:', S_copy.shape)
    remove_cols, idx = get_remove_cols(mu_copy, S_copy)
    mu_final = np.delete(mu_copy, idx)
    S_final = S_copy.drop(S.index[idx])


    S_final = S_final.drop(S_final.columns[idx], axis=1)
    print('After removing negative gain stocks:')
    print('mu:', len(mu_final))
    print('S:', S_final.shape)
  else:
    mu_final = mu_copy
    S_final = S_copy

  final_tickers = S_final.columns

  ticket_to_buy_json = do_optimization(mu_final, S_final, final_tickers, BASE_LINE_HORIZON, max_weight)
  return mu_final, S_final, final_tickers, ticket_to_buy_json


def main(argv):
  periods = None

  try:
      opts, args = getopt.getopt(argv, "p:uo:d:", ["periods=", "output=", "data-dir-prefix="])
  except getopt.GetoptError:
    logger.error('usage: python inference.py --data-dir-prefix <datadir prefix like processed_data> --periods <period list> --output <output directory>')
    sys.exit(2)
    
  data_dir_prefix = "processed_data"
  for opt, arg in opts:
    if opt in ("-p", "--periods"):
      periods = [int(period) for period in arg.split(',')]
    elif opt in ("-o", "--output"):
      output_dir = arg
      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
      logger.info(f'Output directory set to {output_dir}')
    elif opt in ("-d", "--data-dir-prefix"):
      data_dir_prefix = arg
      

  if periods is None:
    logger.error('usage: python inference.py --periods <period list> --output <output directory>')
    sys.exit
    
  if output_dir is None:
    logger.error('usage: python inference.py --periods <period list> --output <output directory>')
    sys.exit

  all_errors, all_mu = get_all_errors_and_mu(periods, data_dir_prefix)
  final_tickers = list(all_errors.columns)
  all_errors = all_errors.fillna(method='ffill').fillna(method='bfill')
  S = get_shrinkage_covariance(all_errors)


  logger.info('Starting optimization long only')
  ticket_to_buy = do_optimization(all_mu, S , final_tickers,
                                       BASE_LINE_HORIZON, 0.01)


  save_json_to_dir(ticket_to_buy, output_dir)

  logger.info('Starting optimization long and short')
  all_errors_short, all_mu_positive = get_errors_mu_short(all_errors, all_mu)
  S_short = get_shrinkage_covariance(all_errors_short)
  ticket_to_buy_short = do_optimization(all_mu_positive, S_short, final_tickers,
                                              BASE_LINE_HORIZON, 0.01)


  weight_short = 0
  weight_long = 0

  for stock in ticket_to_buy_short:
      index = final_tickers.index(stock['id'])
      updated_weight =  update_stock_operation_and_weight(stock, index, all_mu)
      if updated_weight < 0:
          weight_short += updated_weight
      else:
          weight_long += updated_weight

  logger.info(f"Short weight: {weight_short}, Long weight: {weight_long}")

  save_json_to_dir(ticket_to_buy_short, output_dir + '_short')


if __name__ == "__main__":
    main(sys.argv[1:])
