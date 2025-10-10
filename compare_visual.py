# to compare model and human brain
import pandas as pd
import numpy as np
import os
import json

# the date must be working day!
model_trace_log = [
  {
    "date": "2024-05-15",
    "weights": [
      {
        "id": "STAN.L",
        "weight": 0.0988379414494704
      },
      {
        "id": "KO",
        "weight": 0.10885514872237066
      },
      {
        "id": "LLY",
        "weight": 0.16274045692599903
      },
      {
        "id": "GILD",
        "weight": 0.0982685011890021
      },
      {
        "id": "MRK",
        "weight": 0.09608726469000152
      },
      {
        "id": "NOW",
        "weight": 0.10724631420200852
      },
      {
        "id": "VRTX",
        "weight": 0.14695857799430817
      },
      {
        "id": "WMT",
        "weight": 0.18100579482683965
      }
    ]
  },
  {
    "date": "2024-05-16",
    "weights": [
      {
        "id": "CHD",
        "weight": 0.09604227989932394
      },
      {
        "id": "LLY",
        "weight": 0.15701155301162809
      },
      {
        "id": "GILD",
        "weight": 0.08309850279939462
      },
      {
        "id": "MCK",
        "weight": 0.15491295395567367
      },
      {
        "id": "MRK",
        "weight": 0.12300274846059159
      },
      {
        "id": "NOW",
        "weight": 0.0998352785021135
      },
      {
        "id": "VRTX",
        "weight": 0.10564614375465124
      },
      {
        "id": "WMT",
        "weight": 0.18045053961662333
      }
    ]
  },
  {
    "date": "2024-05-22",
    "weights": [
        {
          "id": "KO",
          "weight": 0.10274849558771942
        },
        {
          "id": "LLY",
          "weight": 0.1566002200958428
        },
        {
          "id": "GILD",
          "weight": 0.08707934623088542
        },
        {
          "id": "J",
          "weight": 0.11359767227893916
        },
        {
          "id": "MRK",
          "weight": 0.10567158852690067
        },
        {
          "id": "NOW",
          "weight": 0.11206722758523373
        },
        {
          "id": "VRTX",
          "weight": 0.142924247158989
        },
        {
          "id": "WMT",
          "weight": 0.17931120253548977
        }
      ]
  },
  {
    "date": "2024-06-10",
    "weights": [
        {
            "id": "STAN.L",
            "weight": 0.099809488530
        },
        {
            "id": "LLY",
            "weight": 0.141107090394177
        },
        {
            "id": "GILD",
            "weight": 0.14317362063347
        },
        {
            "id": "JKHY",
            "weight": 0.13209880165885
        },
        {
            "id": "MCK",
            "weight": 0.098208917045372
        },
        {
            "id": "NVR",
            "weight": 0.177313820149842
        },
        {
            "id": "NOW",
            "weight": 0.087160539294239
        },
        {
            "id": "VRTX",
            "weight": 0.12112772229333
        }
    ]
  },
  {
    "date": "2024-08-08",
    "weights": [
    {
        "id": "STAN.L",
        "weight": 0.06940932033704353
    },
    {
        "id": "AMGN",
        "weight": 0.07431243214457392
    },
    {
        "id": "BIIB",
        "weight": 0.042900139778976665
    },
    {
        "id": "COST",
        "weight": 0.03558101407191769
    },
    {
        "id": "CTRA",
        "weight": 0.05635014005671591
    },
    {
        "id": "LLY",
        "weight": 0.03948522028075085
    },
    {
        "id": "GILD",
        "weight": 0.1454264331821711
    },
    {
        "id": "HOLX",
        "weight": 0.05758084586182881
    },
    {
        "id": "JBL",
        "weight": 0.005493340409292151
    },
    {
        "id": "JKHY",
        "weight": 0.1526600094065125
    },
    {
        "id": "MCK",
        "weight": 0.07232195285376533
    },
    {
        "id": "NVR",
        "weight": 0.06949839934412294
    },
    {
        "id": "PGR",
        "weight": 0.04417625540632342
    },
    {
        "id": "NOW",
        "weight": 0.05764335559940242
    },
    {
        "id": "STE",
        "weight": 0.0076256697455101465
    },
    {
        "id": "SMCI",
        "weight": 0.042851698736872286
    },
    {
        "id": "TTWO",
        "weight": 0.010628803276210979
    },
    {
        "id": "VRTX",
        "weight": 0.01605496950800943
    }
  ]
  },
  {
    "date": "2024-08-19",
    "weights":  [
      {
          "id": "STAN.L",
          "weight": 0.02090530701681672
      },
      {
          "id": "CTRA",
          "weight": 0.039525249097928664
      },
      {
          "id": "LLY",
          "weight": 0.07781991214048126
      },
      {
          "id": "GILD",
          "weight": 0.103091631334796
      },
      {
          "id": "JKHY",
          "weight": 0.19322764738301704
      },
      {
          "id": "NVR",
          "weight": 0.19999999999999962
      },
      {
          "id": "SMCI",
          "weight": 0.06095072592843817
      },
      {
          "id": "UNH",
          "weight": 0.0997655409399469
      },
      {
          "id": "VRTX",
          "weight": 0.02215266539408572
      },
      {
          "id": "WMT",
          "weight": 0.07527618703969421
      },
      {
          "id": "600276.SS",
          "weight": 0.02574741655617227
      },
      {
          "id": "601225.SS",
          "weight": 0.08108564359822486
      }
  ]
  },
  {
      "date": "2024-08-27",
      "weights":   [
        {
            "id": "RHM.DE",
            "weight": 0.011395852207041484
        },
        {
            "id": "FRAS.L",
            "weight": 0.09640906381745395
        },
        {
            "id": "CTRA",
            "weight": 0.03023819619758223
        },
        {
            "id": "LLY",
            "weight": 0.058479654890121
        },
        {
            "id": "GILD",
            "weight": 0.16904811360896962
        },
        {
            "id": "JKHY",
            "weight": 0.2
        },
        {
            "id": "PGR",
            "weight": 0.06012139904756649
        },
        {
            "id": "SMCI",
            "weight": 0.07466780405733559
        },
        {
            "id": "VRTX",
            "weight": 0.026939136514627482
        },
        {
            "id": "WMT",
            "weight": 0.09133278380032214
        },
        {
            "id": "600150.SS",
            "weight": 0.018772858604274755
        },
        {
            "id": "600519.SS",
            "weight": 0.12040575367173317
        },
        {
            "id": "600900.SS",
            "weight": 0.0014987798535868154
        },
        {
            "id": "601225.SS",
            "weight": 0.04069060372937267
        }
    ]}
]



def load_data(trace_log_dir, trace_log):
  folder = trace_log_dir
  # the list of the json files
  files = os.listdir(folder)
  # sort the files
  files.sort()

  for file in files:
      # get the date from the file name
      date = file.split('.')[0].split('_')[-1]
      # convert the date from YYYYMMDD to YYYY-MM-DD
      date = '-'.join([date[:4], date[4:6], date[6:]])
      #print('loading data for', date)
      # read the json
      with open(os.path.join(folder, file)) as f:
          data = json.load(f)

      if '_' in data[0]["id"]:
            # Dictionary to store combined weights
            combined_weights = {}
            for stock in data:
                stock_id = stock["id"].split('_')[0]
                weight = stock["weight"]
                if stock_id in combined_weights:
                    combined_weights[stock_id] += weight
                else:
                    combined_weights[stock_id] = weight
            # Convert the dictionary back to a list of dictionaries
            data = [{"id": stock_id, "weight": weight} for stock_id, weight in combined_weights.items()]


      # extract the weights
      trace_log.append(
          {
              "date": date,
              "weights": data
          }
      )
  return trace_log


# the folder containing the computed portfolios
folder = 'processed_data_128/computed_portfolios'
# the list of the json files
model_trace_log = load_data(folder, model_trace_log)
#model_trace_log
model_multi_horizon_trace_log = []
model_multi_horizon_trace_log = load_data('multi_horizon', model_multi_horizon_trace_log)

model_short_log = []
model_short_log = load_data('processed_data_128/computed_portfolios_short', model_short_log)

model_short_v2_log = []
model_short_v2_log = load_data('multi_horizon_short', model_short_v2_log)

model_v2025_log = []
model_v2025_log = load_data('multi_horizon_v2025', model_v2025_log)

model_short_v2025_log = []
model_short_v2025_log = load_data('multi_horizon_v2025_short', model_short_v2025_log)


human_trace_log = []
human_trace_log = load_data('human', human_trace_log)

unique_ids = set()
for log in model_trace_log:
    for asset in log['weights']:
        # remove the minus sign before the stock id
        if asset['id'][0] == '-':
            asset['id'] = asset['id'][1:]
        unique_ids.add(asset['id'])

for log in human_trace_log:
    for asset in log['weights']:
        if asset['id'][0] == '-':
            asset['id'] = asset['id'][1:]
        unique_ids.add(asset['id'])

for log in model_multi_horizon_trace_log:
    for asset in log['weights']:
        if asset['id'][0] == '-':
            asset['id'] = asset['id'][1:]
        unique_ids.add(asset['id'])

for log in model_short_log:
    for asset in log['weights']:
        unique_ids.add(asset['id'])

for log in model_short_v2_log:
    for asset in log['weights']:
        unique_ids.add(asset['id'])
        
for log in model_v2025_log:
    for asset in log['weights']:
        unique_ids.add(asset['id'])
        
for log in model_short_v2025_log:
    for asset in log['weights']:
        unique_ids.add(asset['id'])
        
        
unique_ids.add('^GSPC')

from pandas_datareader import data as pdr
import yfinance as yfin

#stocks_data = yfin.download(list(unique_ids), start='2024-05-14', end=None)['Close']
stocks_data = yfin.download(unique_ids, start='2024-05-14', auto_adjust=False, end=None)['Adj Close']

# convert to USD
from util import convert, get_currency_pair


for stock_name in list(unique_ids):
  stock_suffix = '.' + stock_name.split('.')[-1]

  exchange_name, needs_inversion, exchange_name_yahoo = get_currency_pair(stock_suffix, 'USD')
  # print(stock_suffix)
  # print(exchange_name)
  if exchange_name is not None:
    print(f'Converting {stock_name} to USD')
    df = stocks_data[[stock_name]]
    if len(df) == 0:
      print(f'No data for {stock_name}')
      continue
    df = df.rename(columns={stock_name: 'Adj Close'})
    df['Volume'] = 0
    df = df.sort_index(ascending=True)
    df = convert(df, exchange_name, needs_inversion, exchange_name_yahoo)
    stocks_data[stock_name] = df['Adj Close']
    
duplicates = stocks_data.index.duplicated()
print(stocks_data.index[duplicates])

import pandas as pd
# Assuming stocks_data is already defined and imported
# Create a range of dates from the minimum to the maximum date in the original DataFrame
stocks_data = stocks_data.sort_index()
all_dates = pd.date_range(start=stocks_data.index.min(), end=stocks_data.index.max(), freq='D')


stocks_data = stocks_data.reindex(all_dates, method='ffill')


stocks_data.fillna(method='ffill', inplace=True)
sp500_data = stocks_data['^GSPC']


# remove ^GSPC from stocks_data
stocks_data.drop(columns=['^GSPC'], inplace=True)

# remove ^GSPC from unique_ids
unique_ids.remove('^GSPC')

import numpy as np

sp500_log_return = pd.DataFrame(np.log(sp500_data)).diff()[1:]


log_return = pd.DataFrame(np.log(stocks_data)).diff()[1:]
log_return = log_return.fillna(0)


start_date = pd.to_datetime(model_trace_log[0]['date'])

def update_weight(weights_attr, unique_ids):
  weights = np.zeros(len(unique_ids))
  for asset in weights_attr:
    asset_id = asset['id']
    asset_weight = asset['weight']
    if 'operation' in asset:
      direction = asset['operation']
      if direction == 'short' and asset_weight > 0:
        asset_weight = -asset_weight
    asset_index = sorted(unique_ids).index(asset_id)
    weights[asset_index] = asset_weight
  return weights


# enumerate the dates from start_date to now in the log_return's index

cur_asset_model = 1
cur_asset_model_hist = [1]
model_contrib_hist = {stock: [] for stock in log_return.columns}
model_weights = np.zeros(len(unique_ids))

cur_asset_model_v2 = 1
cur_asset_model_v2_hist = [1]
model_v2_contrib_hist = {stock: [] for stock in log_return.columns}
model_v2_weights = np.zeros(len(unique_ids))

cur_asset_model_short = 1
cur_asset_model_short_hist = [1]
model_short_contrib_hist = {stock: [] for stock in log_return.columns}
model_short_weights = np.zeros(len(unique_ids))

cur_asset_model_short_v2 = 1
cur_asset_model_short_v2_hist = [1]
model_short_v2_contrib_hist = {stock: [] for stock in log_return.columns}
model_short_v2_weights = np.zeros(len(unique_ids))

cur_asset_model_v2025 = 1
cur_asset_model_v2025_hist = [1]
model_v2025_contrib_hist = {stock: [] for stock in log_return.columns}

cur_asset_model_short_v2025 = 1
cur_asset_model_short_v2025_hist = [1]
model_short_v2025_contrib_hist = {stock: [] for stock in log_return.columns}

cur_asset_human = 1
cur_asset_human_hist = [1]
human_contrib_hist = {stock: [] for stock in log_return.columns}
human_weights = np.zeros(len(unique_ids))



cur_asset_sp500 = 1
cur_asset_sp500_hist = [1]
sp500_contrib_hist = {stock: [] for stock in log_return.columns}




for date in log_return.index:
  # get the date as a string
  date_str = str(date.date())
  # enumerate the trade log to see if the date is in the model_trace_log
  for log in model_trace_log:
    if log['date'] == date_str:
      model_weights = update_weight(log['weights'], unique_ids)

  for log in human_trace_log:
    if log['date'] == date_str:
      human_weights = update_weight(log['weights'], unique_ids)

  # daily profit for multi-horizon model are same before 2024-10-10 as the old model
  if date < pd.Timestamp('2024-10-10'):
    model_v2_weights = model_weights
  else:
    for log in model_multi_horizon_trace_log:
      if log['date'] == date_str:
        print(f'update model_v2_weights for {date_str}')
        model_v2_weights = update_weight(log['weights'], unique_ids)


  if date < pd.Timestamp('2024-11-29'):
    model_short_weights = model_weights
  else:
    for log in model_short_log:
      if log['date'] == date_str:
        print(f'update model_short_weights for {date_str}')
        model_short_weights = update_weight(log['weights'], unique_ids)

  if date < pd.Timestamp('2024-11-29'):
    model_short_v2_weights = model_v2_weights
  else:
    for log in model_short_v2_log:
      if log['date'] == date_str:
        print(f'update model_short_weights for {date_str}')
        model_short_v2_weights = update_weight(log['weights'], unique_ids)
        
  if date < pd.Timestamp('2025-10-05'):
    model_v2025_weights = model_v2_weights
  else:
    for log in model_v2025_log:
      if log['date'] == model_v2025_log:
        print(f'update model_v2025_weights for {date_str}')
        model_v2025_weights = update_weight(log['weights'], unique_ids)
        
  if date < pd.Timestamp('2025-10-05'):
    model_short_v2025_weights = model_short_v2_weights
  else:
    for log in model_short_v2025_log:
      if log['date'] == model_short_v2025_log:
        print(f'update model_short_v2025_weights for {date_str}')
        model_short_v2025_weights = update_weight(log['weights'], unique_ids)

  daily_log_return_values = log_return.loc[date].values

  # get the daily profit
  daily_log_return_model = np.dot(model_weights, daily_log_return_values)
  daily_model_contributions = model_weights * daily_log_return_values
  for stock, contrib in zip(log_return.columns, daily_model_contributions):
      model_contrib_hist[stock].append(contrib)

  cur_asset_model = cur_asset_model * np.exp(daily_log_return_model)
  cur_asset_model_hist.append(cur_asset_model)

  # model multiple horizon
  daily_log_return_model_v2 = np.dot(model_v2_weights, daily_log_return_values)
  daily_model_v2_contributions = model_v2_weights * daily_log_return_values
  for stock, contrib in zip(log_return.columns, daily_model_v2_contributions):
    model_v2_contrib_hist[stock].append(contrib)
  cur_asset_model_v2 = cur_asset_model_v2 * np.exp(daily_log_return_model_v2)
  cur_asset_model_v2_hist.append(cur_asset_model_v2)

  # model short
  daily_log_return_model_short = np.dot(model_short_weights, daily_log_return_values)
  daily_model_short_contributions = model_short_weights * daily_log_return_values
  for stock, contrib in zip(log_return.columns, daily_model_short_contributions):
    model_short_contrib_hist[stock].append(contrib)

  cur_asset_model_short = cur_asset_model_short * np.exp(daily_log_return_model_short)
  cur_asset_model_short_hist.append(cur_asset_model_short)

  # model short multiple horizon
  daily_log_return_model_short_v2 = np.dot(model_short_v2_weights, daily_log_return_values)
  daily_model_short_v2_contributions = model_short_v2_weights * daily_log_return_values
  for stock, contrib in zip(log_return.columns, daily_model_short_v2_contributions):
    model_short_v2_contrib_hist[stock].append(contrib)

  cur_asset_model_short_v2 = cur_asset_model_short_v2 * np.exp(daily_log_return_model_short_v2)
  cur_asset_model_short_v2_hist.append(cur_asset_model_short_v2)

  # model v2025
  daily_log_return_model_v2025 = np.dot(model_v2025_weights, daily_log_return_values)
  daily_model_v2025_contributions = model_v2025_weights * daily_log_return_values
  for stock, contrib in zip(log_return.columns, daily_model_v2025_contributions):
    model_v2025_contrib_hist[stock].append(contrib)

  cur_asset_model_v2025 = cur_asset_model_v2025 * np.exp(daily_log_return_model_v2025)
  cur_asset_model_v2025_hist.append(cur_asset_model_v2025)

  # model short v2025
  daily_log_return_model_short_v2025 = np.dot(model_short_v2025_weights, daily_log_return_values)
  daily_model_short_v2025_contributions = model_short_v2025_weights * daily_log_return_values
  for stock, contrib in zip(log_return.columns, daily_model_short_v2025_contributions):
    model_short_v2025_contrib_hist[stock].append(contrib)

  cur_asset_model_short_v2025 = cur_asset_model_short_v2025 * np.exp(daily_log_return_model_short_v2025)
  cur_asset_model_short_v2025_hist.append(cur_asset_model_short_v2025)

  # human brain
  daily_log_return_human = np.dot(human_weights, daily_log_return_values)
  daily_human_contributions = human_weights * daily_log_return_values
  for stock, contrib in zip(log_return.columns, daily_human_contributions):
      human_contrib_hist[stock].append(contrib)

  cur_asset_human = cur_asset_human * np.exp(daily_log_return_human)
  cur_asset_human_hist.append(cur_asset_human)

  # if the date is in the human_brain_trace
  daily_log_return_sp500 = sp500_log_return.loc[date].values[0]
  cur_asset_sp500 = cur_asset_sp500 * np.exp(daily_log_return_sp500)
  cur_asset_sp500_hist.append(cur_asset_sp500)


# After processing, convert contribution histories to dataframe for easy analysis
model_contrib_df = pd.DataFrame(model_contrib_hist, index=log_return.index)
model_v2_contrib_df = pd.DataFrame(model_v2_contrib_hist, index=log_return.index)
model_short_contrib_df = pd.DataFrame(model_short_contrib_hist, index=log_return.index)
model_short_v2_contrib_df = pd.DataFrame(model_short_v2_contrib_hist, index=log_return.index)

human_contrib_df = pd.DataFrame(human_contrib_hist, index=log_return.index)

# Calculate sum of contributions
total_model_contrib = model_contrib_df.sum()
total_model_v2_contrib = model_v2_contrib_df.sum()
total_model_short_contrib = model_short_contrib_df.sum()
total_model_short_v2_contrib = model_short_v2_contrib_df.sum()
total_human_contrib = human_contrib_df.sum()

# Get the top 5 contributors
top5_model_contrib = total_model_contrib.sort_values(ascending=False).head(5)
top5_model_v2_contrib = total_model_v2_contrib.sort_values(ascending=False).head(5)
top5_model_short_contrib = total_model_short_contrib.sort_values(ascending=False).head(5)
top5_model_short_v2_contrib = total_model_short_v2_contrib.sort_values(ascending=False).head(5)
top5_human_contrib = total_human_contrib.sort_values(ascending=False).head(5)

# Get the bottom 5 contributors
bottom5_model_contrib = total_model_contrib.sort_values().head(5)
bottom5_model_v2_contrib = total_model_v2_contrib.sort_values().head(5)
bottom5_model_short_contrib = total_model_short_contrib.sort_values().head(5)
bottom5_model_short_v2_contrib = total_model_short_v2_contrib.sort_values().head(5)
bottom5_human_contrib = total_human_contrib.sort_values().head(5)

print('cur_asset_model')
print(cur_asset_model)
print('cur_asset_model_v2')
print(cur_asset_model_v2)
print('cur_asset_model_short')
print(cur_asset_model_short)
print('cur_asset_model_short_v2')
print(cur_asset_model_short_v2)
print('cur_asset_human')
print(cur_asset_human)
print('cur_asset_sp500')
print(cur_asset_sp500)
print("Top 5 Model Contributors:\n", top5_model_contrib)
print("Bottom 5 Model Contributors:\n", bottom5_model_contrib)
print("Top 5 Model V2 Contributors:\n", top5_model_v2_contrib)
print("Bottom 5 Model Contributors:\n", bottom5_model_v2_contrib)
print("Top 5 Model Short Contributors:\n", top5_model_short_contrib)
print("Bottom 5 Model Contributors:\n", bottom5_model_short_contrib)
print("Top 5 Model Short V2 Contributors:\n", top5_model_short_contrib)
print("Bottom 5 Model V2 Contributors:\n", bottom5_model_short_contrib)
print("Top 5 Human Contributors:\n", top5_human_contrib)
print("Bottom 5 Human Contributors:\n", bottom5_human_contrib)

def add_vertical_line(fig, date, annotation, cur_asset_model_hist, cur_asset_human_hist, cur_asset_sp500_hist):
    # Adding a vertical line and annotation
    combined_hist = cur_asset_model_hist + cur_asset_human_hist + cur_asset_sp500_hist

    fig.add_shape(
        dict(
            type="line",
            x0=date,
            y0=min(combined_hist),
            x1=date,
            y1=max(combined_hist),
            line=dict(
                color="blue",
                width=1
            )
        )
    )

    fig.add_annotation(
        dict(
            x=date,
            y=max(combined_hist),
            text=annotation,
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40
        )
    )
    
    
    # use plotly to plot the curve of the three assets
import plotly.graph_objects as go

fig = go.Figure()
date_list = log_return.index.tolist()
date_list.insert(0, pd.Timestamp('2024-05-14'))

#fig.add_trace(go.Scatter(x=date_list, y=cur_asset_model_hist, mode='lines', name='Model'))
fig.add_trace(go.Scatter(x=date_list, y=cur_asset_model_v2_hist, mode='lines', name='Model Multi'))
#fig.add_trace(go.Scatter(x=date_list, y=cur_asset_model_short_hist, mode='lines', name='Model Short'))
fig.add_trace(go.Scatter(x=date_list, y=cur_asset_model_short_v2_hist, mode='lines', name='Model Short Multi'))
#fig.add_trace(go.Scatter(x=date_list, y=cur_asset_model_shorter_short_v2_hist, mode='lines', name='Model Shorter Short Multi'))
#fig.add_trace(go.Scatter(x=date_list, y=cur_asset_model_perplexity_hist, mode='lines', name='Perplexity.ai'))
#fig.add_trace(go.Scatter(x=date_list, y=cur_asset_model_perplexity_r1_hist, mode='lines', name='Perplexity.ai-R1'))
fig.add_trace(go.Scatter(x=date_list, y=cur_asset_human_hist, mode='lines', name='Human'))
fig.add_trace(go.Scatter(x=date_list, y=cur_asset_sp500_hist, mode='lines', name='SP500'))
fig.add_trace(go.Scatter(x=date_list, y=cur_asset_model_v2025_hist, mode='lines', name='Model Multi v2025'))
fig.add_trace(go.Scatter(x=date_list, y=cur_asset_model_short_v2025_hist, mode='lines', name='Model Short Multi v2025'))

add_vertical_line(fig, '2024-07-20', 'global selectKBest', cur_asset_model_hist, cur_asset_human_hist, cur_asset_sp500_hist)
add_vertical_line(fig, '2024-10-10', 'multi-horizon', cur_asset_model_hist, cur_asset_human_hist, cur_asset_sp500_hist)
add_vertical_line(fig, '2024-11-30', 'Short models', cur_asset_model_hist, cur_asset_human_hist, cur_asset_sp500_hist)
add_vertical_line(fig, '2025-10-05', 'v2025!', cur_asset_model_hist, cur_asset_human_hist, cur_asset_sp500_hist)
fig.write_image("output_chart.png", width=1200, height=800)
