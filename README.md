# AssetScout 

## Pre-requisites
Create an Python environment with below libraries:

scikit-learn              1.3.2
pandas                    2.1.4
pandas-datareader         0.10.0
numpy                     1.26.2
yfinance                  0.2.38
optuna                    3.4.0

## How to run AI model
1. Run 'python train_model.py' to create the models.
2. run 'python inference.py' to find out the optimized portfolio.


# AI vs Human brain on Stock market
This research project is to see if AI could beat human brain on the stock market.
The stocks could be chosen from 4 markets: US, Sweden, Germany and British.


## Rule
Each AI and Human brain decide a financial portfolio. AI choose the stocks based on Random Forest models and Mean-Variance optimization algorithm. The human choose stocks based on his experience.

Both players can only change the portfolio during the second week of each month. By the end of every month the human needs to make a summary and see who wins.

Every time each player change the asset portfolio, the trading_log.txt must be updated.

No loss stop is allowed.



