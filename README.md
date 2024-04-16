# AI vs Human brain on Stock market
This research project is to see if AI could beat human brain on the stock market.
The stocks could be chosen from 4 markets: US, Sweden, Germany and British.


## Rule
Each AI and Human brain decide a financial portfolio. AI choose the stocks based on Random Forest models and Mean-Variance optimization algorithm. The human choose stocks based on his experience.

Both players can only change the portfolio during the second week of each month. By the end of every month the human needs to make a summary and see who wins.

Every time each player change the asset portfolio, the trading_log.txt must be updated.

No loss stop is allowed.

## How to run AI model
- Download the price data by running the notebook: download_price_data.ipynb
- Find the optimized portfolio by running the notebook: predict_model_rf.ipynb


