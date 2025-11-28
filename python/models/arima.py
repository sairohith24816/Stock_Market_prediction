import pymongo
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import calendar
import pmdarima as pm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import adfuller
from config.loader import config

# Connect to MongoDB
client = pymongo.MongoClient(config['database']['uri'])
db = client[config['database']['name']]


def generate_predictions(stock_data, ticker):
    # Create DataFrame from stock_data
    print(f"Processing {ticker}...")
    
    # Convert MongoDB documents to DataFrame
    stock_df = pd.DataFrame(stock_data)
    stock_df = stock_df[['index', 'close']]  # Select relevant columns
    stock_df['index'] = pd.to_datetime(stock_df['index'])
    stock_df.set_index('index', inplace=True)
    stock_df.dropna(inplace=True)
    
    # 1. Clean and set frequency to Business Days
    stock_df = stock_df.asfreq('B')
    stock_df['close'] = stock_df['close'].ffill()
    
    # 2. Use Log Returns for Stationarity
    stock_df['log_close'] = np.log(stock_df['close'])
    stock_df['log_ret'] = stock_df['log_close'].diff()
    stock_df.dropna(inplace=True)

    arima_config = config['models']['arima']
    train_split = arima_config['train_split_ratio']
    
    train_size = int(len(stock_df) * train_split)
    train = stock_df.iloc[:train_size]
    test = stock_df.iloc[train_size:]
    
    y_train = train['log_ret']

    # Prepare holders for fallback model
    model_fit = None
    
    # Box-Jenkins Methodology: Use Auto-ARIMA to find optimal (p, d, q)
    try:
        print(f"Finding optimal ARIMA order for {ticker}...")
        # auto_arima will automatically determine stationarity (d) and optimal p, q
        model_auto = pm.auto_arima(y_train, 
                                   start_p=0, start_q=0,
                                   max_p=5, max_q=5,
                                   m=1,              # frequency of series (1 for non-seasonal)
                                   d=0,              # we already differenced (returns)
                                   seasonal=False,   # No seasonality
                                   start_P=0, 
                                   D=0, 
                                   trace=False,
                                   error_action='ignore',  
                                   suppress_warnings=True, 
                                   stepwise=True)
        
        print(f"Optimal ARIMA order for {ticker}: {model_auto.order}")
        
        # Fit the model with the optimal order using statsmodels
        model = ARIMA(y_train, order=model_auto.order)
        model_fit = model.fit()

        # --- 1. In-Sample Predictions (Training Set) ---
        # Get fitted values (predicted log returns) for the training set
        train_pred_log_ret = model_fit.fittedvalues
        
        # Reconstruct prices for training set
        # We need the starting price. The first log return corresponds to index 1.
        # price[t] = price[t-1] * exp(ret[t])
        # For t=0 (first point of train), we don't have a prediction from differencing, 
        # so we can just use the actual price or skip it.
        
        train_pred_prices = []
        # The fittedvalues index matches y_train index
        
        # We need the price at t-1 for each t in y_train.
        # y_train starts at index 1 of 'train' dataframe (because of diff).
        # So for y_train[i], the previous price is train['close'].iloc[i] (since y_train is shifted by 1)
        # Wait, y_train = log_close.diff(). So y_train.iloc[0] is log_close[1] - log_close[0].
        # Its index is train.index[1].
        
        # To reconstruct: pred_price[i] = train['close'].iloc[i-1] * exp(pred_ret[i])
        # This aligns with the "one-step-ahead" in-sample prediction nature.
        
        train_actuals_shifted = train['close'].shift(1).iloc[1:] # align with y_train
        train_pred_log_ret_aligned = train_pred_log_ret # should match y_train length
        
        # If lengths differ slightly due to how statsmodels handles start, we align by index
        common_index = train_actuals_shifted.index.intersection(train_pred_log_ret.index)
        train_pred_prices_series = train_actuals_shifted.loc[common_index] * np.exp(train_pred_log_ret.loc[common_index])
        
        # --- 2. Rolling Forecast (Test Set) ---
        # We predict one step ahead, then "observe" the actual value, then predict next.
        # This is much more realistic for evaluating time-series models.
        
        history = list(y_train)
        test_log_ret = test['log_ret'].values
        predictions_log_ret = []
        
        # We can use the existing model parameters but update the history
        # For efficiency, we won't re-fit parameters every step (too slow), 
        # but we will filter the new observations.
        
        print(f"Starting rolling forecast for {len(test_log_ret)} steps...")
        
        # Create a new model with all data but fix the parameters to the training set's best fit
        # This is faster than re-fitting every step
        model_all = ARIMA(np.concatenate([y_train, test_log_ret]), order=model_auto.order)
        model_all_fit = model_all.filter(model_fit.params)
        
        # The 'predict' method on the filtered model gives in-sample predictions
        # We want predictions for the test indices.
        # Indices in statsmodels are 0-based. Train ends at len(y_train)-1.
        # Test starts at len(y_train).
        start_idx = len(y_train)
        end_idx = len(y_train) + len(test_log_ret) - 1
        
        # Get dynamic predictions
        predictions_log_ret = model_all_fit.predict(start=start_idx, end=end_idx, dynamic=False)
        
        # --- Reconstruct Prices for Test Set ---
        # We need to reconstruct cumulatively, but using the *actual* previous prices 
        # is cheating for a multi-step forecast, BUT for a rolling 1-step forecast 
        # (which is what we are simulating), we base t+1 on actual t.
        
        # However, to plot a continuous line that looks like a "fit", we often just 
        # apply the predicted returns to the actual previous close.
        
        # pred_price[t] = actual_price[t-1] * exp(pred_ret[t])
        
        pred_test = []
        prev_close = train['close'].iloc[-1]
        
        for ret in predictions_log_ret:
            pred_price = prev_close * np.exp(ret)
            pred_test.append(pred_price)
            # In a true rolling forecast where we trade, we update prev_close to the ACTUAL close
            # of this day (which we would know at the end of the day) for the next prediction.
            # But here we are generating a series to compare. 
            # If we want to show "how well did it predict t given t-1", we update prev_close.
            
            # To match the test_log_ret index, we need the actuals corresponding to these predictions
            # We can't easily get them inside this loop without iterating test['close']
            pass

        # Let's do it vectorised:
        # Shifted actual closes (yesterday's close)
        # We need the close price just before the test set starts
        last_train_close = train['close'].iloc[-1]
        actual_test_closes = test['close'].values
        
        # Previous closes for the test set: [last_train, test[0], test[1], ... test[-2]]
        prev_closes = np.concatenate([[last_train_close], actual_test_closes[:-1]])
        
        pred_test = prev_closes * np.exp(predictions_log_ret)
        pred_test = pd.Series(pred_test, index=test.index)

        # --- Future Forecast (True Out-of-Sample) ---
        # For the future, we don't have actuals, so we must use the model's own predictions recursively
        # or simply use the standard forecast method from the end of the full dataset.
        
        n_future_days = 10
        # Forecast from the end of the filtered model (which includes test data)
        ret_forecast_future = model_all_fit.forecast(steps=n_future_days)
        
        # Reconstruct future prices
        last_actual_close = test['close'].iloc[-1]
        log_price_forecast = np.log(last_actual_close) + np.cumsum(ret_forecast_future)
        pred_future = np.exp(log_price_forecast)

    except Exception as e:
        print(f"ARIMA fitting failed for {ticker}: {e}")
        # Fallback: Simple Moving Average or Linear Trend on Log Prices
        # Here we use a simple drift based on recent history
        last_price = train['close'].iloc[-1]
        pred_test = pd.Series([last_price] * len(test), index=test.index)
        pred_future = pd.Series([last_price] * 10) # Flat line fallback

    # Compute MSE for the test predictions
    # Align indices for MSE calculation
    mse = mean_squared_error(test['close'], pred_test)

    # Build prediction documents
    pred_documents = []
    
    # Training period predictions (In-Sample)
    for date, close_pred in train_pred_prices_series.items():
        pred_doc = {
            'index': date.strftime("%Y-%m-%d"),
            'close': float(close_pred),
            'ticker': ticker,
            'MSE': mse, # Use test MSE as a general metric, or calculate train MSE if needed
            'type': 'train' # Optional marker
        }
        pred_documents.append(pred_doc)
    
    # Test period predictions
    for date, close_pred in zip(test.index, pred_test):
        pred_doc = {
            'index': date.strftime("%Y-%m-%d"),
            'close': float(close_pred),
            'ticker': ticker,
            'MSE': mse
        }
        pred_documents.append(pred_doc)

    # Future predictions
    future_dates = pd.date_range(
        start=test.index[-1] + pd.tseries.offsets.BDay(1),
        periods=10,
        freq='B',
    )
    
    for date, close_pred in zip(future_dates, pred_future):
        pred_documents.append({
            'index': date.strftime("%Y-%m-%d"),
            'close': float(close_pred),
            'ticker': ticker,
            'MSE': mse,
            'future': True
        })

    return pred_documents


# Execution logic moved to main.py
if __name__ == "__main__":
    for collection_name in db.list_collection_names():

        collection = db[collection_name]
        # Get current date
        current_date = datetime.now()
        start_date = current_date - timedelta(days=5*365)
        query = {'index': {'$gte': start_date}}
        stock_data = list(collection.find(query))

        if len(stock_data) < 50:
            continue  # Skip to the next collection
        pred_docs = generate_predictions(stock_data, collection_name)
        pred_collection_name = f"{collection_name}_ARIMA_predicted"
        pred_collection = db[pred_collection_name]
        pred_collection.delete_many({})
        pred_collection.insert_many(pred_docs)

    print("Done processing all collections.")
