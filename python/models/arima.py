import pymongo
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import calendar
# import pmdarima as pm  # Commented out due to NumPy compatibility issues
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import adfuller

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["stocks"]


def generate_predictions(stock_data, ticker):
    # Create DataFrame from stock_data
    print(f"Processing {ticker}...")
    
    # Convert MongoDB documents to DataFrame
    stock_df = pd.DataFrame(stock_data)
    stock_df = stock_df[['index', 'close']]  # Select relevant columns
    stock_df['index'] = pd.to_datetime(stock_df['index'])
    stock_df.set_index('index', inplace=True)
    stock_df.dropna(inplace=True)
    
    train_size = int(len(stock_data) * 4/5)  
    train = stock_df[:train_size]
    train_resampled = train.resample('D').mean()
    train_filled = train_resampled.interpolate(method='linear')
    test = stock_df.iloc[-(len(stock_data) - train_size + 1):]

    # Prepare holders for fallback model
    lr_model = None
    model_fit = None
    model_used_arima = False

    # Simple ARIMA model approach (without auto_arima due to compatibility issues)
    # Using a basic ARIMA(1,1,1) model for simplicity
    try:
        model = ARIMA(train_filled['close'], order=(1, 1, 1))
        model_fit = model.fit()
        model_used_arima = True

        # Make predictions for test period (number of steps = length of test)
        pred = model_fit.forecast(steps=test.shape[0])
    except Exception as e:
        print(f"ARIMA fitting failed for {ticker}: {e}")
        # Fallback to simple linear trend if ARIMA fails
        X = np.arange(len(train_filled)).reshape(-1, 1)
        y = train_filled['close'].values
        lr_model = LinearRegression().fit(X, y)

        # Predict for test period
        X_test = np.arange(len(train_filled), len(train_filled) + test.shape[0]).reshape(-1, 1)
        pred = lr_model.predict(X_test)

    # Align actual prices to prediction length
    actual_prices = stock_df['close'][-len(pred):]

    # Compute MSE for the test predictions
    mse = mean_squared_error(actual_prices, pred)

    # Build prediction documents for the test period
    pred_documents = []
    for date, close_pred in zip(test.index, pred):
        pred_doc = {
            'index': date.strftime("%Y-%m-%d"),  # Convert datetime to string
            'close': float(close_pred),
            'ticker': ticker,
            'MSE': mse
        }
        pred_documents.append(pred_doc)

    # --- Added: future prediction helper (n_days ahead) ---
    def predict_future_arima(model_fit, lr_model, train_filled, test, n_days=7):
        """
        Return n_days future predictions as a list of documents with 'future': True.
        Tries to use fitted ARIMA model if available; otherwise uses linear regression fallback.
        Future dates are calendar days (can be changed to business days if desired).
        """
        future_docs = []
        last_date = stock_df.index[-1]

        if model_fit is not None:
            try:
                future_pred = model_fit.forecast(steps=n_days)
            except Exception as e:
                # If ARIMA forecast fails unexpectedly, fallback to linear regression
                future_pred = None
                print(f"ARIMA forecast failed for future steps for {ticker}: {e}")
        else:
            future_pred = None

        if future_pred is None:
            # Use linear regression fallback if ARIMA unavailable or forecast failed
            if lr_model is None:
                # Train a fresh linear model on train_filled if none exists
                X_full = np.arange(len(train_filled)).reshape(-1, 1)
                y_full = train_filled['close'].values
                lr_model_local = LinearRegression().fit(X_full, y_full)
            else:
                lr_model_local = lr_model

            # Determine the start index for future predictions:
            # if test has length L, previous X_test started at len(train_filled) and had L steps,
            # so the next index starts at len(train_filled) + L
            start_idx = len(train_filled) + test.shape[0]
            X_future = np.arange(start_idx, start_idx + n_days).reshape(-1, 1)
            future_pred = lr_model_local.predict(X_future)

        # Attach future docs
        for i, close_pred in enumerate(future_pred):
            future_date = last_date + timedelta(days=(i + 1))  # calendar day
            future_docs.append({
                'index': future_date.strftime("%Y-%m-%d"),
                'close': float(close_pred),
                'ticker': ticker,
                'MSE': mse,
                'future': True
            })
        return future_docs

    # Generate and append next 7 calendar-day predictions
    future_predictions = predict_future_arima(model_fit, lr_model, train_filled, test, n_days=10)
    pred_documents.extend(future_predictions)
    # --- End added future predictions ---

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
