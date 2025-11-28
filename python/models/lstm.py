import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import pymongo
from config.loader import config

# Connect to MongoDB
client = pymongo.MongoClient(config['database']['uri'])
db = client[config['database']['name']]

# Prepare data for LSTM
def create_dataset(dataset, time_steps):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), :])
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)

# Choose number of time steps for LSTM
def train_model_with_steps(data, step):
    # Choose number of time steps for LSTM
    time_steps = step
    
    lstm_config = config['models']['lstm']

    # Create input sequences and labels
    X, y = create_dataset(data, time_steps)

    # Split data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=lstm_config['test_split'], shuffle=False)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=lstm_config['units'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=lstm_config['units'], return_sequences=False))
    model.add(Dense(units=1))

    # adam optimizer with learning rate
    adam = Adam(learning_rate=lstm_config['learning_rate'])

    # Compile model
    model.compile(optimizer=adam, loss='mean_squared_error')

    # Train model
    model.fit(X_train, y_train, epochs=lstm_config['epochs'], batch_size=lstm_config['batch_size'], verbose=0)

    # Evaluate model
    mse = model.evaluate(X_test, y_test, verbose=0)
    return mse

def generate_predictions(stock_data, ticker):
    # Create DataFrame from stock_data
    print(f"Processing {ticker}...")

    # Convert MongoDB documents to DataFrame
    stock_df = pd.DataFrame(stock_data)
    stock_df = stock_df[['index', 'open', 'high', 'low', 'close', 'volume']]  # Select relevant columns
    stock_df['index'] = pd.to_datetime(stock_df['index'])
    stock_df.set_index('index', inplace=True)
    stock_df.dropna(inplace=True)

    # Normalize data using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Use 'Close' price
    closing_prices = stock_df['close'].values.reshape(-1, 1)
    final_data = scaler.fit_transform(closing_prices)
    
    lstm_config = config['models']['lstm']
    train_data, test_data, _, _ = train_test_split(final_data, final_data, test_size=lstm_config['test_split'], shuffle=False)

    # Iterate over different step values and find the best one (commented out in original)
    best_step = None
    best_mse = float('inf')
    
    lstm_config = config['models']['lstm']
    time_steps = lstm_config['time_steps']

    # Train the final model with the best step value (original used step=1)
    X_final, y_final = create_dataset(final_data, time_steps)
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y_final, test_size=lstm_config['test_split'], shuffle=False)

    # adam optimizer with learning rate
    adam = Adam(learning_rate=lstm_config['learning_rate'])

    final_model = Sequential()
    final_model.add(LSTM(units=lstm_config['units'], return_sequences=True, input_shape=(X_train_final.shape[1], X_train_final.shape[2])))
    final_model.add(LSTM(units=lstm_config['units'], return_sequences=False))
    final_model.add(Dense(units=1))
    final_model.compile(optimizer=adam, loss='mean_squared_error')
    final_model.fit(X_train_final, y_train_final, epochs=lstm_config['epochs'], batch_size=lstm_config['batch_size'], verbose=1)

    # Predictions on Test Set
    y_pred_scaled = final_model.predict(X_test_final)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    
    # Predictions on Training Set (In-Sample)
    y_train_pred_scaled = final_model.predict(X_train_final)
    y_train_pred = scaler.inverse_transform(y_train_pred_scaled)

    pred_documents = []
    
    # Add Training Predictions
    # X_train_final corresponds to the first part of the data (minus time_steps)
    # We need to align dates.
    # The dataset creation drops the first 'time_steps' points.
    # So X_train_final[0] corresponds to predicting index 'time_steps'.
    
    train_dates = stock_df.index[time_steps : time_steps + len(y_train_pred)]
    
    for i in range(len(y_train_pred)):
        pred_doc = {
            'index': train_dates[i].strftime("%Y-%m-%d"),
            'close': float(y_train_pred[i]),
            'ticker': ticker,
            'MSE': 0.0 # Placeholder or calculate train MSE
        }
        pred_documents.append(pred_doc)

    # Get the dates from the index of the testing set
    # Test set starts after train set.
    # X_test_final starts at index: time_steps + len(y_train_pred)
    test_dates = stock_df.index[-len(y_pred):]
    actual_prices = stock_df.close[-len(y_pred):]

    mse = mean_squared_error(actual_prices, y_pred)
    # print("MSE:",mse)

    for i in range(len(y_pred)):
        pred_doc = {
            'index': test_dates[i].strftime("%Y-%m-%d"),
            'close': float(y_pred[i]),
            'ticker': ticker,
            'MSE': mse
        }
        pred_documents.append(pred_doc)

    # --- Added: iterative future prediction helper (auto-regressive) ---
    def predict_future(final_model, scaler, stock_df, ticker, n_days=7, time_steps=time_steps):
        """
        Predict next n_days using the trained final_model by rolling the last time_steps.
        Returns a list of prediction documents similar to pred_documents but marked with 'future': True.
        """
        # last observed close prices (original scale)
        last_close = stock_df['close'].values.reshape(-1, 1)
        # scale using the same scaler fitted earlier
        last_scaled = scaler.transform(last_close)

        # seed input of shape (1, time_steps, 1)
        # If there aren't enough points, pad with the earliest available scaled values
        if last_scaled.shape[0] < time_steps:
            pad_len = time_steps - last_scaled.shape[0]
            pad = np.repeat(last_scaled[0:1, :], pad_len, axis=0)
            seed_vals = np.vstack([pad, last_scaled])
        else:
            seed_vals = last_scaled[-time_steps:]

        cur_input = seed_vals.reshape(1, time_steps, 1).astype(np.float32)

        preds_scaled = []
        for _ in range(n_days):
            p = final_model.predict(cur_input, verbose=0)  # shape (1,1)
            preds_scaled.append(p[0, 0])

            # slide window: drop oldest and append predicted value
            if time_steps > 1:
                cur_input = np.concatenate([cur_input[:, 1:, :], p.reshape(1, 1, 1)], axis=1)
            else:
                cur_input = p.reshape(1, 1, 1)

        # inverse scale to original price space
        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

        last_date = stock_df.index[-1]
        future_docs = []
        for i, val in enumerate(preds):
            future_docs.append({
                'index': (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
                'close': float(val),
                'ticker': ticker,
                'future': True,
                'MSE': mse  # attach mse from test set for reference (optional)
            })
        return future_docs

    # Generate next n calendar-day predictions and append them
    n_future_days = config['data_fetching']['future_prediction_days']
    future_predictions = predict_future(final_model, scaler, stock_df, ticker, n_days=n_future_days, time_steps=X_train_final.shape[1])
    pred_documents.extend(future_predictions)
    # --- End added future-prediction code ---

    return pred_documents

# Execution logic moved to main.py
if __name__ == "__main__":
    # Main loop over collections in DB
    for collection_name in db.list_collection_names():
        # Uncomment to process a specific ticker only
        # if collection_name != "HDFCBANK.NS":
        #     continue

        predicted_collection_name = f"{collection_name}_LSTM_predicted"

        collection = db[collection_name]
        # Get current date
        current_date = datetime.now()
        history_years = config['data_fetching']['prediction_history_years']
        start_date = current_date - timedelta(days=history_years * 365)
        query = {'index': {'$gte': start_date}}
        stock_data = list(collection.find(query))
        if len(stock_data) < config['data_fetching']['min_data_points']:
            continue  # Skip to the next collection
        pred_docs = generate_predictions(stock_data, collection_name)
        pred_collection_name = f"{collection_name}_LSTM_predicted"
        pred_collection = db[pred_collection_name]
        pred_collection.delete_many({})
        pred_collection.insert_many(pred_docs)

    print("Done processing all collections.")
