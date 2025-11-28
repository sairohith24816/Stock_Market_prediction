import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pymongo

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["stocks"]

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.fc1(x))
        x = torch.mm(adj, x)
        x = self.fc2(x)
        return x

def train_model(model, adj, features, labels, epochs=100, lr=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

def predict(model, adj, features):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
    return output

def generate_predictions(stock_data, ticker):
    print(f"Processing {ticker}...")

    # Convert MongoDB documents to DataFrame
    stock_df = pd.DataFrame(stock_data)
    stock_df = stock_df[['index', 'open', 'high', 'low', 'close', 'volume']]  # Select relevant columns
    stock_df['index'] = pd.to_datetime(stock_df['index'])
    stock_df.set_index('index', inplace=True)
    stock_df.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    closing_prices = stock_df['close'].values.reshape(-1, 1)
    final_data = scaler.fit_transform(closing_prices)

    # Prepare features and labels (current script used the same scaler variable repeatedly)
    features = stock_df['open'].values.reshape(-1, 1)
    labels = stock_df['close'].values.reshape(-1, 1)
    # Note: following lines mirror the original script's behavior (re-fitting scaler)
    features = scaler.fit_transform(features)
    labels = scaler.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
    adjacency_matrix_train = np.eye(len(X_train))
    adjacency_matrix_test = np.eye(len(X_test))

    features_train = torch.FloatTensor(X_train)
    features_test = torch.FloatTensor(X_test)
    labels_train = torch.FloatTensor(y_train)
    labels_test = torch.FloatTensor(y_test)
    adj_tensor_train = torch.FloatTensor(adjacency_matrix_train)
    adj_tensor_test = torch.FloatTensor(adjacency_matrix_test)

    model = GCN(input_dim=1, hidden_dim=64, output_dim=1)
    train_model(model, adj_tensor_train, features_train, labels_train)  # features = labels as we used only closed price

    predicted_normalized = predict(model, adj_tensor_test, features_test)
    predicted_test = scaler.inverse_transform(predicted_normalized.numpy())

    test_dates = stock_df.index[-len(predicted_test):]
    actual_prices = stock_df['close'][-len(predicted_test):]

    mse = mean_squared_error(actual_prices, predicted_test)
    print("MSE:", mse)

    pred_documents = [{'index': date.strftime("%Y-%m-%d"), 'close': float(pred), 'ticker': ticker, 'MSE': mse}
                      for date, pred in zip(test_dates, predicted_test)]

    # --- Added: simple iterative future-prediction helper (auto-regressive on a single-node GCN) ---
    def predict_future_gcn(model, scaler, X_test, n_days=20, ticker=ticker):
        """
        Predict next n_days using the trained GCN by repeatedly feeding a single-node feature.
        It uses the last scaled feature from X_test as the input for all future steps (simple approach).
        Returns a list of prediction documents marked with 'future': True.
        """
        future_docs = []
        # Use the last scaled feature from X_test as a seed (shape (1,))
        if len(X_test) == 0:
            return future_docs

        last_scaled_feature = X_test[-1].reshape(1, 1)  # keep shape (1,1)
        # single-node adjacency
        adj1 = torch.FloatTensor(np.eye(1))

        cur_feature = torch.FloatTensor(last_scaled_feature)  # shape (1,1)

        for i in range(n_days):
            # forward pass for single node
            with torch.no_grad():
                pred_scaled = model(cur_feature, adj1).numpy().reshape(-1)  # shape (1,) scaled in label space

            # inverse scale to original price space (scaler is fitted to labels in this script)
            pred_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]

            future_date = stock_df.index[-1] + timedelta(days=(i + 1))
            future_docs.append({
                'index': future_date.strftime("%Y-%m-%d"),
                'close': float(pred_price),
                'ticker': ticker,
                'future': True,
                'MSE': mse
            })

            # For a simple iterative scheme we keep the feature constant (last observed scaled open).
            # If you prefer to change the feature each step (e.g. set next feature = predicted value),
            # you'll need a consistent feature-scaler (not overwritten) — ask me and I can modify that.
            # cur_feature remains the same here:
            cur_feature = torch.FloatTensor(last_scaled_feature)

        return future_docs

    # Generate next 7 calendar-day predictions and append them
    future_predictions = predict_future_gcn(model, scaler, X_test, n_days=20, ticker=ticker)
    pred_documents.extend(future_predictions)
    # --- End added future-prediction code ---

    return pred_documents

# Execution logic moved to main.py
if __name__ == "__main__":
    for collection_name in db.list_collection_names():
        # print(collection_name)
        # if collection_name != "HDFCBANK.NS":
        #     continue

        collection = db[collection_name]
        current_date = datetime.now()
        start_date = current_date - timedelta(days=5 * 365)
        stock_data = list(collection.find({'index': {'$gte': start_date}}))
        if len(stock_data) < 50:
            continue

        pred_docs = generate_predictions(stock_data, collection_name)
        pred_collection_name = f"{collection_name}_GCN_predicted"
        pred_collection = db[pred_collection_name]
        pred_collection.delete_many({})
        pred_collection.insert_many(pred_docs)

    print("Done processing all collections.")
