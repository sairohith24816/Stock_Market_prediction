"""
Main entry point for ML model predictions on stock data.

This script provides functionality to:
1. Run LSTM predictions
2. Run ARIMA predictions  
3. Run GCN predictions
4. Fetch and store stock data

Usage:
    python main.py --model lstm
    python main.py --model arima
    python main.py --model gcn
    python main.py --fetch-data
"""

import argparse
import sys
from datetime import datetime, timedelta
from config.db_config import get_db_connection
from config.loader import config
from models import lstm, arima, gcn
from utils.data_fetcher import fetch_and_store_stocks
import pandas as pd
import os
import wandb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def run_lstm_predictions():
    """Run LSTM model predictions on all stock collections and log metrics to Weights & Biases."""
    print("Running LSTM predictions...")
    db = get_db_connection()

    # Initialise a single W&B run for this model over all stocks
    wandb_cfg = config.get('wandb', {})
    project = wandb_cfg.get('project', 'stock-market-prediction')
    entity = wandb_cfg.get('entity')

    wandb.init(
        project=project,
        entity=entity,
        name=f"lstm_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            'model': 'lstm',
            'prediction_history_years': config['data_fetching']['prediction_history_years'],
            'min_data_points': config['data_fetching']['min_data_points'],
        },
    )

    metrics_table = wandb.Table(columns=[
        "ticker",
        "collection",
        "mse_test",
        "mae_test",
        "r2_test",
        "mse_train",
        "n_points",
        "n_train",
        "n_test",
    ])

    for collection_name in db.list_collection_names():
        if collection_name.endswith('_predicted'):
            continue

        collection = db[collection_name]
        current_date = datetime.now()
        history_years = config['data_fetching']['prediction_history_years']
        start_date = current_date - timedelta(days=history_years * 365)
        query = {'index': {'$gte': start_date}}
        stock_data = list(collection.find(query))

        if len(stock_data) < config['data_fetching']['min_data_points']:
            continue

        pred_docs = lstm.generate_predictions(stock_data, collection_name)
        pred_collection_name = f"{collection_name}_LSTM_predicted"
        pred_collection = db[pred_collection_name]
        pred_collection.delete_many({})
        pred_collection.insert_many(pred_docs)

        if not pred_docs:
            continue

        # Derive train/test splits based on flags
        train_docs = [d for d in pred_docs if d.get('type') == 'train' and not d.get('future')]
        test_docs = [d for d in pred_docs if not d.get('future') and d.get('type') != 'train']

        # Rebuild actual/pred arrays for metrics where possible
        mse_train = None
        mse_test = None
        mae_test = None
        r2_test = None

        # Map index->actual close from original data
        df_orig = pd.DataFrame(stock_data)
        if 'index' in df_orig and 'close' in df_orig:
            df_orig['index'] = pd.to_datetime(df_orig['index']).dt.strftime("%Y-%m-%d")
            actual_map = df_orig.set_index('index')['close'].to_dict()

            if train_docs:
                y_train_true = []
                y_train_pred = []
                for d in train_docs:
                    idx = d.get('index')
                    if idx in actual_map:
                        y_train_true.append(float(actual_map[idx]))
                        y_train_pred.append(float(d.get('close', 0.0)))
                if y_train_true:
                    mse_train = mean_squared_error(y_train_true, y_train_pred)

            if test_docs:
                y_test_true = []
                y_test_pred = []
                for d in test_docs:
                    idx = d.get('index')
                    if idx in actual_map:
                        y_test_true.append(float(actual_map[idx]))
                        y_test_pred.append(float(d.get('close', 0.0)))
                if y_test_true:
                    mse_test = mean_squared_error(y_test_true, y_test_pred)
                    mae_test = mean_absolute_error(y_test_true, y_test_pred)
                    r2_test = r2_score(y_test_true, y_test_pred)

        # Fallback to single stored MSE when detailed reconstruction not possible
        sample_doc = pred_docs[-1]
        if mse_test is None:
            mse_test = float(sample_doc.get('MSE', 0.0))

        ticker = sample_doc.get('ticker', collection_name)
        n_points = len(pred_docs)
        n_train = len(train_docs)
        n_test = len(test_docs)

        metrics_table.add_data(
            ticker,
            collection_name,
            float(mse_test) if mse_test is not None else None,
            float(mae_test) if mae_test is not None else None,
            float(r2_test) if r2_test is not None else None,
            float(mse_train) if mse_train is not None else None,
            n_points,
            n_train,
            n_test,
        )

    # Log the aggregated table once per run
    wandb.log({"lstm_stock_metrics": metrics_table})
    wandb.finish()

    print("LSTM predictions completed.")


def run_arima_predictions():
    """Run ARIMA model predictions on all stock collections and log metrics to Weights & Biases."""
    print("Running ARIMA predictions...")
    db = get_db_connection()

    wandb_cfg = config.get('wandb', {})
    project = wandb_cfg.get('project', 'stock-market-prediction')
    entity = wandb_cfg.get('entity')

    wandb.init(
        project=project,
        entity=entity,
        name=f"arima_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            'model': 'arima',
            'prediction_history_years': config['data_fetching']['prediction_history_years'],
            'min_data_points': config['data_fetching']['min_data_points'],
        },
    )

    metrics_table = wandb.Table(columns=[
        "ticker",
        "collection",
        "mse_test",
        "mae_test",
        "r2_test",
        "mse_train",
        "n_points",
        "n_train",
        "n_test",
    ])

    for collection_name in db.list_collection_names():
        if collection_name.endswith('_predicted'):
            continue

        collection = db[collection_name]
        current_date = datetime.now()
        history_years = config['data_fetching']['prediction_history_years']
        start_date = current_date - timedelta(days=history_years * 365)
        query = {'index': {'$gte': start_date}}
        stock_data = list(collection.find(query))

        if len(stock_data) < config['data_fetching']['min_data_points']:
            continue

        pred_docs = arima.generate_predictions(stock_data, collection_name)
        pred_collection_name = f"{collection_name}_ARIMA_predicted"
        pred_collection = db[pred_collection_name]
        pred_collection.delete_many({})
        pred_collection.insert_many(pred_docs)

        if not pred_docs:
            continue

        train_docs = [d for d in pred_docs if d.get('type') == 'train' and not d.get('future')]
        test_docs = [d for d in pred_docs if not d.get('future') and d.get('type') != 'train']

        mse_train = None
        mse_test = None
        mae_test = None
        r2_test = None

        df_orig = pd.DataFrame(stock_data)
        if 'index' in df_orig and 'close' in df_orig:
            df_orig['index'] = pd.to_datetime(df_orig['index']).dt.strftime("%Y-%m-%d")
            actual_map = df_orig.set_index('index')['close'].to_dict()

            if train_docs:
                y_train_true = []
                y_train_pred = []
                for d in train_docs:
                    idx = d.get('index')
                    if idx in actual_map:
                        y_train_true.append(float(actual_map[idx]))
                        y_train_pred.append(float(d.get('close', 0.0)))
                if y_train_true:
                    mse_train = mean_squared_error(y_train_true, y_train_pred)

            if test_docs:
                y_test_true = []
                y_test_pred = []
                for d in test_docs:
                    idx = d.get('index')
                    if idx in actual_map:
                        y_test_true.append(float(actual_map[idx]))
                        y_test_pred.append(float(d.get('close', 0.0)))
                if y_test_true:
                    mse_test = mean_squared_error(y_test_true, y_test_pred)
                    mae_test = mean_absolute_error(y_test_true, y_test_pred)
                    r2_test = r2_score(y_test_true, y_test_pred)

        sample_doc = pred_docs[-1]
        if mse_test is None:
            mse_test = float(sample_doc.get('MSE', 0.0))

        ticker = sample_doc.get('ticker', collection_name)
        n_points = len(pred_docs)
        n_train = len(train_docs)
        n_test = len(test_docs)

        metrics_table.add_data(
            ticker,
            collection_name,
            float(mse_test) if mse_test is not None else None,
            float(mae_test) if mae_test is not None else None,
            float(r2_test) if r2_test is not None else None,
            float(mse_train) if mse_train is not None else None,
            n_points,
            n_train,
            n_test,
        )

    wandb.log({"arima_stock_metrics": metrics_table})
    wandb.finish()

    print("ARIMA predictions completed.")


def run_gcn_predictions():
    """Run GCN model predictions on all stock collections and log metrics to Weights & Biases."""
    print("Running GCN predictions...")
    db = get_db_connection()

    wandb_cfg = config.get('wandb', {})
    project = wandb_cfg.get('project', 'stock-market-prediction')
    entity = wandb_cfg.get('entity')

    wandb.init(
        project=project,
        entity=entity,
        name=f"gcn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            'model': 'gcn',
            'prediction_history_years': config['data_fetching']['prediction_history_years'],
            'min_data_points': config['data_fetching']['min_data_points'],
        },
    )

    metrics_table = wandb.Table(columns=[
        "ticker",
        "collection",
        "mse_test",
        "mae_test",
        "r2_test",
        "mse_train",
        "n_points",
        "n_train",
        "n_test",
    ])

    for collection_name in db.list_collection_names():
        if collection_name.endswith('_predicted'):
            continue

        collection = db[collection_name]
        current_date = datetime.now()
        history_years = config['data_fetching']['prediction_history_years']
        start_date = current_date - timedelta(days=history_years * 365)
        stock_data = list(collection.find({'index': {'$gte': start_date}}))

        if len(stock_data) < config['data_fetching']['min_data_points']:
            continue

        pred_docs = gcn.generate_predictions(stock_data, collection_name)
        pred_collection_name = f"{collection_name}_GCN_predicted"
        pred_collection = db[pred_collection_name]
        pred_collection.delete_many({})
        pred_collection.insert_many(pred_docs)

        if not pred_docs:
            continue

        train_docs = [d for d in pred_docs if d.get('type') == 'train' and not d.get('future')]
        test_docs = [d for d in pred_docs if not d.get('future') and d.get('type') != 'train']

        mse_train = None
        mse_test = None
        mae_test = None
        r2_test = None

        df_orig = pd.DataFrame(stock_data)
        if 'index' in df_orig and 'close' in df_orig:
            df_orig['index'] = pd.to_datetime(df_orig['index']).dt.strftime("%Y-%m-%d")
            actual_map = df_orig.set_index('index')['close'].to_dict()

            if train_docs:
                y_train_true = []
                y_train_pred = []
                for d in train_docs:
                    idx = d.get('index')
                    if idx in actual_map:
                        y_train_true.append(float(actual_map[idx]))
                        y_train_pred.append(float(d.get('close', 0.0)))
                if y_train_true:
                    mse_train = mean_squared_error(y_train_true, y_train_pred)

            if test_docs:
                y_test_true = []
                y_test_pred = []
                for d in test_docs:
                    idx = d.get('index')
                    if idx in actual_map:
                        y_test_true.append(float(actual_map[idx]))
                        y_test_pred.append(float(d.get('close', 0.0)))
                if y_test_true:
                    mse_test = mean_squared_error(y_test_true, y_test_pred)
                    mae_test = mean_absolute_error(y_test_true, y_test_pred)
                    r2_test = r2_score(y_test_true, y_test_pred)

        sample_doc = pred_docs[-1]
        if mse_test is None:
            mse_test = float(sample_doc.get('MSE', 0.0))

        ticker = sample_doc.get('ticker', collection_name)
        n_points = len(pred_docs)
        n_train = len(train_docs)
        n_test = len(test_docs)

        metrics_table.add_data(
            ticker,
            collection_name,
            float(mse_test) if mse_test is not None else None,
            float(mae_test) if mae_test is not None else None,
            float(r2_test) if r2_test is not None else None,
            float(mse_train) if mse_train is not None else None,
            n_points,
            n_train,
            n_test,
        )

    wandb.log({"gcn_stock_metrics": metrics_table})
    wandb.finish()

    print("GCN predictions completed.")


def fetch_stock_data():
    """Fetch and store stock data from Excel file."""
    print("Fetching stock data...")
    excel_path = os.path.join(os.path.dirname(__file__), "MCAP31122023.xlsx")
    
    if not os.path.exists(excel_path):
        print(f"Excel file not found at {excel_path}.")
        print("Please place MCAP31122023.xlsx in the python folder.")
        return
    
    df = pd.read_excel(excel_path)
    max_stocks = config['data_fetching']['max_stocks_to_fetch']
    stock_names = df['Symbol'].dropna().astype(str).tolist()[:max_stocks]
    fetch_and_store_stocks(stock_names)
    print("Stock data fetching completed.")


def main():
    parser = argparse.ArgumentParser(
        description='Run stock market prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model lstm      # Run LSTM predictions
  python main.py --model arima     # Run ARIMA predictions
  python main.py --model gcn       # Run GCN predictions
  python main.py --model all       # Run all models
  python main.py --fetch-data      # Fetch and store stock data
        """
    )
    
    parser.add_argument(
        '--model',
        choices=['lstm', 'arima', 'gcn', 'all'],
        help='Model to run for predictions'
    )
    
    parser.add_argument(
        '--fetch-data',
        action='store_true',
        help='Fetch and store stock data from Excel file'
    )
    
    args = parser.parse_args()
    
    if not args.model and not args.fetch_data:
        parser.print_help()
        sys.exit(1)
    
    if args.fetch_data:
        fetch_stock_data()
    
    if args.model == 'lstm':
        run_lstm_predictions()
    elif args.model == 'arima':
        run_arima_predictions()
    elif args.model == 'gcn':
        run_gcn_predictions()
    elif args.model == 'all':
        run_lstm_predictions()
        run_arima_predictions()
        run_gcn_predictions()


if __name__ == "__main__":
    main()
