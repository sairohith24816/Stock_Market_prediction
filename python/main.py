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
from models import lstm, arima, gcn
from utils.data_fetcher import fetch_and_store_stocks
import pandas as pd
import os


def run_lstm_predictions():
    """Run LSTM model predictions on all stock collections."""
    print("Running LSTM predictions...")
    db = get_db_connection()
    
    for collection_name in db.list_collection_names():
        if collection_name.endswith('_predicted'):
            continue
            
        collection = db[collection_name]
        current_date = datetime.now()
        start_date = current_date - timedelta(days=5 * 365)
        query = {'index': {'$gte': start_date}}
        stock_data = list(collection.find(query))
        
        if len(stock_data) < 50:
            continue
            
        pred_docs = lstm.generate_predictions(stock_data, collection_name)
        pred_collection_name = f"{collection_name}_LSTM_predicted"
        pred_collection = db[pred_collection_name]
        pred_collection.delete_many({})
        pred_collection.insert_many(pred_docs)
    
    print("LSTM predictions completed.")


def run_arima_predictions():
    """Run ARIMA model predictions on all stock collections."""
    print("Running ARIMA predictions...")
    db = get_db_connection()
    
    for collection_name in db.list_collection_names():
        if collection_name.endswith('_predicted'):
            continue
            
        collection = db[collection_name]
        current_date = datetime.now()
        start_date = current_date - timedelta(days=5 * 365)
        query = {'index': {'$gte': start_date}}
        stock_data = list(collection.find(query))
        
        if len(stock_data) < 50:
            continue
            
        pred_docs = arima.generate_predictions(stock_data, collection_name)
        pred_collection_name = f"{collection_name}_ARIMA_predicted"
        pred_collection = db[pred_collection_name]
        pred_collection.delete_many({})
        pred_collection.insert_many(pred_docs)
    
    print("ARIMA predictions completed.")


def run_gcn_predictions():
    """Run GCN model predictions on all stock collections."""
    print("Running GCN predictions...")
    db = get_db_connection()
    
    for collection_name in db.list_collection_names():
        if collection_name.endswith('_predicted'):
            continue
            
        collection = db[collection_name]
        current_date = datetime.now()
        start_date = current_date - timedelta(days=5 * 365)
        stock_data = list(collection.find({'index': {'$gte': start_date}}))
        
        if len(stock_data) < 50:
            continue
            
        pred_docs = gcn.generate_predictions(stock_data, collection_name)
        pred_collection_name = f"{collection_name}_GCN_predicted"
        pred_collection = db[pred_collection_name]
        pred_collection.delete_many({})
        pred_collection.insert_many(pred_docs)
    
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
    stock_names = df['Symbol'].dropna().astype(str).tolist()[:10]
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
