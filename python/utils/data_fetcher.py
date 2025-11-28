from datetime import datetime
import os
import pandas as pd
from pymongo import MongoClient
import yfinance as yf

# Connect to MongoDB (local)
client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['stocks']


def fetch_and_store_stocks(stock_names):
    """Fetch historical data for a list of stock symbols (no suffix) and store into MongoDB.

    Each symbol is fetched as SYMBOL.NS and stored in a collection named SYMBOL.NS.
    """
    today = datetime.today()

    for stock_name in stock_names:
        stock_name = str(stock_name).strip()
        if not stock_name:
            continue

        try:
            # Download historical data using yfinance
            hist = yf.download(f"{stock_name}.NS", start="2018-01-01", end=today.strftime('%Y-%m-%d'), progress=False)

            if hist is None or hist.empty:
                print(f"No historical data for {stock_name}, skipping.")
                continue

            # Sector info
            try:
                ticker = yf.Ticker(f"{stock_name}.NS")
                info = getattr(ticker, 'info', {}) or {}
                sector = info.get('sector') if isinstance(info, dict) else None
            except Exception:
                sector = None

            collection_name = f"{stock_name}.NS"
            hist = hist.reset_index()

            # Flatten MultiIndex if needed
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [col[0] if isinstance(col, tuple) else col for col in hist.columns]

            # Keep only required columns
            hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            hist.rename(columns={'Date': 'index', 'Volume': 'volume', 'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'}, inplace=True)

            # Add sector column
            hist['sector'] = sector

            # Ensure order of columns
            hist = hist[['index', 'open', 'high', 'low', 'close', 'volume', 'sector']]

            records = hist.to_dict(orient='records')

            if records:
                db[collection_name].insert_many(records)
                print(f"Data for {stock_name} stored successfully. ({len(records)} records)")

        except Exception as exc:
            print(f"Stock not Registered {stock_name}: {exc}")


if __name__ == "__main__":
    excel_path = os.path.join(os.path.dirname(__file__), "..", "MCAP31122023.xlsx")
    if not os.path.exists(excel_path):
        print(f"Excel file not found at {excel_path}. Please place MCAP31122023.xlsx in the python folder.")
        raise SystemExit(1)

    df = pd.read_excel(excel_path)
    stock_names = df['Symbol'].dropna().astype(str).tolist()[:10]
    fetch_and_store_stocks(stock_names)
