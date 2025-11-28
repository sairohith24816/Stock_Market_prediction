# Stock Market Prediction

## Project Setup and Usage

### Prerequisites
- Python 3.8+
- Node.js and npm
- MongoDB (running on localhost:27017)

### Installation

1.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install Client Dependencies:**
    ```bash
    cd client
    npm install
    cd ..
    ```

3.  **Install Server Dependencies:**
    ```bash
    cd server
    npm install
    cd ..
    ```

### Running the Application

#### 1. Data Fetching and Model Training (Python)

Use the `main.py` script to fetch data and run prediction models.

*   **Fetch Stock Data:**
    ```bash
    python main.py --fetch-data
    ```
    This will read from `python/MCAP31122023.xlsx` and populate the MongoDB database.

*   **Run Prediction Models:**
    ```bash
    python main.py --model lstm      # Run LSTM predictions
    python main.py --model arima     # Run ARIMA predictions
    python main.py --model gcn       # Run GCN predictions
    python main.py --model all       # Run all models
    ```

#### 2. Start the Backend Server
```bash
cd server
npm start
```
The server runs on `http://localhost:8080`.

#### 3. Start the Frontend Client
```bash
cd client
npm start
```
The client runs on `http://localhost:3000`.

## Dependencies:

openpyxl
pandas
pymongo
datetime
Flask 
plotly


Reference for API:  https://algotrading101.com/learn/yahoo-finance-api-guide/


yfinance (library to use api call to fetch yhe stocks   API: Yahoo Finance)
$ pip install yfinance --upgrade --no-cache-dir


ANother library for api: using this
yahoo_fin
pip install yahoo_fin

Dependencies for yahoo_fin  pre-installed or may need to:
ftplib
io
pandas
requests
requests_html


//For historical data
from yahoo_fin.stock_info import get_data
get_data(ticker, start_date = None, end_date = None, index_as_date = True, interval = “1d”)

dataframe of each ticker/stock got in rsopnse 
date      open         high          low        close     adjclose   volume       ticker


npm install react react-dom react-router-dom lightweight-charts


do delete the database command
db.getCollectionNames().forEach(function (collectionName) { if (collectionName.endsWith("predicted")) { db[collectionName].drop(); } });