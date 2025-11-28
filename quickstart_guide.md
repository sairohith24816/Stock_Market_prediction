# Quickstart Guide

Follow these steps to set up and run the Stock Market Prediction project.

## Prerequisites

*   **MongoDB**: Ensure MongoDB is installed and running locally on port `27017`.
*   **Node.js**: Ensure Node.js and npm are installed.
*   **Python**: Ensure Python (3.8+) is installed.

## 1. Setup Python Backend (Data & Predictions)

This step populates your database with stock data and runs prediction models.

1.  Open a terminal in the root directory.
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the main script:
    ```bash
    python python/main.py
    ```
    > **Note:** By default, this runs `RunPredictions()`. If you need to fetch initial stock data first, open `python/main.py` and uncomment `StoringData()` at the bottom of the file before running it.

## 2. Start the API Server

This serves the data from MongoDB to the frontend.

1.  Open a **new** terminal.
2.  Navigate to the server directory:
    ```bash
    cd server
    ```
3.  Install dependencies:
    ```bash
    npm install
    ```
4.  Start the server:
    ```bash
    npm start
    ```
    The server will start on `http://localhost:8080`.

## 3. Start the Client (Frontend)

This launches the React application.

1.  Open a **new** terminal.
2.  Navigate to the client directory:
    ```bash
    cd client
    ```
3.  Install dependencies:
    ```bash
    npm install
    ```
4.  Start the application:
    ```bash
    npm start
    ```
    The application will open in your browser at `http://localhost:3000`.
