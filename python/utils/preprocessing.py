import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def clean_stock_data(stock_df):
    """
    Clean stock data by removing NaN values and ensuring proper data types.
    
    Args:
        stock_df: DataFrame with stock data
        
    Returns:
        Cleaned DataFrame
    """
    stock_df = stock_df.copy()
    stock_df.dropna(inplace=True)
    return stock_df


def normalize_data(data, feature_range=(0, 1)):
    """
    Normalize data using MinMaxScaler.
    
    Args:
        data: numpy array or DataFrame column to normalize
        feature_range: tuple specifying the range for normalization
        
    Returns:
        scaler: fitted MinMaxScaler object
        normalized_data: normalized data
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    normalized_data = scaler.fit_transform(data)
    return scaler, normalized_data


def prepare_stock_dataframe(stock_data, columns):
    """
    Convert MongoDB documents to DataFrame and prepare for processing.
    
    Args:
        stock_data: list of MongoDB documents
        columns: list of column names to keep
        
    Returns:
        Prepared DataFrame with datetime index
    """
    stock_df = pd.DataFrame(stock_data)
    stock_df = stock_df[columns]
    stock_df['index'] = pd.to_datetime(stock_df['index'])
    stock_df.set_index('index', inplace=True)
    stock_df.dropna(inplace=True)
    return stock_df


def resample_and_fill(df, freq='D', method='linear'):
    """
    Resample time series data and fill missing values.
    
    Args:
        df: DataFrame with datetime index
        freq: resampling frequency (e.g., 'D' for daily)
        method: interpolation method
        
    Returns:
        Resampled and filled DataFrame
    """
    df_resampled = df.resample(freq).mean()
    df_filled = df_resampled.interpolate(method=method)
    return df_filled
