import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_market_data(tickers=None, period="1mo", interval="1d"):
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "NVDA", "NFLX", "ADBE", "INTC"]
    
    # Load market data using yfinance
    data = yf.download(tickers, period=period, interval=interval)['Close']
    
    # Combine closing prices into a single dataset
    combined_data = {}
    for ticker in tickers:
        combined_data[ticker] = data[ticker].values
    
    return combined_data

def create_features(prices, window=5):
    X, y = [], []
    for i in range(len(prices) - window):
        X.append(prices[i:i + window])
        y.append(prices[i + window])
    return np.array(X), np.array(y)