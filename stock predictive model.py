import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn. ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Load historical stock price data
ticker_symbol = 'HDFCBANK.NS'
today = datetime.today()
one_year_ago = today - timedelta(days=365)
start_date = "2013-08-23"
end_date = "2023-08-22"
data = yf.download(ticker_symbol, start=start_date, end=end_date)
# Fill NaN values with the first available price data.fillna(method='ffill', inplace=True)
data.fillna(method='ffill',inplace=True)

# Feature engineering (Adding moving averages as an example)

data['50d_MA'] = data['Close'].rolling(window=50).mean()

data['200d_MA'] = data['Close'].rolling(window=200).mean()

# Define features and target variable
features = ['Open', 'High', 'Low', 'Close', 'Volume', '50d_MA', '200d_MA']
target = 'Close'

# Create input (X) and output (y) data 
X= data[features].dropna()
y =X.pop(target)

# Create and train a Linear Regression model
X_train, X_test, y_train, y_test  =train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)

# Split data into training and testing sets


# Calculate total return
portfolio_value = 100000
total_return = (portfolio_value) / 100000

print(f"Total Return: {total_return:.2%}")

# Make predictions on the testing set
predictions = model.predict(X_test)

predicted_next_day_close = predictions [-1]

print("Predicted Next Day's Closing Price:s",{predicted_next_day_close}.2f)

# Backtesting
import numpy as np
from scipy.optimize import newton

# Backtesting with Moving Average Crossover Strategy
portfolio_value = 100000 # Starting portfolio value

cash_flows = [-portfolio_value] # Initial investment is an outflow

position = 0 # 0 for no position, 1 for long, -1 for short

short_window = 20

long_window = 50

short_ma = y_test.rolling(window=short_window).mean()
Long_ma = y_test.rolling (window=long_window).mean()


