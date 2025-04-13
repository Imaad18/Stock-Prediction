import yfinance as yf    # To fetch Stock Data
import numpy as np       # For Data Manipulation
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
import streamlit as st
import time

# Streamlit app layout and title
st.title("Stock Price Prediction with LSTM")
st.sidebar.header("Stock Prediction Parameters")

# User input for stock ticker
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, TSLA):", "AAPL")
st.sidebar.write("Stock Ticker Selected:", ticker)

# Fetch Data
@st.cache
def fetch_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2024-12-31")
    df = df[["Close"]]
    return df

df = fetch_data(ticker)

# Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Create Sequence for LSTM
sequence_length = 60
x, y = [], []

for i in range(sequence_length, len(scaled_data)):
    x.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Train-test split Data
split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

# Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")

# Training the model
st.text("Training the LSTM model...")
history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test), verbose=1)

# Real-Time Data Update - Live Stock Predictions
def get_live_predictions():
    # Fetch the latest stock data
    df_live = yf.download(ticker, start="2024-01-01", end="2025-04-13")
    df_live = df_live[["Close"]]
    
    # Preprocess the data
    scaled_data_live = scaler.transform(df_live)
    x_live, y_live = [], []
    for i in range(sequence_length, len(scaled_data_live)):
        x_live.append(scaled_data_live[i-sequence_length:i, 0])
        y_live.append(scaled_data_live[i, 0])
    
    x_live = np.array(x_live)
    x_live = np.reshape(x_live, (x_live.shape[0], x_live.shape[1], 1))
    
    # Predict stock prices
    predicted = model.predict(x_live)
    predicted_prices = scaler.inverse_transform(predicted.reshape(-1,1))
    
    return predicted_prices, df_live

# Display the live update section
st.text("Fetching live updates...")
live_predicted_prices, live_df = get_live_predictions()

# Plotting the live stock price predictions
plt.figure(figsize=(14, 6))
plt.plot(live_df.index[-100:], live_df['Close'][-100:], color='blue', label='Actual Price')
plt.plot(live_df.index[-len(live_predicted_prices):], live_predicted_prices, color='red', label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction - Live Updates')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(plt)

# RMSE Evaluation on Live Data
real_prices_live = scaler.inverse_transform(live_df["Close"].values[-len(live_predicted_prices):].reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(real_prices_live, live_predicted_prices))
st.text(f"Root Mean Squared Error on Live Data: {rmse:.2f}")

# Setting a timer for live updates every 5 minutes (can be adjusted)
while True:
    time.sleep(300)  # 5 minutes
    live_predicted_prices, live_df = get_live_predictions()
    plt.figure(figsize=(14, 6))
    plt.plot(live_df.index[-100:], live_df['Close'][-100:], color='blue', label='Actual Price')
    plt.plot(live_df.index[-len(live_predicted_prices):], live_predicted_prices, color='red', label='Predicted Price')
    plt.title(f'{ticker} Stock Price Prediction - Live Updates')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    st.pyplot(plt)
