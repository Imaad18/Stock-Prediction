

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf

# Web UI
st.title("📈 Stock Price Predictor using LSTM")
ticker = st.text_input("Enter Stock Ticker", value="AAPL")

# Load Data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2024-12-31")
    return df[["Close"]]

df = load_data(ticker)
st.line_chart(df)

# Preprocess
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

sequence_length = 60
x, y = [], []
for i in range(sequence_length, len(scaled_data)):
    x.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

# Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
with st.spinner("Training model..."):
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=0)

# Predict
predicted = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot Results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(real_prices, label="Actual Price", color='blue')
ax.plot(predicted_prices, label="Predicted Price", color='red')
ax.set_title(f"{ticker} Stock Price Prediction")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# RMSE
rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
st.success(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
