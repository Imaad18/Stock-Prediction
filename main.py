# Import Required Libraries
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error

# Streamlit: Set Title
st.title("Stock Price Prediction with LSTM")

# Sidebar: Stock Ticker and Date Range Input
st.sidebar.header("Stock Input")
ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Fetch Stock Data
df = yf.download(ticker, start=start_date, end=end_date)
df = df[["Close"]]

# Display DataFrame on Streamlit
st.write(f"Stock Data for {ticker}")
st.dataframe(df.tail())

# Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Create Sequences for LSTM
sequence_length = 60
x, y = [], []
for i in range(sequence_length, len(scaled_data)):
    x.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Train-Test Split
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

# Train Model
history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))

# Predictions
predicted = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted.reshape(-1,1))
real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# Plotting the Graph
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(real_prices, color='blue', label='Actual Price')
ax.plot(predicted_prices, color='red', label='Predicted Price')
ax.set_title(f'{ticker} Stock Price Prediction')
ax.set_xlabel('Time')
ax.set_ylabel('Stock Price')
ax.legend()
st.pyplot(fig)

# Evaluate Performance
rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Sidebar: Current Stock Update
if not df.empty:
    latest_price = df["Close"].iloc[-1]
    latest_date = df.index[-1]
    st.sidebar.subheader(f"Latest Update: {ticker}")
    st.sidebar.write(f"Current Price: ${latest_price:.2f}")
    st.sidebar.write(f"Date: {latest_date.strftime('%Y-%m-%d')}")
else:
    st.sidebar.write("No data available for the selected stock and date range.")
