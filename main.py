# Import Required Libraries
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
import feedparser
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error

# Streamlit App Title
st.title("📈 Stock Price Prediction with LSTM")

# Sidebar: Inputs
st.sidebar.header("Stock Input")
ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")
tickers_compare = st.sidebar.text_input("Compare with another stock (comma separated):", "GOOG")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Combine tickers for main and comparison stocks
tickers = [ticker] + [x.strip() for x in tickers_compare.split(',')]  # Add comparison tickers

# Download Stock Data for each ticker
dfs = {}
for ticker in tickers:
    try:
        df_temp = yf.download(ticker, start=start_date, end=end_date)
        if df_temp.empty:
            st.write(f"⚠️ No data available for ticker {ticker}. Please check the ticker or date range.")
        else:
            dfs[ticker] = df_temp
    except Exception as e:
        st.write(f"❌ Error fetching data for {ticker}: {str(e)}")

# Check if any data was fetched successfully
if not dfs:
    st.write("❌ No valid stock data fetched. Please check your ticker symbols and date range.")
else:
    # Show Full Stock Data Table for the selected ticker
    st.subheader(f"📊 Stock Data for {ticker}")
    df_all = dfs[ticker][["Open", "High", "Low", "Close", "Volume"]]
    st.dataframe(df_all.tail(100))  # Show last 100 rows

    # Continue only if data is valid
    if len(dfs[ticker]) > 60:

        # Use only Close price for prediction
        df = dfs[ticker][["Close"]]

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
        model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))

        # Predictions
        predicted = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plotting the Prediction for the Selected Ticker
        fig, ax = plt.subplots(figsize=(14, 6))
        ax
