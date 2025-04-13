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
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Fetch Stock Data
df = yf.download(ticker, start=start_date, end=end_date)
df_all = df.copy()  # Keep full data for table

# Show Full Stock Data Table
st.subheader(f"📊 Stock Data for {ticker}")
if not df_all.empty:
    df_all = df_all[["Open", "High", "Low", "Close", "Volume"]]
    st.dataframe(df_all.tail(100))  # Show last 100 rows
else:
    st.write("No stock data found for the given input.")

# Continue only if data is valid
if not df.empty and len(df) > 60:

    # Use only Close price for prediction
    df = df[["Close"]]

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

    # Plotting the Prediction
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(real_prices, color='blue', label='Actual Price')
    ax.plot(predicted_prices, color='red', label='Predicted Price')
    ax.set_title(f'{ticker} Stock Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.legend()
    st.pyplot(fig)

    # Evaluate Model
    rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
    st.write(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Sidebar: Latest Price Info
    if "Close" in df.columns:
        latest_price = df["Close"].iloc[-1]
        latest_date = df.index[-1]
        try:
            latest_price_float = float(latest_price)
            st.sidebar.subheader(f"💰 Latest Update: {ticker}")
            st.sidebar.write(f"Current Price: ${latest_price_float:.2f}")
            st.sidebar.write(f"Date: {latest_date.strftime('%Y-%m-%d')}")
        except:
            st.sidebar.write("Error displaying latest price.")
    else:
        st.sidebar.write("No price data found.")

else:
    st.write("📌 Not enough data to run prediction. Please select a larger date range.")

# Sidebar: News Feed
st.sidebar.subheader("📢 Related News")
news_query = ticker if ticker else "Stock Market"
rss_url = f"https://news.google.com/rss/search?q={news_query}&hl=en-US&gl=US&ceid=US:en"

try:
    feed = feedparser.parse(rss_url)
    max_articles = 5
    if feed.entries:
        for entry in feed.entries[:max_articles]:
            st.sidebar.markdown(f"**[{entry.title}]({entry.link})**")
    else:
        st.sidebar.write("No news found.")
except Exception as e:
    st.sidebar.write("Error fetching news.")
