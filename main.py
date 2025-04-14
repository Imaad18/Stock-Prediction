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
tickers = [ticker] + [x.strip() for x in tickers_compare.split(',')]

# Download Stock Data for each ticker
dfs = {}
for t in tickers:
    try:
        df_temp = yf.download(t, start=start_date, end=end_date)
        if df_temp.empty:
            st.warning(f"⚠️ No data available for ticker {t}. Please check the ticker or date range.")
        else:
            dfs[t] = df_temp
    except Exception as e:
        st.error(f"❌ Error fetching data for {t}: {str(e)}")

# Check if any data was fetched successfully
if not dfs:
    st.error("❌ No valid stock data fetched. Please check your ticker symbols and date range.")
else:
    # Show Full Stock Data Table for the selected ticker
    st.subheader(f"📊 Stock Data for {ticker}")
    if ticker in dfs:
        df_all = dfs[ticker][["Open", "High", "Low", "Close", "Volume"]]
        st.dataframe(df_all.tail(100))
    else:
        st.warning(f"⚠️ No data available to display for {ticker}.")

    # Continue with prediction only if data is valid for the main ticker
    if ticker in dfs and len(dfs[ticker]) > 60:
        # Use only Close price for prediction
        df = dfs[ticker][["Close"]].copy()

        # Normalize Data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        # Create Sequences for LSTM
        sequence_length = 60
        x, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            x.append(scaled_data[i - sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        x, y = np.array(x), np.array(y)

        # Reshape x to be 3D for LSTM [samples, time steps, features]
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
        model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test), verbose=0)  # Reduced verbosity

        # Predictions
        predicted = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plotting the Prediction for the Selected Ticker
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df.index[split + sequence_length:], real_prices, color='blue', label=f'{ticker} Actual Price')
        ax.plot(df.index[split + sequence_length:], predicted_prices, color='red', label=f'{ticker} Predicted Price')

        # Add comparison stock data to the plot
        for ticker_comp in tickers_compare.split(','):
            ticker_comp = ticker_comp.strip()
            if ticker_comp in dfs and len(dfs[ticker_comp]) > 0:
                df_comp = dfs[ticker_comp][["Close"]]
                ax.plot(df_comp.index, df_comp["Close"], label=f'{ticker_comp} Actual Price', alpha=0.6)

        ax.set_title(f'{ticker} Stock Price Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')
        ax.legend()
        st.pyplot(fig)

        # Evaluate Model
        rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
        st.write(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Sidebar: Latest Price Info
        if ticker in dfs and "Close" in dfs[ticker].columns:
            latest_price = dfs[ticker]["Close"].iloc[-1]
            latest_date = dfs[ticker].index[-1]
            try:
                latest_price_float = float(latest_price)
                st.sidebar.subheader(f"💰 Latest Update: {ticker}")
                st.sidebar.write(f"Current Price: ${latest_price_float:.2f}")
                st.sidebar.write(f"Date: {latest_date.strftime('%Y-%m-%d')}")
            except ValueError:
                st.sidebar.warning("Error displaying latest price.")
        else:
            st.sidebar.warning("No price data found for the latest update.")

    else:
        st.warning("📌 Not enough data to run prediction for the selected ticker. Please select a larger date range.")

# Sidebar: News Feed
st.sidebar.subheader("📢 Related News")
news_query = ticker if ticker else "Stock Market"
rss_url = f"https://news.google.com/rss/search?q={news_query}&hl=en-US&gl=US&ceid=US:en"

try:
    feed = feedparser.parse(rss_url)
    max_articles = 5
    if feed.entries:
        st.sidebar.write(f"Latest news related to: **{news_query}**")
        for entry in feed.entries[:max_articles]:
            st.sidebar.markdown(f"- [{entry.title}]({entry.link})")
    else:
        st.sidebar.info("No news found for the current query.")
except Exception as e:
    st.sidebar.error(f"Error fetching news: {str(e)}")
