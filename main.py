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
        ax.plot(real_prices, color='blue', label=f'{ticker} Actual Price')
        ax.plot(predicted_prices, color='red', label=f'{ticker} Predicted Price')

        # Add comparison stock data to the plot
        for ticker_comp in tickers_compare.split(','):
            ticker_comp = ticker_comp.strip()
            if ticker_comp in dfs:
                df_comp = dfs[ticker_comp][["Close"]]
                scaled_data_comp = scaler.fit_transform(df_comp)
                x_comp, y_comp = [], []
                for i in range(sequence_length, len(scaled_data_comp)):
                    x_comp.append(scaled_data_comp[i-sequence_length:i, 0])
                    y_comp.append(scaled_data_comp[i, 0])
                x_comp, y_comp = np.array(x_comp), np.array(y_comp)
                x_comp = np.reshape(x_comp, (x_comp.shape[0], x_comp.shape[1], 1))

                # Model prediction for comparison stock
                model.fit(x_comp, y_comp, batch_size=32, epochs=20, validation_data=(x_test, y_test))
                predicted_comp = model.predict(x_comp)
                predicted_prices_comp = scaler.inverse_transform(predicted_comp.reshape(-1, 1))
                ax.plot(predicted_prices_comp, label=f'{ticker_comp} Predicted Price')

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
