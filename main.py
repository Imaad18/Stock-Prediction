import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
import concurrent.futures
import time
from datetime import datetime, timedelta
import requests

# Enable caching for expensive operations
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return df[["Close"]] if not df.empty else None
    except:
        return None

@st.cache_resource(show_spinner=False)
def create_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Fetch news for a stock
def fetch_news(ticker):
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json()
            news_links = []
            for item in results.get("news", [])[:5]:  # Top 5 articles
                title = item.get("title", "Untitled")
                link = item.get("link", "#")
                news_links.append((title, link))
            return news_links
    except Exception:
        return []
    return []

# Streamlit App
st.title("⚡ Optimized Stock Price Prediction")

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    ticker = st.text_input("Main Stock Ticker", "AAPL").upper()
    compare_tickers = st.text_input("Compare with (comma separated)", "MSFT,GOOG").upper()
    start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
    run_prediction = st.button("Run Prediction", type="primary")

# Prepare tickers
all_tickers = [ticker] + [t.strip() for t in compare_tickers.split(",") if t.strip()]

# Fetch data in parallel
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(load_data, t, start_date, end_date): t for t in all_tickers}
    data = {}
    for future in concurrent.futures.as_completed(futures):
        ticker_result = futures[future]
        data[ticker_result] = future.result()

# Validate fetched data
valid_tickers = {k: v for k, v in data.items() if v is not None}
invalid_tickers = set(all_tickers) - set(valid_tickers.keys())

if invalid_tickers:
    st.warning(f"Could not fetch data for: {', '.join(invalid_tickers)}")

if not valid_tickers:
    st.error("No valid stock data available. Please check your inputs.")
    st.stop()

# Show main stock data
if run_prediction:
    main_df = valid_tickers.get(ticker)
    if main_df is not None:
        with st.expander(f"📊 {ticker} Stock Data (Last 20 Days)"):
            st.dataframe(main_df.tail(20))

# Run prediction
if run_prediction and ticker in valid_tickers:
    df = valid_tickers[ticker]
    if len(df) < 100:
        st.warning("Not enough data for accurate prediction. Please select a longer time period.")
        st.stop()
    
    progress_bar = st.progress(0)
    
    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    progress_bar.progress(20)
    
    # Sequence creation
    seq_length = 60
    x, y = [], []
    for i in range(seq_length, len(scaled_data)):
        x.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    x, y = np.array(x), np.array(y)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    progress_bar.progress(40)
    
    # Train-test split
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Model creation and training
    model = create_model((x_train.shape[1], 1))
    history = model.fit(x_train, y_train, 
                       batch_size=32, 
                       epochs=15, 
                       validation_data=(x_test, y_test),
                       verbose=0)
    progress_bar.progress(80)
    
    # Predictions
    predictions = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    progress_bar.progress(95)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    
    # Plot prediction
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index[split+seq_length:], actual_prices, 'b-', label='Actual')
    ax.plot(df.index[split+seq_length:], predicted_prices, 'r--', label='Predicted')
    ax.set_title(f"{ticker} Price Prediction (RMSE: {rmse:.2f})")
    ax.legend()
    progress_bar.progress(100)
    st.pyplot(fig)
    
    # Comparison chart
    if len(valid_tickers) > 1:
        st.subheader("Comparison with Other Stocks")
        fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
        for t, df_t in valid_tickers.items():
            ax_comp.plot(df_t.index, df_t['Close'], label=t)
        ax_comp.set_title("Stock Price Comparison")
        ax_comp.legend()
        st.pyplot(fig_comp)

    # Closing prices last 20 days
    st.subheader("📉 Last 20-Day Closing Prices")
    for t, df_t in valid_tickers.items():
        if df_t is not None and not df_t.empty:
            last_20 = df_t.tail(20).copy()
            last_20.index = last_20.index.strftime('%Y-%m-%d')
            st.markdown(f"**{t}**")
            st.dataframe(last_20[['Close']].rename(columns={'Close': 'Closing Price'}))

# Sidebar - Latest Prices
with st.sidebar:
    st.header("Latest Prices")
    for t, df_t in valid_tickers.items():
        if not df_t.empty:
            try:
                last_price = df_t['Close'].iloc[-1]
                if hasattr(last_price, 'values'):
                    last_price = last_price.values[0]
                st.metric(label=t, value=f"${float(last_price):.2f}")
            except Exception as e:
                st.error(f"Error displaying price for {t}: {str(e)}")

    st.info(f"Data loaded in {time.time()-start_time:.2f} seconds")

    # Sidebar - News Links
    st.header("🗞️ News (Top 5)")
    for t in valid_tickers:
        st.markdown(f"**{t}**")
        news = fetch_news(t)
        if news:
            for title, link in news:
                st.markdown(f"- [{title}]({link})")
        else:
            st.write("No news found.")
