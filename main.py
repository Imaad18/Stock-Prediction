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
import requests
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from matplotlib.animation import FuncAnimation

# Load stock data
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return df[["Open", "High", "Low", "Close", "Volume"]] if not df.empty else None
    except:
        return None

# LSTM model
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

# Sentiment analysis using VADER
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score['compound']

# Fetch general stock market news
def fetch_market_news():
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json()
            news_links = []
            for item in results.get("finance", {}).get("result", []):
                for news_item in item.get("news", [])[:5]:
                    title = news_item.get("title", "Untitled")
                    link = news_item.get("link", "#")
                    sentiment_score = analyze_sentiment(title)
                    news_links.append((title, link, sentiment_score))
            return news_links
    except:
        return []
    return []

# App UI
st.set_page_config(page_title="Stock Predictor AI", layout="wide")
st.title("📈 AI-Powered Stock Price Prediction & Market Insights")

# Sidebar inputs
with st.sidebar:
    st.header("🔧 Parameters")
    ticker = st.text_input("Main Stock Ticker", "AAPL").upper()
    compare_tickers = st.text_input("Compare with (comma separated)", "MSFT,GOOG").upper()
    start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
    run_prediction = st.button("🚀 Run Prediction")

# Prepare tickers
all_tickers = [ticker] + [t.strip() for t in compare_tickers.split(",") if t.strip()]

# Parallel fetching
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(load_data, t, start_date, end_date): t for t in all_tickers}
    data = {}
    for future in concurrent.futures.as_completed(futures):
        ticker_result = futures[future]
        data[ticker_result] = future.result()

valid_tickers = {k: v for k, v in data.items() if v is not None}
invalid_tickers = set(all_tickers) - set(valid_tickers.keys())

if invalid_tickers:
    st.warning(f"⚠ Could not fetch data for: {', '.join(invalid_tickers)}")

if not valid_tickers:
    st.error("❌ No valid stock data available.")
    st.stop()

# Main stock table
if run_prediction:
    main_df = valid_tickers.get(ticker)
    if main_df is not None:
        with st.expander(f"📊 {ticker} Stock Data (Last 20 Days)"):
            st.dataframe(main_df.tail(20))

if run_prediction and ticker in valid_tickers:
    df = valid_tickers[ticker]
    if len(df) < 100:
        st.warning("⚠ Not enough data for accurate prediction.")
        st.stop()

    progress_bar = st.progress(0)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    progress_bar.progress(20)
    
    seq_length = 60
    x, y = [], []
    for i in range(seq_length, len(scaled_data)):
        x.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    x, y = np.array(x), np.array(y)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    progress_bar.progress(40)
    
    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    model = create_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test), verbose=0)
    progress_bar.progress(80)

    predictions = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    progress_bar.progress(95)

    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    progress_bar.progress(100)

    st.subheader(f"📈 {ticker} Price Prediction (RMSE: {rmse:.2f})")
    
    # Plot with Animation
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index[split+seq_length:], actual_prices, label="Actual", color='blue')
    ax.set_title(f"{ticker} Stock Price Prediction (RMSE: {rmse:.2f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")

    def update(frame):
        ax.plot(df.index[split+seq_length:frame], predicted_prices[:frame], label="Predicted", color='red')
        return ax,

    ani = FuncAnimation(fig, update, frames=len(predicted_prices), blit=False, interval=100)
    st.pyplot(fig)

    # Download button
    pred_df = pd.DataFrame({
        "Date": df.index[split+seq_length:].strftime("%Y-%m-%d"),
        "Actual Price": actual_prices.flatten(),
        "Predicted Price": predicted_prices.flatten()
    })
    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions as CSV", csv, file_name=f"{ticker}_predictions.csv")

    # Comparison chart
    if len(valid_tickers) > 1:
        st.subheader("📊 Stock Price Comparison")
        fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
        for t, df_t in valid_tickers.items():
            ax_comp.plot(df_t.index, df_t['Close'], label=t)
        ax_comp.legend()
        st.pyplot(fig_comp)

    # News section with sentiment analysis
    st.subheader("📰 Latest Market News with Sentiment")
    news_links = fetch_market_news()
    for title, link, sentiment_score in news_links:
        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
        st.markdown(f"**{sentiment}** - <a href='{link}' target='_blank'>{title}</a>", unsafe_allow_html=True)

st.sidebar.info(f"Data loaded in {time.time()-start_time:.2f} seconds")
