# Import Required Libraries
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
import feedparser
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta

# Streamlit App Configuration
st.set_page_config(layout="wide", page_title="📈 Advanced Stock Prediction")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {border-radius: 5px;}
    .stSelectbox, .stTextInput, .stDateInput {margin-bottom: 15px;}
    .plot-container {border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .metric-card {background: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .ticker-header {color: #2c3e50; font-weight: 700;}
    .positive {color: #27ae60;}
    .negative {color: #e74c3c;}
</style>
""", unsafe_allow_html=True)

# App Title
st.title("📈 Advanced Stock Price Prediction with LSTM")

# Sidebar: Inputs
st.sidebar.header("📌 Stock Input Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    ticker = st.text_input("Main Ticker:", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
with col2:
    tickers_compare = st.text_input("Compare with:", "MSFT,GOOG")
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

# Advanced Options
st.sidebar.header("⚙️ Model Parameters")
epochs = st.sidebar.slider("Epochs", 10, 100, 20)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
sequence_length = st.sidebar.slider("Sequence Length", 30, 90, 60)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

# Combine tickers for main and comparison stocks
tickers = [ticker] + [x.strip() for x in tickers_compare.split(',') if x.strip()]

# Download Stock Data for each ticker
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1))
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

dfs = {}
for ticker in tickers:
    df_temp = load_data(ticker, start_date, end_date)
    if df_temp is not None:
        dfs[ticker] = df_temp

# Check if any data was fetched successfully
if not dfs:
    st.error("❌ No valid stock data fetched. Please check your ticker symbols and date range.")
    st.stop()

# Show Full Stock Data Table for the selected ticker
st.subheader(f"📊 {ticker} Stock Data")
with st.expander("View Raw Data"):
    st.dataframe(dfs[ticker][["Open", "High", "Low", "Close", "Volume"]].tail(100).style.format("{:.2f}"))

# Main Ticker Analysis
if ticker in dfs:
    df = dfs[ticker]
    
    # 1. Candlestick Chart with Volume
    st.subheader(f"📈 {ticker} Candlestick Chart with Volume")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'Volume'), 
                       row_width=[0.2, 0.7])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                name='Price'), row=1, col=1)
    
    # Volume
    colors = ['green' if row['Open'] - row['Close'] >= 0 else 'red' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, 
                         y=df['Volume'],
                         marker_color=colors,
                         name='Volume'), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=False, 
                      xaxis_rangeslider_visible=False,
                      margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # 2. Technical Indicators
    st.subheader(f"📊 {ticker} Technical Indicators")
    
    # Calculate indicators
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], name='20-day MA', line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], name='50-day MA', line=dict(color='green', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_200'], name='200-day MA', line=dict(color='red', width=1.5)))
    
    fig.update_layout(height=500, hovermode='x unified',
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    # 3. LSTM Prediction Model
    st.subheader(f"🤖 {ticker} LSTM Price Prediction")
    
    # Use only Close price for prediction
    df_lstm = df[['Close']]
    
    # Normalize Data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_lstm)
    
    # Create Sequences for LSTM
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    x, y = np.array(x), np.array(y)
    
    # Reshape x_train and x_test to be 3D for LSTM
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # Reshape for LSTM input
    
    # Train-Test Split
    split = int((1 - test_size) * len(x))
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
    
    # Train with progress bar
    with st.spinner('Training LSTM model...'):
        progress_bar = st.progress(0)
        for epoch in range(epochs):
            model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
            progress_bar.progress((epoch + 1) / epochs)
    
    # Predictions
    predicted = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Create DataFrame for results
    results = pd.DataFrame({
        'Date': df.index[-len(real_prices):],
        'Actual': real_prices.flatten(),
        'Predicted': predicted_prices.flatten()
    })
    
    # Plot predictions
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=results['Date'], y=results['Actual'], 
                            name='Actual Price', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=results['Date'], y=results['Predicted'], 
                            name='Predicted Price', line=dict(color='red', width=2)))
    
    fig.update_layout(
        height=500,
        title=f'{ticker} Actual vs Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Evaluation Metrics
    st.subheader("📉 Model Evaluation Metrics")
    
    mse = mean_squared_error(real_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_prices, predicted_prices)
    r2 = r2_score(real_prices, predicted_prices)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Squared Error", f"{mse:.2f}")
    col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
    col3.metric("Mean Absolute Error", f"{mae:.2f}")
    col4.metric("R² Score", f"{r2:.4f}")
    
    # Comparison with other tickers
    if len(tickers) > 1:
        st.subheader("🔍 Comparison with Other Stocks")
        
        # Normalize all prices to percentage change for fair comparison
        comparison_data = []
        for t in tickers:
            if t in dfs:
                temp_df = dfs[t][['Close']].rename(columns={'Close': t})
                # Calculate percentage change from start date
                temp_df[t] = (temp_df[t] / temp_df[t].iloc[0] - 1) * 100
                comparison_data.append(temp_df)
        
        if comparison_data:
            comparison_df = pd.concat(comparison_data, axis=1)
            
            fig = px.line(comparison_df, x=comparison_df.index, y=comparison_df.columns,
                         title='Normalized Price Comparison (Base 100)',
                         labels={'value': 'Percentage Change (%)', 'variable': 'Ticker'})
            
            fig.update_layout(height=500, hovermode='x unified',
                            legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

# Sidebar: Latest Price Info
st.sidebar.header("💰 Latest Market Data")
for t in tickers:
    if t in dfs:
        df_temp = dfs[t]
        if not df_temp.empty:
            latest_price = df_temp['Close'].iloc[-1]
            prev_price = df_temp['Close'].iloc[-2] if len(df_temp) > 1 else latest_price
            change = ((latest_price - prev_price) / prev_price) * 100
            change_color = "positive" if change >= 0 else "negative"
            
            st.sidebar.markdown(f"""
            <div class="metric-card">
                <h3 class="ticker-header">{t}</h3>
                <p>Price: ${latest_price:.2f}</p>
                <p class="{change_color}">Change: {change:.2f}%</p>
                <p>Date: {df_temp.index[-1].strftime('%Y-%m-%d')}</p>
            </div>
            """, unsafe_allow_html=True)

# Sidebar: News Feed
st.sidebar.header("📢 Financial News")
news_query = ticker if ticker else "Stock Market"
rss_url = f"https://news.google.com/rss/search?q={news_query}+stock&hl=en-US&gl=US&ceid=US:en"

try:
    feed = feedparser.parse(rss_url)
    max_articles = 5
    if feed.entries:
        for entry in feed.entries[:max_articles]:
            st.sidebar.markdown(f"""
            <div class="metric-card" style="margin-bottom: 10px;">
                <a href="{entry.link}" target="_blank" style="color: inherit; text-decoration: none;">
                    <p style="font-weight: 600; margin-bottom: 5px;">{entry.title}</p>
                    <p style="font-size: 0.8em; color: #666;">{entry.published if 'published' in entry else ''}</p>
                </a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.sidebar.write("No news found.")
except Exception as e:
    st.sidebar.error("Error fetching news feed.")
