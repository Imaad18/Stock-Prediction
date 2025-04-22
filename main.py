import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import concurrent.futures
import time
from datetime import datetime

# Safe imports of TensorFlow components
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
except ImportError:
    try:
        from keras.models import Sequential
        from keras.layers import Dense, LSTM, Dropout
    except ImportError:
        st.error("Could not import TensorFlow/Keras. Please check installation.")
        st.stop()

# Enable caching for expensive operations
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
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

# Function to calculate technical indicators manually without pandas_ta
def calculate_indicators(df):
    # Create a copy to avoid modifying original dataframe
    df_copy = df.copy()
    
    # Calculate Simple Moving Averages
    df_copy['SMA20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['SMA50'] = df_copy['Close'].rolling(window=50).mean()
    df_copy['SMA200'] = df_copy['Close'].rolling(window=200).mean()
    
    # Calculate Bollinger Bands
    df_copy['Middle_Band'] = df_copy['SMA20']
    df_copy['Std_Dev'] = df_copy['Close'].rolling(window=20).std()
    df_copy['Upper_Band'] = df_copy['Middle_Band'] + (df_copy['Std_Dev'] * 2)
    df_copy['Lower_Band'] = df_copy['Middle_Band'] - (df_copy['Std_Dev'] * 2)
    
    # Calculate RSI (14-period)
    delta = df_copy['Close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    
    return df_copy

# Function to detect patterns and technical events
def detect_patterns(df):
    if df is None or len(df) < 100:
        return None
    
    patterns = {}
    
    # Calculate indicators
    df_analysis = calculate_indicators(df)
    
    # Drop NaN values
    df_analysis = df_analysis.dropna()
    
    if len(df_analysis) < 2:
        return None
    
    # Detect basic patterns
    
    # Moving average crossovers
    if df_analysis['SMA20'].iloc[-2] < df_analysis['SMA50'].iloc[-2] and df_analysis['SMA20'].iloc[-1] > df_analysis['SMA50'].iloc[-1]:
        patterns['Golden Cross'] = {
            'type': 'bullish',
            'desc': 'Short-term momentum is turning positive (20-day SMA crossed above 50-day SMA)',
            'date': df_analysis.index[-1].strftime('%Y-%m-%d')
        }
    
    if df_analysis['SMA20'].iloc[-2] > df_analysis['SMA50'].iloc[-2] and df_analysis['SMA20'].iloc[-1] < df_analysis['SMA50'].iloc[-1]:
        patterns['Death Cross'] = {
            'type': 'bearish',
            'desc': 'Short-term momentum is turning negative (20-day SMA crossed below 50-day SMA)',
            'date': df_analysis.index[-1].strftime('%Y-%m-%d')
        }
    
    # Support/Resistance
    last_close = df_analysis['Close'].iloc[-1]
    
    # Check if price is near SMA200 (potential support/resistance)
    if 'SMA200' in df_analysis.columns:
        distance_to_sma200 = abs(last_close - df_analysis['SMA200'].iloc[-1]) / last_close
        if distance_to_sma200 < 0.03:  # Within 3%
            patterns['Key Level'] = {
                'type': 'neutral',
                'desc': f'Price is near 200-day SMA (${df_analysis["SMA200"].iloc[-1]:.2f}), a key support/resistance level',
                'date': df_analysis.index[-1].strftime('%Y-%m-%d')
            }
    
    # Bollinger Band signals
    if df_analysis['Close'].iloc[-1] > df_analysis['Upper_Band'].iloc[-1]:
        patterns['Overbought'] = {
            'type': 'bearish',
            'desc': 'Price is above upper Bollinger Band, potentially overbought',
            'date': df_analysis.index[-1].strftime('%Y-%m-%d')
        }
    
    if df_analysis['Close'].iloc[-1] < df_analysis['Lower_Band'].iloc[-1]:
        patterns['Oversold'] = {
            'type': 'bullish',
            'desc': 'Price is below lower Bollinger Band, potentially oversold',
            'date': df_analysis.index[-1].strftime('%Y-%m-%d')
        }
    
    # RSI signals
    if 'RSI' in df_analysis.columns:
        if df_analysis['RSI'].iloc[-1] > 70:
            patterns['RSI Overbought'] = {
                'type': 'bearish',
                'desc': f'RSI is overbought at {df_analysis["RSI"].iloc[-1]:.1f}',
                'date': df_analysis.index[-1].strftime('%Y-%m-%d')
            }
        
        if df_analysis['RSI'].iloc[-1] < 30:
            patterns['RSI Oversold'] = {
                'type': 'bullish',
                'desc': f'RSI is oversold at {df_analysis["RSI"].iloc[-1]:.1f}',
                'date': df_analysis.index[-1].strftime('%Y-%m-%d')
            }
    
    # Trend strength
    last_20_days = df_analysis['Close'].iloc[-20:].values
    price_change = (last_20_days[-1] - last_20_days[0]) / last_20_days[0]
    
    if price_change > 0.10:  # 10% up in 20 days
        patterns['Strong Uptrend'] = {
            'type': 'bullish',
            'desc': f'Strong uptrend: {price_change:.1%} increase in last 20 days',
            'date': df_analysis.index[-1].strftime('%Y-%m-%d')
        }
    
    if price_change < -0.10:  # 10% down in 20 days
        patterns['Strong Downtrend'] = {
            'type': 'bearish',
            'desc': f'Strong downtrend: {price_change:.1%} decrease in last 20 days',
            'date': df_analysis.index[-1].strftime('%Y-%m-%d')
        }
    
    return patterns, df_analysis

# Helper function to load and save alerts
def get_alerts():
    if 'alerts' not in st.session_state:
        st.session_state.alerts = {}
    return st.session_state.alerts

def save_alert(ticker, alert_type, price, active=True):
    alerts = get_alerts()
    if ticker not in alerts:
        alerts[ticker] = []
    
    # Add new alert
    alerts[ticker].append({
        'type': alert_type,
        'price': float(price),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'active': active
    })
    
    # Update session state
    st.session_state.alerts = alerts

def delete_alert(ticker, alert_index):
    alerts = get_alerts()
    if ticker in alerts and alert_index < len(alerts[ticker]):
        alerts[ticker].pop(alert_index)
        st.session_state.alerts = alerts

# Streamlit App
st.title("⚡ Advanced Stock Price Prediction")

# Initialize session state
if 'tab' not in st.session_state:
    st.session_state.tab = "Prediction"

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    ticker = st.text_input("Main Stock Ticker", "AAPL").upper()
    compare_tickers = st.text_input("Compare with (comma separated)", "MSFT,GOOG").upper()
    start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime(datetime.today().strftime('%Y-%m-%d')))  # Default to today
    
    # Add a button to trigger the prediction
    run_prediction = st.button("Run Prediction", type="primary")

# Create tabs
tabs = st.tabs(["Prediction", "Pattern Recognition", "Alerts"])

# Get all tickers to fetch
all_tickers = [ticker] + [t.strip() for t in compare_tickers.split(",") if t.strip()]

# Fetch data in parallel
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(load_data, t, start_date, end_date): t for t in all_tickers}
    data = {}
    for future in concurrent.futures.as_completed(futures):
        ticker_name = futures[future]
        data[ticker_name] = future.result()

# Filter out None values
valid_tickers = {k: v for k, v in data.items() if v is not None}
invalid_tickers = set(all_tickers) - set(valid_tickers.keys())

if invalid_tickers:
    st.warning(f"Could not fetch data for: {', '.join(invalid_tickers)}")

if not valid_tickers:
    st.error("No valid stock data available. Please check your inputs.")
    st.stop()

# Process alerts
with tabs[2]:
    st.header("Price Alerts")
    st.write("Set alerts for specific price levels or conditions.")
    
    if ticker in valid_tickers:
        current_price = valid_tickers[ticker]['Close'].iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            alert_type = st.selectbox(
                "Alert Type", 
                ["Price Above", "Price Below", "Prediction Above", "Prediction Below"]
            )
        
        with col2:
            alert_price = st.number_input(
                "Target Price", 
                min_value=0.01, 
                value=float(current_price),
                format="%.2f"
            )
        
        if st.button("Add Alert"):
            save_alert(ticker, alert_type, alert_price)
            st.success(f"Alert added for {ticker}: {alert_type} ${alert_price:.2f}")
    
    # Display existing alerts
    alerts = get_alerts()
    
    if not any(alerts.values()):
        st.info("No alerts set. Add your first alert above.")
    else:
        for t, ticker_alerts in alerts.items():
            if ticker_alerts:
                st.subheader(f"{t} Alerts")
                
                for i, alert in enumerate(ticker_alerts):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{alert['type']}**: ${alert['price']:.2f}")
                    
                    with col2:
                        st.write(f"Created: {alert['created_at']}")
                    
                    with col3:
                        if st.button("Delete", key=f"del_{t}_{i}"):
                            delete_alert(t, i)
                            st.rerun()

# Pattern Recognition tab
with tabs[1]:
    st.header("Technical Patterns & Events")
    
    if ticker in valid_tickers:
        df = valid_tickers[ticker]
        
        # Detect patterns
        result = detect_patterns(df)
        if result is not None:
            patterns, df_ta = result
            
            if patterns:
                # Display patterns in cards
                for pattern_name, details in patterns.items():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if details['type'] == 'bullish':
                            st.markdown("### 🟢")
                        elif details['type'] == 'bearish':
                            st.markdown("### 🔴")
                        else:
                            st.markdown("### ⚪")
                    
                    with col2:
                        st.subheader(pattern_name)
                        st.write(details['desc'])
                        st.caption(f"Detected on {details['date']}")
                    
                    st.divider()
            else:
                st.info("No significant patterns detected in the current timeframe.")
            
            # Display technical indicators
            st.subheader("Technical Indicators")
            
            # Create indicator chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_ta.index, df_ta['Close'], label='Price')
            ax.plot(df_ta.index, df_ta['SMA20'], label='20-day SMA', linestyle='--')
            ax.plot(df_ta.index, df_ta['SMA50'], label='50-day SMA', linestyle='--')
            ax.plot(df_ta.index, df_ta['SMA200'], label='200-day SMA', linestyle='--')
            
            # Highlighting potential patterns
            if patterns:
                for pattern, details in patterns.items():
                    pattern_date = pd.to_datetime(details['date'])
                    if pattern_date in df_ta.index:
                        idx = df_ta.index.get_loc(pattern_date)
                        if idx > 0 and idx < len(df_ta):
                            price = df_ta['Close'].iloc[idx]
                            if details['type'] == 'bullish':
                                ax.plot(pattern_date, price, 'go', markersize=10)
                            elif details['type'] == 'bearish':
                                ax.plot(pattern_date, price, 'ro', markersize=10)
                            else:
                                ax.plot(pattern_date, price, 'ko', markersize=10)
            
            ax.set_title(f"{ticker} Price with Moving Averages")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Not enough data to analyze patterns. Select a longer time period.")

# Prediction tab content
with tabs[0]:
    # Show main stock data quickly
    if run_prediction:
        main_df = valid_tickers.get(ticker)
        if main_df is not None:
            with st.expander(f"📊 {ticker} Complete Stock Data (Last 20 Days)"):
                # Show full data table with all columns
                st.dataframe(main_df.tail(20))

    # Prediction only runs when button is clicked and we have enough data
    if run_prediction and ticker in valid_tickers:
        df = valid_tickers[ticker][["Close"]]  # Use only Close for prediction
        if len(df) < 100:
            st.warning("Not enough data for accurate prediction. Please select a longer time period.")
            st.stop()
        
        # Progress bar for user feedback
        progress_bar = st.progress(0)
        
        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        progress_bar.progress(20)
        
        # Create sequences
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
        
        # Create and train model
        model = create_model((x_train.shape[1], 1))
        history = model.fit(x_train, y_train, 
                           batch_size=32, 
                           epochs=15, 
                           validation_data=(x_test, y_test),
                           verbose=0)
        progress_bar.progress(80)
        
        # Make predictions
        predictions = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
        progress_bar.progress(95)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index[split+seq_length:], actual_prices, 'b-', label='Actual')
        ax.plot(df.index[split+seq_length:], predicted_prices, 'r--', label='Predicted')
        
        # Add alerts if any exist for this ticker
        alerts = get_alerts()
        if ticker in alerts and alerts[ticker]:
            for alert in alerts[ticker]:
                if alert['active']:
                    # Draw horizontal line for the alert
                    ax.axhline(y=alert['price'], color='g', linestyle='-', alpha=0.5)
                    # Add annotation
                    ax.text(df.index[len(df) // 2], alert['price'], 
                           f" {alert['type']} ${alert['price']:.2f}", 
                           verticalalignment='bottom', color='green')
        
        ax.set_title(f"{ticker} Price Prediction (RMSE: {rmse:.2f})")
        ax.legend()
        progress_bar.progress(100)
        st.pyplot(fig)
        
        # Show comparison charts if other tickers exist
        if len(valid_tickers) > 1:
            st.subheader("Comparison with Other Stocks")
            fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
            for t, df_t in valid_tickers.items():
                ax_comp.plot(df_t.index, df_t['Close'], label=t)
            ax_comp.set_title("Stock Price Comparison")
            ax_comp.legend()
            st.pyplot(fig_comp)

# Show latest prices in sidebar
with st.sidebar:
    st.header("Latest Prices")
    for t, df_t in valid_tickers.items():
        if not df_t.empty:
            try:
                # Get the last price (ensure it's a scalar value)
                last_price = float(df_t['Close'].iloc[-1])
                
                # Get price change from previous close
                if len(df_t) > 1:
                    prev_price = float(df_t['Close'].iloc[-2])
                    price_change = last_price - prev_price
                    percent_change = (price_change / prev_price) * 100
                else:
                    price_change = 0
                    percent_change = 0
                
                # Check if there are any triggered alerts
                alerts = get_alerts()
                alert_triggered = False
                alert_message = ""
                
                if t in alerts:
                    for alert in alerts[t]:
                        if alert['active']:
                            if (alert['type'] == 'Price Above' and last_price > alert['price']) or \
                               (alert['type'] == 'Price Below' and last_price < alert['price']):
                                alert_triggered = True
                                alert_message = f"⚠️ Alert: {alert['type']} ${alert['price']:.2f}"
                
                # Display price with change and alert if triggered
                st.metric(
                    label=t,
                    value=f"${last_price:.2f}",
                    delta=f"{price_change:.2f} ({percent_change:.2f}%)",
                    delta_color="inverse" if price_change < 0 else "normal"
                )
                
                if alert_triggered:
                    st.warning(alert_message)
                    
            except Exception as e:
                st.error(f"Error displaying price for {t}: {str(e)}")
    
    # Add stock information websites
    st.sidebar.header("Stock Information Resources")
    st.sidebar.markdown("""
    ### Research & News
    - [Yahoo Finance](https://finance.yahoo.com/)
    - [CNBC](https://www.cnbc.com/stock-markets/)
    - [MarketWatch](https://www.marketwatch.com/)
    - [Bloomberg](https://www.bloomberg.com/markets/stocks)
    - [Investing.com](https://www.investing.com/)
    
    ### Advanced Analysis
    - [TradingView](https://www.tradingview.com/)
    - [Seeking Alpha](https://seekingalpha.com/)
    - [Morningstar](https://www.morningstar.com/)
    - [Finviz](https://finviz.com/)
    
    ### SEC Filings
    - [SEC EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch)
    """)

st.sidebar.info(f"Data loaded in {time.time()-start_time:.2f} seconds")
