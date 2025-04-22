import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import concurrent.futures
import time
from datetime import datetime, timedelta

# TensorFlow imports with error handling
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
except ImportError:
    try:
        from keras.models import Sequential
        from keras.layers import Dense, LSTM, Dropout
    except ImportError:
        st.error("Could not import TensorFlow/Keras. Please install required packages.")
        st.stop()

# App configuration
st.set_page_config(
    page_title="Advanced Stock Prediction", 
    page_icon="📈", 
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/your-repo/stock-prediction',
        'Report a bug': "https://github.com/your-repo/stock-prediction/issues",
        'About': "# Advanced Stock Prediction App"
    }
)

# Cached functions
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, start_date, end_date):
    """Load stock data from Yahoo Finance"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            st.warning(f"No data found for ticker: {ticker}")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def create_model(input_shape):
    """Create LSTM model with given input shape"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def detect_technical_patterns(df):
    """Detect common technical patterns in stock data"""
    if len(df) < 100:
        return None
    
    patterns = {}
    df_analysis = df.copy()
    
    try:
        # Calculate indicators
        df_analysis['SMA20'] = df_analysis['Close'].rolling(window=20, min_periods=1).mean()
        df_analysis['SMA50'] = df_analysis['Close'].rolling(window=50, min_periods=1).mean()
        df_analysis['SMA200'] = df_analysis['Close'].rolling(window=200, min_periods=1).mean()
        df_analysis['Std'] = df_analysis['Close'].rolling(window=20, min_periods=1).std()
        df_analysis['Upper_Band'] = df_analysis['SMA20'] + (df_analysis['Std'] * 2)
        df_analysis['Lower_Band'] = df_analysis['SMA20'] - (df_analysis['Std'] * 2)
        df_analysis = df_analysis.dropna()
        
        if len(df_analysis) < 2:
            return None
        
        # Get scalar values for comparison
        last_close = float(df_analysis['Close'].iloc[-1])
        upper_band = float(df_analysis['Upper_Band'].iloc[-1])
        lower_band = float(df_analysis['Lower_Band'].iloc[-1])
        sma20 = float(df_analysis['SMA20'].iloc[-1])
        sma50 = float(df_analysis['SMA50'].iloc[-1])
        
        # Moving average crossovers
        if len(df_analysis) >= 2:
            sma20_prev = float(df_analysis['SMA20'].iloc[-2])
            sma50_prev = float(df_analysis['SMA50'].iloc[-2])
            
            if sma20_prev < sma50_prev and sma20 > sma50:
                patterns['Golden Cross'] = {
                    'type': 'bullish',
                    'desc': '20-day SMA crossed above 50-day SMA',
                    'date': df_analysis.index[-1].strftime('%Y-%m-%d')
                }
            
            if sma20_prev > sma50_prev and sma20 < sma50:
                patterns['Death Cross'] = {
                    'type': 'bearish',
                    'desc': '20-day SMA crossed below 50-day SMA',
                    'date': df_analysis.index[-1].strftime('%Y-%m-%d')
                }
        
        # Bollinger Bands signals
        if last_close > upper_band:
            patterns['Overbought'] = {
                'type': 'bearish',
                'desc': 'Price above upper Bollinger Band',
                'date': df_analysis.index[-1].strftime('%Y-%m-%d')
            }
        
        if last_close < lower_band:
            patterns['Oversold'] = {
                'type': 'bullish',
                'desc': 'Price below lower Bollinger Band',
                'date': df_analysis.index[-1].strftime('%Y-%m-%d')
            }
        
        # Trend strength
        if len(df_analysis) >= 20:
            last_20_days = df_analysis['Close'].iloc[-20:].values
            price_change = (last_20_days[-1] - last_20_days[0]) / last_20_days[0]
            
            if price_change > 0.10:
                patterns['Strong Uptrend'] = {
                    'type': 'bullish',
                    'desc': f'{price_change:.1%} increase in last 20 days',
                    'date': df_analysis.index[-1].strftime('%Y-%m-%d')
                }
            
            if price_change < -0.10:
                patterns['Strong Downtrend'] = {
                    'type': 'bearish',
                    'desc': f'{price_change:.1%} decrease in last 20 days',
                    'date': df_analysis.index[-1].strftime('%Y-%m-%d')
                }
    
    except Exception as e:
        st.error(f"Error in pattern detection: {str(e)}")
        return None
    
    return patterns

    

def get_alerts():
    """Initialize or retrieve alerts from session state"""
    if 'alerts' not in st.session_state:
        st.session_state.alerts = {}
    return st.session_state.alerts

def save_alert(ticker, alert_type, price):
    """Save new alert to session state"""
    alerts = get_alerts()
    if ticker not in alerts:
        alerts[ticker] = []
    
    alerts[ticker].append({
        'type': alert_type,
        'price': float(price),
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'active': True
    })
    
    st.session_state.alerts = alerts

def delete_alert(ticker, alert_index):
    """Remove alert from session state"""
    alerts = get_alerts()
    if ticker in alerts and alert_index < len(alerts[ticker]):
        alerts[ticker].pop(alert_index)
        st.session_state.alerts = alerts

def main():
    st.title("📈 Advanced Stock Price Prediction")
    
    # Initialize session state
    if 'tab' not in st.session_state:
        st.session_state.tab = "Prediction"
    
    # Sidebar inputs
    with st.sidebar:
        st.header("Input Parameters")
        ticker = st.text_input("Stock Ticker", "AAPL").upper()
        compare_tickers = st.text_input("Compare With", "MSFT,GOOG").upper()
        start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
        run_prediction = st.button("Run Analysis", type="primary")
        
        # Display latest prices
        st.header("Latest Prices")
        if 'data' in st.session_state:
            for t, df_t in st.session_state.data.items():
                if not df_t.empty:
                    last_price = df_t['Close'].iloc[-1]
                    alerts = get_alerts()
                    alert_msg = ""
                    
                    if t in alerts:
                        for alert in alerts[t]:
                            if (alert['type'] == 'Price Above' and last_price > alert['price']) or \
                               (alert['type'] == 'Price Below' and last_price < alert['price']):
                                alert_msg = f"⚠️ {alert['type']} ${alert['price']:.2f}"
                    
                    st.metric(label=t, value=f"${float(last_price):.2f}", delta=alert_msg)
        
        # Stock Research Resources
        st.header("Stock Research Resources")
        st.markdown("""
        ### Market Data & News
        - [Yahoo Finance](https://finance.yahoo.com/)
        - [CNBC Markets](https://www.cnbc.com/markets/)
        - [Bloomberg Markets](https://www.bloomberg.com/markets)
        
        ### Technical Analysis
        - [TradingView](https://www.tradingview.com/)
        - [StockCharts](https://stockcharts.com/)
        - [Finviz](https://finviz.com/)
        
        ### Fundamental Analysis
        - [Morningstar](https://www.morningstar.com/)
        - [Seeking Alpha](https://seekingalpha.com/)
        - [MarketWatch](https://www.marketwatch.com/)
        
        ### Regulatory Filings
        - [SEC EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch.html)
        - [Investor.gov](https://www.investor.gov/)
        """)
    
    # Fetch data in parallel
    all_tickers = [ticker] + [t.strip() for t in compare_tickers.split(",") if t.strip()]
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_data, t, start_date, end_date): t for t in all_tickers}
        data = {}
        for future in concurrent.futures.as_completed(futures):
            ticker_name = futures[future]
            data[ticker_name] = future.result()
    
    valid_tickers = {k: v for k, v in data.items() if v is not None}
    st.session_state.data = valid_tickers
    
    if invalid_tickers := set(all_tickers) - set(valid_tickers.keys()):
        st.warning(f"Could not fetch data for: {', '.join(invalid_tickers)}")
    
    if not valid_tickers:
        st.error("No valid stock data available. Please check your inputs.")
        st.stop()
    
    # Create tabs
    tabs = st.tabs(["Prediction", "Technical Analysis", "Price Alerts"])
    
    # Prediction Tab
    with tabs[0]:
        if run_prediction and ticker in valid_tickers:
            df = valid_tickers[ticker][["Close"]]
            
            if len(df) < 100:
                st.warning("Insufficient data for prediction. Please select a longer time period.")
                st.stop()
            
            with st.spinner("Training prediction model..."):
                # Data preprocessing
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df)
                
                # Create sequences
                seq_length = 60
                x, y = [], []
                for i in range(seq_length, len(scaled_data)):
                    x.append(scaled_data[i-seq_length:i, 0])
                    y.append(scaled_data[i, 0])
                
                x, y = np.array(x), np.array(y)
                x = x.reshape((x.shape[0], x.shape[1], 1))
                
                # Train-test split
                split = int(0.8 * len(x))
                x_train, x_test = x[:split], x[split:]
                y_train, y_test = y[:split], y[split:]
                
                # Model training
                model = create_model((x_train.shape[1], 1))
                model.fit(x_train, y_train, batch_size=32, epochs=15, 
                         validation_data=(x_test, y_test), verbose=0)
                
                # Make predictions
                predictions = model.predict(x_test)
                predicted_prices = scaler.inverse_transform(predictions)
                actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
                rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
                
                # Plot results
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df.index[split+seq_length:], actual_prices, 'b-', label='Actual')
                ax.plot(df.index[split+seq_length:], predicted_prices, 'r--', label='Predicted')
                
                # Add alert markers if any exist
                alerts = get_alerts()
                if ticker in alerts:
                    for alert in alerts[ticker]:
                        if alert['active']:
                            ax.axhline(y=alert['price'], color='g', linestyle='-', alpha=0.3)
                            ax.text(df.index[len(df) // 2], alert['price'], 
                                   f" {alert['type']} ${alert['price']:.2f}", 
                                   verticalalignment='bottom')
                
                ax.set_title(f"{ticker} Price Prediction (RMSE: {rmse:.2f})")
                ax.legend()
                st.pyplot(fig)
            
            # Show comparison chart if multiple tickers
            if len(valid_tickers) > 1:
                st.subheader("Stock Comparison")
                fig_comp, ax_comp = plt.subplots(figsize=(12, 5))
                for t, df_t in valid_tickers.items():
                    ax_comp.plot(df_t.index, df_t['Close'], label=t)
                ax_comp.set_title("Price Comparison")
                ax_comp.legend()
                st.pyplot(fig_comp)
    
    # Technical Analysis Tab
    with tabs[1]:
        if ticker in valid_tickers:
            df = valid_tickers[ticker]
            patterns = detect_technical_patterns(df)
            
            if patterns:
                st.subheader("Detected Patterns")
                cols = st.columns(3)
                col_idx = 0
                
                for pattern, details in patterns.items():
                    with cols[col_idx]:
                        if details['type'] == 'bullish':
                            st.success(f"**{pattern}**")
                        elif details['type'] == 'bearish':
                            st.error(f"**{pattern}**")
                        else:
                            st.info(f"**{pattern}**")
                        
                        st.write(details['desc'])
                        st.caption(f"Detected on {details['date']}")
                    
                    col_idx = (col_idx + 1) % 3
            else:
                st.info("No significant patterns detected")
            
            # Technical indicators chart
            st.subheader("Technical Indicators")
            df_ta = df.copy()
            df_ta['SMA20'] = df_ta['Close'].rolling(window=20).mean()
            df_ta['SMA50'] = df_ta['Close'].rolling(window=50).mean()
            df_ta['SMA200'] = df_ta['Close'].rolling(window=200).mean()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_ta.index, df_ta['Close'], label='Price', alpha=0.8)
            ax.plot(df_ta.index, df_ta['SMA20'], label='20-day SMA', linestyle='--')
            ax.plot(df_ta.index, df_ta['SMA50'], label='50-day SMA', linestyle='--')
            ax.plot(df_ta.index, df_ta['SMA200'], label='200-day SMA', linestyle='--')
            
            if patterns:
                for pattern, details in patterns.items():
                    pattern_date = pd.to_datetime(details['date'])
                    if pattern_date in df_ta.index:
                        idx = df_ta.index.get_loc(pattern_date)
                        if idx > 0 and idx < len(df_ta):
                            price = df_ta['Close'].iloc[idx]
                            marker = 'go' if details['type'] == 'bullish' else 'ro'
                            ax.plot(pattern_date, price, marker, markersize=10)
            
            ax.set_title(f"{ticker} Technical Indicators")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    # Alerts Tab
    with tabs[2]:
        st.header("Price Alerts Management")
        
        if ticker in valid_tickers:
            current_price = valid_tickers[ticker]['Close'].iloc[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                alert_type = st.selectbox(
                    "Alert Condition", 
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
                st.success(f"Alert set for {ticker}: {alert_type} ${alert_price:.2f}")
        
        # Display existing alerts
        alerts = get_alerts()
        
        if not any(alerts.values()):
            st.info("No alerts currently set")
        else:
            st.subheader("Your Active Alerts")
            for ticker_name, ticker_alerts in alerts.items():
                if ticker_alerts:
                    st.write(f"**{ticker_name}**")
                    
                    for i, alert in enumerate(ticker_alerts):
                        cols = st.columns([3, 1, 1])
                        with cols[0]:
                            st.write(f"{alert['type']} ${alert['price']:.2f}")
                        with cols[1]:
                            st.write(alert['created_at'])
                        with cols[2]:
                            if st.button("Delete", key=f"del_{ticker_name}_{i}"):
                                delete_alert(ticker_name, i)
                                st.rerun()
    
    st.sidebar.info(f"Data loaded in {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()
