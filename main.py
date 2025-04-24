import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import concurrent.futures
import time
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Try importing statsmodels components with graceful fallback
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("Advanced forecasting features unavailable. Install 'statsmodels' package for full functionality.")

# Initialize the start time at the beginning of the script
start_time = time.time()

# Enable caching for expensive operations
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, start_date, end_date):
    try:
        if not isinstance(ticker, str) or not ticker.strip():
            return None
            
        ticker = ticker.strip()  # Ensure no whitespace
        
        # Add 1 day to end_date to ensure we get the most recent data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Function to create a linear regression model
def create_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to calculate technical indicators manually
def calculate_indicators(df):
    try:
        if df is None or len(df) < 10:
            return df
            
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
        # Use .loc instead of direct assignment to avoid Series truth value is ambiguous error
        gain.loc[gain < 0] = 0
        loss.loc[loss > 0] = 0
        loss = abs(loss)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Replace zeros with small value to avoid division by zero
        avg_loss = avg_loss.replace(0, 0.001)  
        
        rs = avg_gain / avg_loss
        df_copy['RSI'] = 100 - (100 / (1 + rs))
        
        return df_copy
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return df  # Return original dataframe if calculation fails

# Function to detect patterns and technical events
def detect_patterns(df):
    if df is None or len(df) < 100:
        return None, None
    
    patterns = {}
    
    try:
        # Calculate indicators
        df_analysis = calculate_indicators(df)
        
        # Drop NaN values
        df_analysis = df_analysis.dropna()
        
        if df_analysis is None or len(df_analysis) < 2:
            return None, None
        
        # Detect basic patterns
        
        # Moving average crossovers - ensure we're comparing scalar values
        if 'SMA20' in df_analysis.columns and 'SMA50' in df_analysis.columns:
            # Get scalar values instead of Series objects
            sma20_prev = float(df_analysis['SMA20'].iloc[-2])
            sma50_prev = float(df_analysis['SMA50'].iloc[-2])
            sma20_curr = float(df_analysis['SMA20'].iloc[-1])
            sma50_curr = float(df_analysis['SMA50'].iloc[-1])
            
            if sma20_prev < sma50_prev and sma20_curr > sma50_curr:
                patterns['Golden Cross'] = {
                    'type': 'bullish',
                    'desc': 'Short-term momentum is turning positive (20-day SMA crossed above 50-day SMA)',
                    'date': df_analysis.index[-1].strftime('%Y-%m-%d')
                }
            
            if sma20_prev > sma50_prev and sma20_curr < sma50_curr:
                patterns['Death Cross'] = {
                    'type': 'bearish',
                    'desc': 'Short-term momentum is turning negative (20-day SMA crossed below 50-day SMA)',
                    'date': df_analysis.index[-1].strftime('%Y-%m-%d')
                }
        
        # Support/Resistance
        last_close = float(df_analysis['Close'].iloc[-1])
        
        # Check if price is near SMA200 (potential support/resistance)
        if 'SMA200' in df_analysis.columns:
            sma200_value = float(df_analysis['SMA200'].iloc[-1])
            distance_to_sma200 = abs(last_close - sma200_value) / last_close
            if distance_to_sma200 < 0.03:  # Within 3%
                patterns['Key Level'] = {
                    'type': 'neutral',
                    'desc': f'Price is near 200-day SMA (${sma200_value:.2f}), a key support/resistance level',
                    'date': df_analysis.index[-1].strftime('%Y-%m-%d')
                }
        
        # Bollinger Band signals - use scalar values
        if 'Upper_Band' in df_analysis.columns:
            upper_band = float(df_analysis['Upper_Band'].iloc[-1])
            if last_close > upper_band:
                patterns['Overbought'] = {
                    'type': 'bearish',
                    'desc': 'Price is above upper Bollinger Band, potentially overbought',
                    'date': df_analysis.index[-1].strftime('%Y-%m-%d')
                }
        
        if 'Lower_Band' in df_analysis.columns:
            lower_band = float(df_analysis['Lower_Band'].iloc[-1])
            if last_close < lower_band:
                patterns['Oversold'] = {
                    'type': 'bullish',
                    'desc': 'Price is below lower Bollinger Band, potentially oversold',
                    'date': df_analysis.index[-1].strftime('%Y-%m-%d')
                }
        
        # RSI signals - use scalar values
        if 'RSI' in df_analysis.columns:
            rsi_value = float(df_analysis['RSI'].iloc[-1])
            if rsi_value > 70:
                patterns['RSI Overbought'] = {
                    'type': 'bearish',
                    'desc': f'RSI is overbought at {rsi_value:.1f}',
                    'date': df_analysis.index[-1].strftime('%Y-%m-%d')
                }
            
            if rsi_value < 30:
                patterns['RSI Oversold'] = {
                    'type': 'bullish',
                    'desc': f'RSI is oversold at {rsi_value:.1f}',
                    'date': df_analysis.index[-1].strftime('%Y-%m-%d')
                }
        
        # Trend strength
        last_20_days = df_analysis['Close'].iloc[-20:].values
        price_change = (float(last_20_days[-1]) - float(last_20_days[0])) / float(last_20_days[0])
        
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
    except Exception as e:
        st.error(f"Error detecting patterns: {str(e)}")
        return {}, df  # Return empty patterns if detection fails

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
        'active': active,
        'triggered': False
    })
    
    # Update session state
    st.session_state.alerts = alerts

def delete_alert(ticker, alert_index):
    alerts = get_alerts()
    if ticker in alerts and alert_index < len(alerts[ticker]):
        alerts[ticker].pop(alert_index)
        st.session_state.alerts = alerts
        return True
    return False


@st.cache_data(ttl=3600, show_spinner=False)
def create_forecast_models(df, forecast_days=30):
    """
    Create and return multiple forecasting models.
    
    Args:
        df: DataFrame with 'Close' prices
        forecast_days: Number of days to forecast
        
    Returns:
        Dictionary with forecast results from different models
    """
    if df is None or len(df) < 60:
        return None
    
    forecasts = {}
    close_prices = df['Close'].values
    dates = df.index
    
    try:
        # Only try ARIMA and Exponential if statsmodels is available
        if STATSMODELS_AVAILABLE:
            # 1. ARIMA Model
            try:
                # Try simple ARIMA model (1,1,1)
                arima_model = ARIMA(close_prices, order=(1, 1, 1))
                arima_result = arima_model.fit()
                arima_forecast = arima_result.forecast(steps=forecast_days)
                forecasts['ARIMA'] = arima_forecast
            except Exception as e:
                st.warning(f"ARIMA model error: {str(e)}")

            # 2. Exponential Smoothing Model
            try:
                # Holt-Winters exponential smoothing
                exp_model = ExponentialSmoothing(
                    close_prices, 
                    trend='add', 
                    seasonal='add', 
                    seasonal_periods=5
                )
                exp_result = exp_model.fit()
                exp_forecast = exp_result.forecast(forecast_days)
                forecasts['Exponential'] = exp_forecast
            except Exception as e:
                st.warning(f"Exponential Smoothing error: {str(e)}")
        
        # 3. Linear Regression (simple trend-based) - always available as sklearn should be installed
        try:
            # Use linear regression for trend projection
            x = np.arange(len(close_prices)).reshape(-1, 1)
            y = close_prices
            
            model = LinearRegression()
            model.fit(x, y)
            
            # Forecast future dates
            future_x = np.arange(len(close_prices), len(close_prices) + forecast_days).reshape(-1, 1)
            linear_forecast = model.predict(future_x)
            forecasts['Linear'] = linear_forecast
        except Exception as e:
            st.warning(f"Linear trend forecast error: {str(e)}")
        
        # Generate future dates for the forecasts
        last_date = dates[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        return {
            'forecasts': forecasts,
            'actual_dates': dates,
            'actual_prices': close_prices,
            'future_dates': future_dates,
            'last_price': float(close_prices[-1])
        }
        
    except Exception as e:
        st.error(f"Error creating forecast models: {str(e)}")
        return None
    
# Streamlit App
st.title("⚡ Stock Analysis and Prediction")

# Initialize session state
if 'tab' not in st.session_state:
    st.session_state.tab = "Prediction"

if 'alerts' not in st.session_state:
    st.session_state.alerts = {}

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    ticker = st.text_input("Main Stock Ticker", "AAPL").strip().upper()
    compare_tickers = st.text_input("Compare with (comma separated)", "MSFT,GOOG").split(",")
    start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", datetime.now())
    
    # Add a button to trigger the prediction
    run_prediction = st.button("Run Analysis", type="primary")

# Create tabs
tabs = st.tabs(["Prediction", "Pattern Recognition", "Alerts", "Forecast"])

# Get all tickers to fetch
all_tickers = [ticker]
for t in compare_tickers:
    sanitized = t.strip().upper()
    if sanitized.length > 1 and sanitized not in all_tickers:
    all_tickers.append(sanitized)
    all_tickers = list(set(all_tickers))

# Fetch data in parallel
valid_tickers = {}
invalid_tickers = set()

with st.spinner("Loading stock data..."):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_data, t, start_date, end_date): t for t in all_tickers}
        data = {}
        for future in concurrent.futures.as_completed(futures):
            ticker_name = futures[future]
            result = future.result()
            if result is not None and not result.empty:
                valid_tickers[ticker_name] = result
            else:
                invalid_tickers.add(ticker_name)

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
        current_price = float(valid_tickers[ticker]['Close'].iloc[-1])
        
        col1, col2 = st.columns(2)
        
        with col1:
            alert_type = st.selectbox(
                "Alert Type", 
                ["Price Above", "Price Below"]
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
                        status = "✅" if alert['triggered'] else "⏳"
                        st.write(f"{status} **{alert['type']}**: ${alert['price']:.2f}")
                    
                    with col2:
                        st.write(f"Created: {alert['created_at']}")
                    
                    with col3:
                        if st.button("Delete", key=f"del_{t}_{i}"):
                            if delete_alert(t, i):
                                st.success(f"Alert deleted")
                                time.sleep(0.5)  # Give time for the success message to appear
                                st.rerun()

# Pattern Recognition tab
with tabs[1]:
    st.header("Technical Patterns & Events")
    
    if ticker in valid_tickers:
        df = valid_tickers[ticker]
        
        # Detect patterns
        with st.spinner("Analyzing patterns..."):
            result = detect_patterns(df)
            
        if result is not None:
            patterns, df_ta = result
            
            if patterns and len(patterns) > 0:
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
            
            # Display technical indicators if df_ta is not None
            if df_ta is not None and not df_ta.empty:
                st.subheader("Technical Indicators")
                
                try:
                    # Create indicator chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(df_ta.index, df_ta['Close'], label='Price')
                    
                    # Only plot SMAs if they exist in the dataframe
                    for sma_column, label in [('SMA20', '20-day SMA'), ('SMA50', '50-day SMA'), ('SMA200', '200-day SMA')]:
                        if sma_column in df_ta.columns:
                            ax.plot(df_ta.index, df_ta[sma_column], label=label, linestyle='--')
                    
                    # Highlighting potential patterns
                    if patterns and len(patterns) > 0:
                        for pattern, details in patterns.items():
                            try:
                                pattern_date = pd.to_datetime(details['date'])
                                if pattern_date in df_ta.index:
                                    idx = df_ta.index.get_loc(pattern_date)
                                    if idx > 0 and idx < len(df_ta):
                                        price = float(df_ta['Close'].iloc[idx])
                                        if details['type'] == 'bullish':
                                            ax.plot(pattern_date, price, 'go', markersize=10)
                                        elif details['type'] == 'bearish':
                                            ax.plot(pattern_date, price, 'ro', markersize=10)
                                        else:
                                            ax.plot(pattern_date, price, 'ko', markersize=10)
                            except Exception as e:
                                st.error(f"Error highlighting pattern {pattern}: {str(e)}")
                    
                    ax.set_title(f"{ticker} Price with Moving Averages")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating technical indicators chart: {str(e)}")
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
        if len(df) < 60:
            st.warning("Not enough data for prediction. Please select a longer time period.")
        else:
            try:
                # Progress bar for user feedback
                progress_bar = st.progress(0)
                
                # Create features (using last 30 days to predict next day)
                window_size = 30
                X = []
                y = []
                
                for i in range(window_size, len(df)):
                    X.append(df['Close'].iloc[i-window_size:i].values)
                    y.append(df['Close'].iloc[i])
                
                X = np.array(X)
                y = np.array(y)
                
                # Ensure we have enough data
                if len(X) == 0 or len(y) == 0:
                    st.warning("Not enough data points to create training sequences.")
                    if progress_bar:
                        progress_bar.empty()
                else:
                    progress_bar.progress(30)
                    
                    # Train-test split (80% train, 20% test)
                    split = int(0.8 * len(X))
                    if split == 0:
                        split = 1  # Ensure at least one training sample
                    
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]
                    
                    # Reshape X_train and X_test for LinearRegression
                    X_train_2d = X_train.reshape(X_train.shape[0], -1)
                    X_test_2d = X_test.reshape(X_test.shape[0], -1)
                    
                    # Create and train a linear regression model
                    with st.spinner("Training prediction model..."):
                        model = create_linear_model(X_train_2d, y_train)
                    
                    progress_bar.progress(60)
                    
                    # Make predictions
                    predictions = model.predict(X_test_2d)
                    
                    # Calculate RMSE
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    
                    progress_bar.progress(80)
                    
                    # Predict future price
                    last_window = df['Close'].iloc[-window_size:].values
                    last_window_2d = last_window.reshape(1, -1)  # Reshape for prediction
                    future_prediction = float(model.predict(last_window_2d)[0])
                    
                    # Create index for test data
                    test_index = df.index[split + window_size:]
                    
                    # Plot results
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Plot actual prices
                    ax.plot(test_index, y_test, 'b-', label='Actual')
                    
                    # Plot predicted prices
                    ax.plot(test_index, predictions, 'r--', label='Predicted')
                    
                    # Add alerts if any exist for this ticker
                    alerts = get_alerts()
                    if ticker in alerts and alerts[ticker]:
                        for alert in alerts[ticker]:
                            if alert['active']:
                                try:
                                    # Draw horizontal line for the alert
                                    ax.axhline(y=alert['price'], color='g', linestyle='-', alpha=0.5)
                                    # Add annotation
                                    ax.text(test_index[len(test_index) // 2], alert['price'], 
                                        f" {alert['type']} ${alert['price']:.2f}", 
                                        verticalalignment='bottom', color='green')
                                except Exception as e:
                                    st.warning(f"Could not display alert: {str(e)}")
                    
                    ax.set_title(f"{ticker} Price Prediction (RMSE: {rmse:.2f})")
                    ax.legend()
                    progress_bar.progress(100)
                    st.pyplot(fig)
                    
                    # Show prediction for next day
                    st.subheader("Price Prediction")
                    
                    last_actual_price = float(df['Close'].iloc[-1])
                    price_change = ((future_prediction - last_actual_price) / last_actual_price) * 100
                    
                    direction = "📈" if price_change > 0 else "📉"
                    st.write(f"### Next Trading Day Prediction: ${future_prediction:.2f} {direction}")
                    st.write(f"Expected change: {price_change:.2f}% from current price (${last_actual_price:.2f})")
                    
                    # Show comparison charts if other tickers exist
                    if len(valid_tickers) > 1:
                        st.subheader("Market Comparison")
                        try:
                            fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
                            
                            # Normalize all prices to same scale for better comparison
                            for t, df_t in valid_tickers.items():
                                # Convert to percentage change from first day
                                normalized = df_t['Close'] / float(df_t['Close'].iloc[0]) * 100
                                ax_comp.plot(df_t.index, normalized, label=f"{t} ({float(df_t['Close'].iloc[-1]):.2f})")
                                
                            ax_comp.set_title("Normalized Stock Price Comparison (Base=100)")
                            ax_comp.legend()
                            ax_comp.grid(True, alpha=0.3)
                            st.pyplot(fig_comp)
                        except Exception as e:
                            st.error(f"Error creating comparison chart: {str(e)}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Show latest prices in sidebar
with st.sidebar:
    st.header("Latest Prices")
    for t, df_t in valid_tickers.items():
        if not df_t.empty:
            try:
                # Get the latest price and ensure it's a float
                last_price = float(df_t['Close'].iloc[-1])
                
                # Calculate price change percentage
                if len(df_t) > 1:
                    prev_price = float(df_t['Close'].iloc[-2])
                    price_delta = f"{((last_price - prev_price) / prev_price) * 100:.2f}%"
                else:
                    price_delta = None
                
                # Check alerts for this ticker
                alerts = get_alerts()
                alert_triggered = False
                alert_message = ""
                
                if t in alerts:
                    for i, alert in enumerate(alerts[t]):
                        if alert['active'] and not alert['triggered']:
                            if (alert['type'] == 'Price Above' and last_price > alert['price']) or \
                               (alert['type'] == 'Price Below' and last_price < alert['price']):
                                alert_triggered = True
                                alert_message = f"⚠️ Alert: {alert['type']} ${alert['price']:.2f}"
                                # Mark alert as triggered
                                st.session_state.alerts[t][i]['triggered'] = True
                
                # Display price, with alert if triggered
                if alert_triggered:
                    st.warning(alert_message)
                    st.metric(label=t, value=f"${last_price:.2f}", delta=price_delta)
                else:
                    st.metric(label=t, value=f"${last_price:.2f}", delta=price_delta)
                    
            except Exception as e:
                st.error(f"Error displaying price for {t}: {str(e)}")

with tabs[3]:  # This is the new Forecast tab
    st.header("Long-Term Price Forecast")
    
    if ticker in valid_tickers:
        df = valid_tickers[ticker]
        
        # Add forecast period selection
        forecast_period = st.slider(
            "Forecast Period (Days)", 
            min_value=7, 
            max_value=90, 
            value=30,
            step=7
        )
        
        forecast_button = st.button("Generate Forecast", key="forecast_btn")
        
        if forecast_button:
            with st.spinner("Calculating forecasts..."):
                # Create forecast models
                forecast_data = create_forecast_models(df, forecast_days=forecast_period)
                
                if forecast_data and 'forecasts' in forecast_data and len(forecast_data['forecasts']) > 0:
                    # Create interactive Plotly chart
                    fig = make_subplots(
                        rows=2, 
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("Price Forecast", "Forecast Comparison"),
                        row_heights=[0.7, 0.3]
                    )
                    
                    # Add historical data
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_data['actual_dates'][-90:],  # Show last 90 days
                            y=forecast_data['actual_prices'][-90:],
                            mode='lines',
                            name='Historical',
                            line=dict(color='black')
                        ),
                        row=1, col=1
                    )
                    
                    # Add forecast traces with confidence intervals
                    colors = {
                        'ARIMA': 'blue',
                        'Exponential': 'red',
                        'Linear': 'green'
                    }
                    
                    # Add each forecast model's prediction
                    for model_name, forecast_values in forecast_data['forecasts'].items():
                        # Add main forecast line
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_data['future_dates'],
                                y=forecast_values,
                                mode='lines',
                                name=f"{model_name} Forecast",
                                line=dict(color=colors.get(model_name, 'orange')),
                            ),
                            row=1, col=1
                        )
                        
                        # Add percentage difference from last price (bottom panel)
                        pct_diff = [(x - forecast_data['last_price']) / forecast_data['last_price'] * 100 
                                    for x in forecast_values]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_data['future_dates'],
                                y=pct_diff,
                                mode='lines',
                                name=f"{model_name} % Change",
                                line=dict(color=colors.get(model_name, 'orange')),
                            ),
                            row=2, col=1
                        )
                    
                    # Add zero line for percentage plot
                    fig.add_shape(
                        type="line",
                        x0=forecast_data['future_dates'][0],
                        y0=0,
                        x1=forecast_data['future_dates'][-1],
                        y1=0,
                        line=dict(color="gray", width=1, dash="dash"),
                        row=2, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{ticker} {forecast_period}-Day Price Forecast",
                        height=700,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        hovermode="x unified"
                    )
                    
                    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                    fig.update_yaxes(title_text="% Change", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display forecast summary table
                    st.subheader("Forecast Summary")
                    
                    # Create a summary table
                    forecast_summary = {
                        'Date': forecast_data['future_dates'].strftime('%Y-%m-%d'),
                    }
                    
                    for model_name, forecast_values in forecast_data['forecasts'].items():
                        forecast_summary[f'{model_name}'] = forecast_values.round(2)
                        
                    # Create DataFrame and display
                    summary_df = pd.DataFrame(forecast_summary)
                    
                    # Show only selected rows (first day, week, month end)
                    display_indices = [0, 6, 13, 29] if forecast_period >= 30 else [0, 6, forecast_period-1]
                    display_indices = [i for i in display_indices if i < forecast_period]
                    
                    st.dataframe(summary_df.iloc[display_indices])
                    
                    # Add download button for full forecast data
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast Data",
                        data=csv,
                        file_name=f"{ticker}_forecast_{forecast_period}days.csv",
                        mime="text/csv",
                    )
                    
                    # Add forecast analytics
                    st.subheader("Forecast Analytics")
                    
                    # Calculate agreement between models
                    last_price = forecast_data['last_price']
                    end_forecasts = {model: values[-1] for model, values in forecast_data['forecasts'].items()}
                    
                    bullish_count = sum(1 for value in end_forecasts.values() if value > last_price)
                    bearish_count = sum(1 for value in end_forecasts.values() if value < last_price)
                    
                    # Display sentiment gauges as columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sentiment = "Bullish" if bullish_count > bearish_count else "Bearish" if bearish_count > bullish_count else "Neutral"
                        color = "green" if sentiment == "Bullish" else "red" if sentiment == "Bearish" else "gray"
                        st.markdown(f"<h3 style='text-align: center; color: {color};'>{sentiment}</h3>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align: center;'>Overall Forecast Sentiment</p>", unsafe_allow_html=True)
                    
                    with col2:
                        avg_end_price = sum(end_forecasts.values()) / len(end_forecasts)
                        pct_change = (avg_end_price - last_price) / last_price * 100
                        color = "green" if pct_change > 0 else "red"
                        st.markdown(f"<h3 style='text-align: center; color: {color};'>{pct_change:.2f}%</h3>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align: center;'>Average Projected Change</p>", unsafe_allow_html=True)
                    
                    with col3:
                        # Calculate average price at forecast period/2 (middle point)
                        mid_point = forecast_period // 2
                        mid_forecasts = {model: values[mid_point] for model, values in forecast_data['forecasts'].items()}
                        mid_avg = sum(mid_forecasts.values()) / len(mid_forecasts)
                        end_avg = sum(end_forecasts.values()) / len(end_forecasts)
                        
                        # Determine if trend is accelerating or decelerating
                        mid_pct = (mid_avg - last_price) / last_price
                        end_pct = (end_avg - last_price) / last_price
                        trend_speed = abs(end_pct) > abs(mid_pct * 2)
                        
                        if end_pct > 0:
                            trend_text = "Accelerating Up" if trend_speed else "Steady Up"
                            color = "green"
                        else:
                            trend_text = "Accelerating Down" if trend_speed else "Steady Down"
                            color = "red"
                            
                        st.markdown(f"<h3 style='text-align: center; color: {color};'>{trend_text}</h3>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align: center;'>Trend Velocity</p>", unsafe_allow_html=True)
                    
                    # Add explanatory text
                    st.markdown("""
                    **Understanding the Forecast:**
                    - The forecast combines multiple statistical models (ARIMA, Exponential Smoothing, and Linear Trend)
                    - Each model has different strengths and may perform better in different market conditions
                    - The percentage change chart shows projected movement from the last closing price
                    - For longer-term investment decisions, consider using longer forecast periods
                    """)
                    
                else:
                    st.error("Unable to generate forecast. Please try a different ticker or time period.")
    else:
        st.warning(f"No data available for {ticker}. Please enter a valid ticker symbol.")

     
    # Add stock information websites
    st.sidebar.header("Stock Resources")
    st.sidebar.markdown("""
    - [Yahoo Finance](https://finance.yahoo.com/)
    - [CNBC](https://www.cnbc.com/stock-markets/)
    - [MarketWatch](https://www.marketwatch.com/)
    - [Investing.com](https://www.investing.com/)
    """)

# Calculate and show elapsed time properly
elapsed_time = time.time() - start_time
st.sidebar.info(f"Data loaded in {elapsed_time:.2f} seconds")
