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

# Enable caching for expensive operations
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, start_date, end_date):
    try:
        # Add 1 day to end_date to ensure we get the most recent data
        df = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), progress=False)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to create a linear regression model
def create_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to calculate technical indicators manually
def calculate_indicators(df):
    try:
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
    if len(df) < 100:
        return None, None
    
    patterns = {}
    
    try:
        # Calculate indicators
        df_analysis = calculate_indicators(df)
        
        # Drop NaN values
        df_analysis = df_analysis.dropna()
        
        if len(df_analysis) < 2:
            return None, None
        
        # Detect basic patterns
        
        # Moving average crossovers - ensure we're comparing scalar values
        if 'SMA20' in df_analysis.columns and 'SMA50' in df_analysis.columns:
            # Get scalar values instead of Series objects
            sma20_prev = df_analysis['SMA20'].iloc[-2]
            sma50_prev = df_analysis['SMA50'].iloc[-2]
            sma20_curr = df_analysis['SMA20'].iloc[-1]
            sma50_curr = df_analysis['SMA50'].iloc[-1]
            
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
    except Exception as e:
        st.error(f"Error detecting patterns: {str(e)}")
        return {}, df  # Return empty patterns if detection fails

# Prediction tab content - Fix for LinearRegression reshaping
# Add this to the relevant part of your tabs[0] section:
def create_prediction_model(df):
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
        return None, None, None, None
    
    # Train-test split (80% train, 20% test)
    split = int(0.8 * len(X))
    if split == 0:
        split = 1  # Ensure at least one training sample
    
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Important: Reshape for LinearRegression
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train_2d, y_train)
    
    # Make predictions
    predictions = model.predict(X_test_2d)
    
    return model, X_test_2d, y_test, predictions, split, window_size

# Fix for sidebar latest prices display
def display_latest_prices(valid_tickers):
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
    
        
    
    # Add stock information websites
    st.sidebar.header("Stock Resources")
    st.sidebar.markdown("""
    - [Yahoo Finance](https://finance.yahoo.com/)
    - [CNBC](https://www.cnbc.com/stock-markets/)
    - [MarketWatch](https://www.marketwatch.com/)
    - [Investing.com](https://www.investing.com/)
    """)

st.sidebar.info(f"Data loaded in {time.time()-start_time:.2f} seconds")
