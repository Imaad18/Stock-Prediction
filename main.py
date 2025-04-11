import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import time

# Streamlit page config
st.set_page_config(layout="wide", page_title="Stock Price Predictor", page_icon="📈")
st.title("📈 Stock Price Predictor with LSTM")

# Sidebar inputs
st.sidebar.header("Input Parameters")
stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Function to fetch live data
def fetch_live_data(symbol):
    live_data = yf.download(symbol, period="1d", interval="1m")
    return live_data

# Button to trigger prediction
if st.sidebar.button("Predict"):
    # Set auto-refresh every 60 seconds (to simulate live updates)
    st.autorefresh(interval=60 * 1000)  # 60 seconds refresh

    # Fetch live stock data
    df = fetch_live_data(stock_symbol)
    
    if df.empty:
        st.error("No data found for this stock.")
    else:
        st.success("Live data loaded successfully!")
        df = df[['Close']]
        
        # Calculate moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA100'] = df['Close'].rolling(window=100).mean()

        # Get last valid values of MAs
        latest_ma20 = df['MA20'].iloc[-1]
        latest_ma100 = df['MA100'].iloc[-1]

        # Decision Logic
        if latest_ma20 > latest_ma100:
            suggestion = "📈 **Recommendation: BUY** - Short-term trend is stronger than long-term."
        elif latest_ma20 < latest_ma100:
            suggestion = "🔻 **Recommendation: SELL** - Short-term weakness against long-term trend."
        else:
            suggestion = "⏸️ **Recommendation: HOLD** - No clear signal at the moment."

        # Show some metrics
        st.subheader("📊 Stock Overview")
        col1, col2 = st.columns(2)
        col1.metric("Latest Close Price", f"${df['Close'][-1]:.2f}")
        col2.metric("Data Points", len(df))

        # Display recommendation
        st.subheader("🤖 AI-Based Trading Suggestion")
        st.markdown(suggestion)

        # Preprocessing for LSTM model
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Close']])

        # Create sequences
        seq_len = 60
        X, y = [], []
        for i in range(seq_len, len(scaled_data)):
            X.append(scaled_data[i - seq_len:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Split
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # LSTM Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train
        with st.spinner("Training the model..."):
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predict and scale back
        predicted = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plotly Interactive Chart
        st.subheader(f"📉 {stock_symbol} Actual vs Predicted Price")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=real_prices.flatten(), name='Actual Price', mode='lines', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=predicted_prices.flatten(), name='Predicted Price', mode='lines', line=dict(color='red')))
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            legend=dict(x=0, y=1.0),
            margin=dict(l=40, r=40, t=40, b=40),
            height=500,
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Built with LSTM model using Keras and Streamlit.")

