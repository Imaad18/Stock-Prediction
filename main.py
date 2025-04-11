import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


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
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

# Button to trigger prediction
if st.sidebar.button("Predict"):
    # Fetch stock data
    df = fetch_live_data(stock_symbol)
    
    if df.empty:
        st.error("No data found for this stock.")
    else:
        st.success("Data loaded successfully!")
        df = df[['Close']]  # Use only the 'Close' price

        # Calculate moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA100'] = df['Close'].rolling(window=100).mean()

        # Show metrics
        st.subheader("📊 Stock Overview")
        col1, col2 = st.columns(2)
        col1.metric("Latest Close Price", f"${df['Close'][-1]:.2f}")
        col2.metric("Data Points", len(df))

        # Preprocessing for LSTM model
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['Close']])

        # Create sequences
        seq_len = 60
        X, y = [], []
        for i in range(seq_len, len(scaled_data)):
            X.append(scaled_data[i - seq_len:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Split into train and test
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




          # Train the model
        with st.spinner("Training the model..."):
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predict and scale back
        predicted = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plotting
        st.subheader(f"📉 {stock_symbol} Actual vs Predicted Price")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(real_prices.flatten(), label="Actual Price", color="blue")
        ax.plot(predicted_prices.flatten(), label="Predicted Price", color="red")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        ax.set_title(f"Stock Price Prediction for {stock_symbol}")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)

        st.caption("Built with LSTM model using Keras and Streamlit.")
