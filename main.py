
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

st.set_page_config(layout='wide')
st.title("📈 Stock Price Predictor with LSTM")

# Sidebar for input
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

if st.sidebar.button("Predict"):
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    if df.empty:
        st.error("No data found for the selected stock.")
    else:
        df = df[['Close']]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        seq_len = 60
        X, y = [], []
        for i in range(seq_len, len(scaled_data)):
            X.append(scaled_data[i - seq_len:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Build LSTM model
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

        # Predict
        predicted = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(real_prices, label="Actual Price", color='blue')
        ax.plot(predicted_prices, label="Predicted Price", color='red')
        ax.set_title(f"{stock_symbol} - Actual vs Predicted Price")
        ax.set_xlabel("Time")
        ax.set_ylabel("Stock Price")
        ax.legend()
        st.pyplot(fig)
