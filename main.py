import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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

# Button to trigger prediction
if st.sidebar.button("Predict"):
    df = yf.download(stock_symbol, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found for this stock.")
    else:
        st.success("Data loaded successfully!")
        df = df[['Close']]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

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

        # Show some metrics
        st.subheader("📊 Stock Overview")
        col1, col2 = st.columns(2)
        col1.metric("Latest Close Price", f"${df['Close'][-1]:.2f}")
        col2.metric("Data Points", len(df))

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

