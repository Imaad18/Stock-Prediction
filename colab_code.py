# Import Required libraries(part1)

import yfinance as yf    # To fetch Stock Data
import numpy as np       # For Data Manipulation
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns   # Same as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error



# Fetch Data(part2)

ticker = "AAPL"
df = yf.download(ticker, start="2020-01-01", end="2024-12-31")
df = df[["Close"]]



# Normalize Data(part3)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Create Sequence for LSTM(part4)

sequence_length = 60
x,y = [],[]

for i in range(sequence_length, len(scaled_data)):
  x.append(scaled_data[i-sequence_length:i, 0])
  y.append(scaled_data[i, 0])


x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))  



# Train_test split Data(part5)

split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]



# Build LSTM Model(part6)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1))) # Change [1] to 1
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")

history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))


# Predictions(part7)


predicted = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted.reshape(-1,1))
real_prices = scaler.inverse_transform(y_test.reshape(-1,1))


# Plot(part8)

plt.figure(figsize=(14, 6))
plt.plot(real_prices, color='blue', label='Actual Price')
plt.plot(predicted_prices, color='red', label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()



# Evaluate Performance(part9)

rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
print(f"Root Mean Squared Error: {rmse:.2f}")

