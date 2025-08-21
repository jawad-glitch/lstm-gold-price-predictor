# lstm_gld_predictor.py

import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import os

# CONFIGURATION
STOCK = "GLD"
INTERVAL = "15m"
LOOK_BACK = 30  # Number of past intervals (15-min) used for prediction
DAYS = 5  # Number of past days of data to fetch
MODEL_PATH = "lstm_realtime_model.keras"

# FETCH DATA
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=DAYS)

print(f"Fetching {STOCK} 15-min interval data from {start_date.date()} to {end_date.date()}...")
df = yf.download(
    STOCK,
    interval=INTERVAL,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    progress=False
)

if df.empty:
    raise Exception(" No data fetched from yfinance.")

# Use 'Close' prices only
data = df['Close'].values.reshape(-1, 1)

#  SCALE DATA
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# PREPARE SEQUENCES
X, y = [], []
for i in range(LOOK_BACK, len(scaled_data)):
    X.append(scaled_data[i - LOOK_BACK:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# LOAD OR CREATE MODEL
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(" Loaded existing model.")
else:
    print("️ Creating new model...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
# TRAIN MODEL
print(" Training model...")
model.fit(X, y, epochs=10, batch_size=32, verbose=1, validation_split=0.1)

# SAVE MODEL
model.save(MODEL_PATH)
print(f" Model saved to: {MODEL_PATH}")

# MAKE PREDICTION
last_sequence = scaled_data[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
predicted_scaled = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

print(f"Predicted next 15-min close price: ${predicted_price:.2f}")
print(f"️ Script started at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
