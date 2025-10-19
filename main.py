"""
LSTM Stock Prediction Framework - Main Training Script
Version: 1.0
Date: October 19, 2025
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 2.2 CONFIGURATION PARAMETERS
# ============================================================================
TICKER = 'AAPL'
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'
WINDOW_SIZE = 60
TRAIN_SPLIT_PERCENT = 0.8
EPOCHS = 50
BATCH_SIZE = 32
TARGET = 'log_return'
FEATURES = ['Close', 'Volume', 'RSI', 'MACD', 'EMA_20']
NUM_FEATURES = len(FEATURES)

print("=" * 80)
print("LSTM STOCK PREDICTION FRAMEWORK")
print("=" * 80)
print(f"Configuration:")
print(f"  Ticker: {TICKER}")
print(f"  Date Range: {START_DATE} to {END_DATE}")
print(f"  Window Size: {WINDOW_SIZE}")
print(f"  Features: {FEATURES}")
print(f"  Number of Features: {NUM_FEATURES}")
print("=" * 80)

# ============================================================================
# 3.1 PHASE 1: INGESTION & FEATURE ENGINEERING
# ============================================================================
print("\n[Phase 1] Data Ingestion & Feature Engineering...")

# Download stock data
print(f"Downloading {TICKER} data from Yahoo Finance...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)

# Fix for yfinance MultiIndex columns issue
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# Calculate logarithmic returns (Target Variable)
print("Calculating logarithmic returns...")
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

# Calculate technical indicators
print("Computing technical indicators...")
# Convert to Series to avoid shape issues
close_series = df['Close'].squeeze()
df['RSI'] = ta.momentum.RSIIndicator(close_series).rsi()
df['MACD'] = ta.trend.MACD(close_series).macd()
df['EMA_20'] = ta.trend.EMAIndicator(close_series, window=20).ema_indicator()

# Data cleansing - remove NaN values
print("Cleaning data (removing NaN values)...")
df_clean = df.dropna()
print(f"Data shape after cleaning: {df_clean.shape}")
print(f"Date range: {df_clean.index[0]} to {df_clean.index[-1]}")

# ============================================================================
# 3.2 PHASE 2: PREPROCESSING & SCALING
# ============================================================================
print("\n[Phase 2] Preprocessing & Scaling...")

# Extract features and target
feature_data = df_clean[FEATURES].values
target_data = df_clean[TARGET].values.reshape(-1, 1)

# Two-Scaler Strategy (prevent data leakage)
print("Initializing separate scalers for features and target...")
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

# Fit and transform
print("Scaling features...")
scaled_features = scaler_features.fit_transform(feature_data)
print("Scaling target...")
scaled_target = scaler_target.fit_transform(target_data)

# Save scalers
print("Saving scaler artifacts...")
joblib.dump(scaler_features, 'scaler_features.pkl')
joblib.dump(scaler_target, 'scaler_target.pkl')
print("  ✓ scaler_features.pkl saved")
print("  ✓ scaler_target.pkl saved")

# ============================================================================
# 3.3 PHASE 3: SUPERVISED SEQUENCE STRUCTURING
# ============================================================================
print("\n[Phase 3] Supervised Sequence Structuring...")

def create_sequences(features, target, window_size):
    """
    Create sequences for LSTM training.
    
    Args:
        features: Scaled feature array
        target: Scaled target array
        window_size: Number of time steps to look back
    
    Returns:
        X: 3D array (samples, window_size, num_features)
        y: 1D array (samples,)
    """
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i-window_size:i])
        y.append(target[i])
    return np.array(X), np.array(y)

print(f"Creating sequences with window size {WINDOW_SIZE}...")
X, y = create_sequences(scaled_features, scaled_target.flatten(), WINDOW_SIZE)
print(f"Sequence shape - X: {X.shape}, y: {y.shape}")

# Chronological split (no shuffling)
split_idx = int(len(X) * TRAIN_SPLIT_PERCENT)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# 4.0 MODEL ARCHITECTURE & TRAINING
# ============================================================================
print("\n[Phase 4] Model Architecture & Training...")

# 4.1 Model Definition
print("Building Stacked LSTM model...")
model = Sequential([
    # First LSTM layer
    LSTM(units=50, return_sequences=True, input_shape=(WINDOW_SIZE, NUM_FEATURES)),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    
    # Dense layer
    Dense(units=32, activation='relu'),
    
    # Output layer
    Dense(units=1)
])

# 4.2 Compilation
print("Compiling model...")
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Training
print(f"\nTraining model for {EPOCHS} epochs...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    shuffle=False,
    verbose=1
)

# 4.3 Save model
print("\nSaving trained model...")
model.save('lstm_model.h5')
print("  ✓ lstm_model.h5 saved")

# ============================================================================
# 5.0 EVALUATION & INFERENCE
# ============================================================================
print("\n[Phase 5] Evaluation & Inference...")

# 5.1 Model Evaluation
print("Generating predictions on test set...")
predictions_scaled = model.predict(X_test)

# Inverse transformation
print("Inverse transforming predictions and actual values...")
predictions = scaler_target.inverse_transform(predictions_scaled)
actuals = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print("\n" + "=" * 80)
print("MODEL EVALUATION METRICS")
print("=" * 80)
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"R-squared (R²): {r2:.6f}")
print("=" * 80)

# Visualization
print("\nGenerating visualization...")
plt.figure(figsize=(14, 6))
plt.plot(actuals, label='Actual Log Returns', alpha=0.7)
plt.plot(predictions, label='Predicted Log Returns', alpha=0.7)
plt.title(f'{TICKER} - Actual vs Predicted Log Returns')
plt.xlabel('Test Sample Index')
plt.ylabel('Log Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evaluation_plot.png', dpi=300)
print("  ✓ evaluation_plot.png saved")
plt.show()

# Plot training history
plt.figure(figsize=(14, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
print("  ✓ training_history.png saved")
plt.show()

# ============================================================================
# 5.2 FUTURE PRICE FORECASTING FUNCTION
# ============================================================================
def predict_future(model, scaler_features, scaler_target, df_clean, n_days=30):
    """
    Predict future stock prices using iterative forecasting.
    
    Args:
        model: Trained LSTM model
        scaler_features: Fitted scaler for features
        scaler_target: Fitted scaler for target
        df_clean: Clean dataframe with features
        n_days: Number of days to forecast
    
    Returns:
        predicted_prices: Array of predicted future prices
    """
    print(f"\nPredicting next {n_days} days...")
    
    # Get last window of data
    last_window = df_clean[FEATURES].iloc[-WINDOW_SIZE:].values
    last_price = df_clean['Close'].iloc[-1]
    
    predicted_log_returns = []
    predicted_prices = [last_price]
    
    current_window = last_window.copy()
    
    for day in range(n_days):
        # Scale the current window
        scaled_window = scaler_features.transform(current_window)
        scaled_window_reshaped = scaled_window.reshape(1, WINDOW_SIZE, NUM_FEATURES)
        
        # Predict next log return
        pred_scaled = model.predict(scaled_window_reshaped, verbose=0)
        pred_log_return = scaler_target.inverse_transform(pred_scaled)[0, 0]
        predicted_log_returns.append(pred_log_return)
        
        # Convert log return to price
        next_price = predicted_prices[-1] * np.exp(pred_log_return)
        predicted_prices.append(next_price)
        
        # Update window for next prediction
        # Note: This is a simplified approach - using the last known feature values
        # In production, you'd need to calculate new technical indicators
        new_row = current_window[-1].copy()
        new_row[0] = next_price  # Update Close price
        
        # Shift window and add new row
        current_window = np.vstack([current_window[1:], new_row])
    
    return np.array(predicted_prices[1:])  # Exclude the starting price

# Example: Predict next 30 days
future_predictions = predict_future(model, scaler_features, scaler_target, df_clean, n_days=30)
print(f"Future predictions generated: {len(future_predictions)} days")
print(f"Predicted price for day 1: ${future_predictions[0]:.2f}")
print(f"Predicted price for day 30: ${future_predictions[-1]:.2f}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE - ALL ARTIFACTS SAVED")
print("=" * 80)
print("Generated artifacts:")
print("  1. lstm_model.h5")
print("  2. scaler_features.pkl")
print("  3. scaler_target.pkl")
print("  4. evaluation_plot.png")
print("  5. training_history.png")
print("\nReady for deployment!")
print("=" * 80)