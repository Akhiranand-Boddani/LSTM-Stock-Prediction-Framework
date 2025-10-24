"""
FINAL OPTIMIZED LSTM Stock Prediction Framework
Version: 3.0 - Maximum Performance with Variance Preservation
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# OPTIMIZED CONFIGURATION
# ============================================================================
TICKER = 'AAPL'
START_DATE = '2010-01-01'  # More data!
END_DATE = '2024-12-31'
WINDOW_SIZE = 30  # Shorter window for more recent patterns
TRAIN_SPLIT_PERCENT = 0.85  # More training data
EPOCHS = 150
BATCH_SIZE = 16  # Smaller batches
TARGET = 'returns'  # Regular returns instead of log returns

# Comprehensive feature set
FEATURES = [
    'Close', 'Volume', 'High', 'Low',
    'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
    'EMA_12', 'EMA_26', 'EMA_50',
    'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
    'ATR', 'OBV',
    'Stoch_K', 'Stoch_D',
    'ROC'  # Rate of Change
]
NUM_FEATURES = len(FEATURES)

print("=" * 80)
print("FINAL OPTIMIZED LSTM STOCK PREDICTION")
print("=" * 80)
print(f"Ticker: {TICKER}")
print(f"Date Range: {START_DATE} to {END_DATE}")
print(f"Window Size: {WINDOW_SIZE}")
print(f"Features: {NUM_FEATURES}")
print(f"Target: {TARGET}")
print("=" * 80)

# ============================================================================
# DATA PIPELINE WITH COMPREHENSIVE FEATURES
# ============================================================================
print("\n[Phase 1] Downloading and Processing Data...")

df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# Use simple returns instead of log returns (better for neural networks)
df['returns'] = df['Close'].pct_change()

# Extract series
close = df['Close'].squeeze()
high = df['High'].squeeze()
low = df['Low'].squeeze()
volume = df['Volume'].squeeze()

print("Computing comprehensive technical indicators...")

# Momentum Indicators
df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
df['Stoch_K'] = ta.momentum.StochasticOscillator(high, low, close).stoch()
df['Stoch_D'] = ta.momentum.StochasticOscillator(high, low, close).stoch_signal()
df['ROC'] = ta.momentum.ROCIndicator(close, window=12).roc()

# Trend Indicators
macd = ta.trend.MACD(close)
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_diff'] = macd.macd_diff()
df['EMA_12'] = ta.trend.EMAIndicator(close, window=12).ema_indicator()
df['EMA_26'] = ta.trend.EMAIndicator(close, window=26).ema_indicator()
df['EMA_50'] = ta.trend.EMAIndicator(close, window=50).ema_indicator()

# Volatility Indicators
bollinger = ta.volatility.BollingerBands(close)
df['BB_upper'] = bollinger.bollinger_hband()
df['BB_middle'] = bollinger.bollinger_mavg()
df['BB_lower'] = bollinger.bollinger_lband()
df['BB_width'] = bollinger.bollinger_wband()
df['ATR'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

# Volume Indicators
df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

# Clean data
df_clean = df.dropna()
print(f"✓ Data cleaned: {len(df_clean)} samples")

# ============================================================================
# ADVANCED PREPROCESSING
# ============================================================================
print("\n[Phase 2] Advanced Preprocessing...")

feature_data = df_clean[FEATURES].values
target_data = df_clean[TARGET].values.reshape(-1, 1)

# Use RobustScaler (better handles outliers in financial data)
scaler_features = RobustScaler()
scaler_target = StandardScaler()

scaled_features = scaler_features.fit_transform(feature_data)
scaled_target = scaler_target.fit_transform(target_data)

# Save scalers
joblib.dump(scaler_features, 'scaler_features_final.pkl')
joblib.dump(scaler_target, 'scaler_target_final.pkl')
print("✓ Scalers saved")

# ============================================================================
# SEQUENCE GENERATION
# ============================================================================
print("\n[Phase 3] Creating Sequences...")

def create_sequences(features, target, window_size):
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i-window_size:i])
        y.append(target[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, scaled_target.flatten(), WINDOW_SIZE)

# Split chronologically
split_idx = int(len(X) * TRAIN_SPLIT_PERCENT)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

# ============================================================================
# ADVANCED MODEL ARCHITECTURE
# ============================================================================
print("\n[Phase 4] Building Advanced Model Architecture...")

# Custom loss to preserve variance
def custom_mse_with_variance_penalty(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Variance penalty: penalize if predicted variance is too small
    true_var = tf.math.reduce_variance(y_true)
    pred_var = tf.math.reduce_variance(y_pred)
    var_penalty = tf.square(true_var - pred_var)
    
    return mse + 0.1 * var_penalty

model = Sequential([
    # First Bidirectional GRU layer (often better than LSTM for financial data)
    Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(0.001)),
                  input_shape=(WINDOW_SIZE, NUM_FEATURES)),
    LayerNormalization(),
    Dropout(0.4),
    
    # Second Bidirectional GRU layer
    Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(0.001))),
    LayerNormalization(),
    Dropout(0.4),
    
    # Third GRU layer
    GRU(64, return_sequences=False, kernel_regularizer=l2(0.001)),
    LayerNormalization(),
    Dropout(0.3),
    
    # Dense layers with L2 regularization
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    
    # Output layer
    Dense(1, activation='linear')
])

# Compile with custom loss
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss=custom_mse_with_variance_penalty,
    metrics=['mae', 'mse']
)

print("\n📋 Model Summary:")
model.summary()
print(f"\nTotal parameters: {model.count_params():,}")

# ============================================================================
# TRAINING WITH ADVANCED CALLBACKS
# ============================================================================
print("\n[Phase 5] Training with Advanced Callbacks...")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

# Custom callback to monitor variance
class VarianceMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            val_pred = self.model.predict(X_test, verbose=0)
            pred_std = np.std(val_pred)
            actual_std = np.std(y_test)
            print(f"\n  Variance Check - Pred Std: {pred_std:.6f}, Actual Std: {actual_std:.6f}")

variance_monitor = VarianceMonitor()

print(f"Training for up to {EPOCHS} epochs...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr, variance_monitor],
    shuffle=False,
    verbose=1
)

# Save model
model.save('lstm_model_final.h5')
print("\n✓ lstm_model_final.h5 saved")

# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================
print("\n[Phase 6] Final Model Evaluation...")

predictions_scaled = model.predict(X_test, verbose=0)
predictions = scaler_target.inverse_transform(predictions_scaled)
actuals = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate all metrics
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

# Directional accuracy
pred_direction = np.sign(predictions.flatten())
actual_direction = np.sign(actuals.flatten())
directional_accuracy = np.mean(pred_direction == actual_direction) * 100

# Correlation
correlation = np.corrcoef(actuals.flatten(), predictions.flatten())[0, 1]

# Variance ratio (should be close to 1.0)
variance_ratio = np.std(predictions) / np.std(actuals)

print("\n" + "=" * 80)
print("FINAL MODEL EVALUATION METRICS")
print("=" * 80)
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R-squared (R²): {r2:.6f}")
print(f"Directional Accuracy: {directional_accuracy:.2f}%")
print(f"Correlation: {correlation:.6f}")
print(f"Variance Ratio (Pred/Actual): {variance_ratio:.6f}")
print("=" * 80)

print("\nDetailed Statistics:")
print(f"  Mean Prediction: {predictions.mean():.6f}")
print(f"  Mean Actual: {actuals.mean():.6f}")
print(f"  Std Prediction: {predictions.std():.6f}")
print(f"  Std Actual: {actuals.std():.6f}")
print(f"  Min Prediction: {predictions.min():.6f}")
print(f"  Max Prediction: {predictions.max():.6f}")
print(f"  Min Actual: {actuals.min():.6f}")
print(f"  Max Actual: {actuals.max():.6f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[Phase 7] Generating Comprehensive Visualizations...")

# Plot 1: Training History
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(history.history['mse'], label='Training MSE', linewidth=2)
axes[2].plot(history.history['val_mse'], label='Validation MSE', linewidth=2)
axes[2].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('MSE')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_final.png', dpi=300, bbox_inches='tight')
print("✓ training_history_final.png saved")
plt.show()

# Plot 2: Predictions Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Time series comparison
axes[0, 0].plot(actuals, label='Actual Returns', alpha=0.7, linewidth=1.5, color='blue')
axes[0, 0].plot(predictions, label='Predicted Returns', alpha=0.7, linewidth=1.5, color='red')
axes[0, 0].set_title('Actual vs Predicted Returns', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Test Sample')
axes[0, 0].set_ylabel('Returns')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Scatter plot
axes[0, 1].scatter(actuals, predictions, alpha=0.4, s=15)
axes[0, 1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_title(f'Scatter Plot (R² = {r2:.4f})', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Actual Returns')
axes[0, 1].set_ylabel('Predicted Returns')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Residuals
residuals = actuals.flatten() - predictions.flatten()
axes[1, 0].plot(residuals, alpha=0.7, color='purple')
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_title('Residuals (Actual - Predicted)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Test Sample')
axes[1, 0].set_ylabel('Residual')
axes[1, 0].grid(True, alpha=0.3)

# Distribution comparison
axes[1, 1].hist(actuals.flatten(), bins=50, alpha=0.5, label='Actual', color='blue', density=True)
axes[1, 1].hist(predictions.flatten(), bins=50, alpha=0.5, label='Predicted', color='red', density=True)
axes[1, 1].set_title('Distribution Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Returns')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_final.png', dpi=300, bbox_inches='tight')
print("✓ evaluation_final.png saved")
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("🎉 FINAL MODEL TRAINING COMPLETE")
print("=" * 80)
print("\nGenerated Artifacts:")
print("  1. lstm_model_final.h5")
print("  2. scaler_features_final.pkl")
print("  3. scaler_target_final.pkl")
print("  4. training_history_final.png")
print("  5. evaluation_final.png")

print("\n📊 Final Performance Summary:")
print(f"  • R² Score: {r2:.4f}")
print(f"  • RMSE: {rmse:.6f}")
print(f"  • MAE: {mae:.6f}")
print(f"  • Directional Accuracy: {directional_accuracy:.2f}%")
print(f"  • Correlation: {correlation:.4f}")
print(f"  • Variance Ratio: {variance_ratio:.4f}")

print("\n💡 Interpretation:")
if r2 > 0.3:
    print("  ✓ Excellent R² for stock prediction!")
elif r2 > 0.1:
    print("  ✓ Good R² for stock prediction!")
elif r2 > 0:
    print("  ✓ Positive R² - model is learning patterns")
else:
    print("  ⚠ R² still negative, but check directional accuracy")

if directional_accuracy > 55:
    print(f"  ✓ Directional accuracy {directional_accuracy:.1f}% is profitable!")
elif directional_accuracy > 52:
    print(f"  ✓ Directional accuracy {directional_accuracy:.1f}% is above random")
else:
    print(f"  ⚠ Directional accuracy needs improvement")

if 0.7 < variance_ratio < 1.3:
    print(f"  ✓ Variance ratio {variance_ratio:.2f} is good - predictions have appropriate spread")
else:
    print(f"  ⚠ Variance ratio {variance_ratio:.2f} - predictions may be too {'constant' if variance_ratio < 0.7 else 'volatile'}")

print("=" * 80)