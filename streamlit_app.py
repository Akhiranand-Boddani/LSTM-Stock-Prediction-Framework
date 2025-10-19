"""
LSTM Stock Prediction Dashboard
Streamlit Deployment Application
Version: 1.0
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="LSTM Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# ============================================================================
# CONFIGURATION
# ============================================================================
WINDOW_SIZE = 60
FEATURES = ['Close', 'Volume', 'RSI', 'MACD', 'EMA_20']
NUM_FEATURES = len(FEATURES)

# ============================================================================
# LOAD ARTIFACTS
# ============================================================================
@st.cache_resource
def load_model_and_scalers():
    """Load the trained model and scalers from disk."""
    try:
        model = keras.models.load_model('lstm_model.h5')
        scaler_features = joblib.load('scaler_features.pkl')
        scaler_target = joblib.load('scaler_target.pkl')
        return model, scaler_features, scaler_target, None
    except Exception as e:
        return None, None, None, str(e)

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================
def download_and_process_data(ticker, start_date, end_date):
    """Download stock data and compute technical indicators."""
    try:
        # Download data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return None, "No data available for this ticker."
        
        # Fix for yfinance MultiIndex columns issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Calculate logarithmic returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Calculate technical indicators
        # Convert to Series to avoid shape issues
        close_series = df['Close'].squeeze()
        df['RSI'] = ta.momentum.RSIIndicator(close_series).rsi()
        df['MACD'] = ta.trend.MACD(close_series).macd()
        df['EMA_20'] = ta.trend.EMAIndicator(close_series, window=20).ema_indicator()
        
        # Clean data
        df_clean = df.dropna()
        
        return df_clean, None
    except Exception as e:
        return None, str(e)

def predict_future_prices(model, scaler_features, scaler_target, df_clean, n_days):
    """
    Predict future stock prices using iterative forecasting.
    """
    # Get last window of data
    last_window = df_clean[FEATURES].iloc[-WINDOW_SIZE:].values
    last_price = df_clean['Close'].iloc[-1]
    last_date = df_clean.index[-1]
    
    predicted_prices = [last_price]
    predicted_dates = [last_date]
    
    current_window = last_window.copy()
    
    for day in range(n_days):
        # Scale the current window
        scaled_window = scaler_features.transform(current_window)
        scaled_window_reshaped = scaled_window.reshape(1, WINDOW_SIZE, NUM_FEATURES)
        
        # Predict next log return
        pred_scaled = model.predict(scaled_window_reshaped, verbose=0)
        pred_log_return = scaler_target.inverse_transform(pred_scaled)[0, 0]
        
        # Convert log return to price
        next_price = predicted_prices[-1] * np.exp(pred_log_return)
        predicted_prices.append(next_price)
        
        # Calculate next business day
        next_date = last_date + timedelta(days=day+1)
        predicted_dates.append(next_date)
        
        # Update window - simplified approach
        new_row = current_window[-1].copy()
        new_row[0] = next_price  # Update Close price
        current_window = np.vstack([current_window[1:], new_row])
    
    return predicted_dates[1:], predicted_prices[1:]

def calculate_historical_metrics(model, scaler_features, scaler_target, df_clean):
    """Calculate model performance on historical data."""
    try:
        # Prepare data
        feature_data = df_clean[FEATURES].values
        target_data = df_clean['log_return'].values.reshape(-1, 1)
        
        scaled_features = scaler_features.transform(feature_data)
        scaled_target = scaler_target.transform(target_data)
        
        # Create sequences
        X, y = [], []
        for i in range(WINDOW_SIZE, len(scaled_features)):
            X.append(scaled_features[i-WINDOW_SIZE:i])
            y.append(scaled_target[i])
        X, y = np.array(X), np.array(y)
        
        # Use last 20% as test set
        split_idx = int(len(X) * 0.8)
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        # Predict
        predictions_scaled = model.predict(X_test, verbose=0)
        predictions = scaler_target.inverse_transform(predictions_scaled)
        actuals = scaler_target.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        
        return rmse, r2, actuals, predictions
    except Exception as e:
        return None, None, None, None

# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    # Header
    st.title("📈 LSTM Stock Price Predictor")
    st.markdown("### High-Fidelity Multivariate Time-Series Prediction Framework")
    st.markdown("---")
    
    # Load model and scalers
    with st.spinner("Loading model and scalers..."):
        model, scaler_features, scaler_target, error = load_model_and_scalers()
    
    if error:
        st.error(f"❌ Error loading artifacts: {error}")
        st.info("Please ensure lstm_model.h5, scaler_features.pkl, and scaler_target.pkl are in the same directory.")
        return
    
    st.success("✅ Model and scalers loaded successfully!")
    
    # Sidebar inputs
    st.sidebar.header("Configuration")
    
    # Ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        help="Enter a valid stock ticker (e.g., AAPL, GOOGL, MSFT)"
    ).upper()
    
    # Forecast days
    forecast_days = st.sidebar.number_input(
        "Forecast Horizon (days)",
        min_value=1,
        max_value=90,
        value=30,
        help="Number of future days to predict"
    )
    
    # Historical data range
    st.sidebar.subheader("Historical Data Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    # Predict button
    predict_button = st.sidebar.button("🚀 Generate Predictions", type="primary")
    
    # Main content
    if predict_button:
        with st.spinner(f"Fetching data for {ticker}..."):
            df_clean, error = download_and_process_data(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        
        if error:
            st.error(f"❌ Error: {error}")
            return
        
        if df_clean is None or len(df_clean) < WINDOW_SIZE + 50:
            st.error(f"❌ Insufficient data for {ticker}. Please try another ticker.")
            return
        
        st.success(f"✅ Downloaded {len(df_clean)} days of data for {ticker}")
        
        # Calculate metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${df_clean['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("Data Points", len(df_clean))
        with col3:
            st.metric("Date Range", f"{df_clean.index[0].strftime('%Y-%m-%d')}")
        with col4:
            st.metric("Latest Date", f"{df_clean.index[-1].strftime('%Y-%m-%d')}")
        
        # Historical performance
        st.markdown("---")
        st.subheader("📊 Model Performance on Historical Data")
        
        with st.spinner("Evaluating model performance..."):
            rmse, r2, actuals, predictions = calculate_historical_metrics(
                model, scaler_features, scaler_target, df_clean
            )
        
        if rmse is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE (Log Returns)", f"{rmse:.6f}")
            with col2:
                st.metric("R² Score", f"{r2:.4f}")
            
            # Plot historical performance
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                y=actuals.flatten(),
                mode='lines',
                name='Actual Log Returns',
                line=dict(color='blue', width=2)
            ))
            fig_hist.add_trace(go.Scatter(
                y=predictions.flatten(),
                mode='lines',
                name='Predicted Log Returns',
                line=dict(color='red', width=2)
            ))
            fig_hist.update_layout(
                title=f"{ticker} - Historical Prediction Performance",
                xaxis_title="Test Sample Index",
                yaxis_title="Log Return",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Future predictions
        st.markdown("---")
        st.subheader(f"🔮 {forecast_days}-Day Price Forecast")
        
        with st.spinner(f"Generating {forecast_days}-day forecast..."):
            pred_dates, pred_prices = predict_future_prices(
                model, scaler_features, scaler_target, df_clean, forecast_days
            )
        
        # Display prediction summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Starting Price", f"${df_clean['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("Predicted Price (Day 1)", f"${pred_prices[0]:.2f}",
                     delta=f"{((pred_prices[0] / df_clean['Close'].iloc[-1] - 1) * 100):.2f}%")
        with col3:
            st.metric(f"Predicted Price (Day {forecast_days})", f"${pred_prices[-1]:.2f}",
                     delta=f"{((pred_prices[-1] / df_clean['Close'].iloc[-1] - 1) * 100):.2f}%")
        
        # Plot future predictions
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=df_clean.index[-90:],
            y=df_clean['Close'].iloc[-90:],
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        ))
        
        # Predicted prices
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            mode='lines+markers',
            name='Predicted Prices',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f"{ticker} - Historical & Predicted Stock Prices",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction table
        st.markdown("---")
        st.subheader("📋 Detailed Predictions")
        
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Predicted Price': [f"${p:.2f}" for p in pred_prices],
            'Change from Current': [f"{((p / df_clean['Close'].iloc[-1] - 1) * 100):.2f}%" for p in pred_prices]
        })
        st.dataframe(pred_df, use_container_width=True)
        
        # Disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Disclaimer**: This tool is for educational and research purposes only. 
        Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. 
        Always consult with a qualified financial advisor before making investment decisions.
        """)
    
    else:
        # Display instructions
        st.info("""
        👋 Welcome to the LSTM Stock Price Predictor!
        
        **How to use:**
        1. Enter a stock ticker symbol in the sidebar (e.g., AAPL, GOOGL, MSFT)
        2. Choose how many days ahead you want to forecast
        3. Click the "Generate Predictions" button
        4. View the model's performance metrics and future price predictions
        
        **Model Features:**
        - Predicts logarithmic returns for statistical stationarity
        - Uses multivariate input: Close price, Volume, RSI, MACD, and EMA_20
        - Stacked LSTM architecture with dropout regularization
        - Leakage-proof scaling methodology
        """)
        
        # Display model architecture
        st.markdown("---")
        st.subheader("🏗️ Model Architecture")
        st.code("""
        Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 5)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # Output: Predicted log return
        ])
        """, language="python")

if __name__ == "__main__":
    main()