# 📈 LSTM Stock Price Prediction Framework

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/YOUR_USERNAME/lstm-stock-prediction/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/YOUR_USERNAME/lstm-stock-prediction/actions)

A high-fidelity, production-ready LSTM-based framework for multivariate time-series stock price prediction. This project implements a methodologically sound approach to financial forecasting with proper handling of non-stationary data and prevention of data leakage.

## 🌐 Live Demo

**Try it now**: [Live Streamlit App](https://YOUR-APP-URL.streamlit.app) *(Update after deployment)*

## 🎯 Key Features

- **Stationary Target Prediction**: Predicts logarithmic returns instead of raw prices for statistical stability
- **Multivariate Analysis**: Leverages multiple technical indicators (RSI, MACD, EMA) alongside price data
- **Leakage-Proof Scaling**: Implements separate scalers for features and target to prevent data contamination
- **Stacked LSTM Architecture**: Deep learning model with dropout regularization
- **Interactive Dashboard**: Streamlit-based web interface for real-time predictions
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations

## 📋 System Requirements

- Python 3.8 or higher
- 8GB RAM minimum
- GPU recommended (but not required)

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd lstm-stock-prediction
pip install -r requirements.txt
```

### 2. Train the Model

Run the main training script:

```bash
python main.py
```

This will:
- Download historical stock data (default: AAPL from 2015-2024)
- Calculate technical indicators
- Train the LSTM model
- Generate evaluation metrics and visualizations
- Save model artifacts (`lstm_model.h5`, `scaler_features.pkl`, `scaler_target.pkl`)

**Training time**: Approximately 10-15 minutes on CPU, 2-3 minutes on GPU

### 3. Launch the Dashboard

Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
lstm-stock-prediction/
│
├── main.py                    # Main training script
├── streamlit_app.py          # Deployment dashboard
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── lstm_model.h5             # Trained model (generated)
├── scaler_features.pkl       # Feature scaler (generated)
├── scaler_target.pkl         # Target scaler (generated)
│
├── evaluation_plot.png       # Performance visualization (generated)
└── training_history.png      # Training loss plot (generated)
```

## 🔧 Configuration

All key parameters can be modified in the configuration section of `main.py`:

```python
TICKER = 'AAPL'                    # Stock ticker symbol
START_DATE = '2015-01-01'          # Training data start date
END_DATE = '2024-12-31'            # Training data end date
WINDOW_SIZE = 60                   # Sequence length (lookback period)
TRAIN_SPLIT_PERCENT = 0.8          # Training/test split ratio
EPOCHS = 50                        # Number of training epochs
BATCH_SIZE = 32                    # Training batch size
FEATURES = ['Close', 'Volume', 'RSI', 'MACD', 'EMA_20']  # Input features
```

## 📊 Model Architecture

The framework uses a Stacked LSTM architecture:

```
Input Layer: (60 timesteps, 5 features)
    ↓
LSTM Layer 1: 50 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 2: 50 units, return_sequences=False
    ↓
Dropout: 0.2
    ↓
Dense Layer: 32 units, ReLU activation
    ↓
Output Layer: 1 unit (predicted log return)
```

**Total Parameters**: ~27,000 trainable parameters

## 🧮 Technical Methodology

### Feature Engineering

1. **Price Data**: Close price and Volume
2. **Technical Indicators**:
   - **RSI (Relative Strength Index)**: Momentum oscillator (14-period)
   - **MACD (Moving Average Convergence Divergence)**: Trend indicator
   - **EMA_20 (Exponential Moving Average)**: 20-period smoothed average

### Target Variable

The model predicts **logarithmic returns** rather than absolute prices:

```
log_return = ln(Price_t / Price_{t-1})
```

**Why log returns?**
- Transforms non-stationary price series into stationary returns
- Symmetric treatment of gains and losses
- Mathematical convenience for time-series analysis
- More stable model training

### Scaling Strategy

**Two-Scaler Approach** to prevent data leakage:

1. `scaler_features`: Scales input features independently
2. `scaler_target`: Scales target variable independently

This ensures no information from the target distribution "leaks" into the training features.

### Sequence Generation

The model uses a sliding window approach:
- Window size: 60 days
- Each sequence contains 60 timesteps of 5 features
- Predicts the log return for day 61

### Chronological Splitting

Data is split chronologically (no shuffling):
- First 80% → Training set
- Last 20% → Test set

This respects the temporal nature of financial data and prevents look-ahead bias.

## 📈 Using the Dashboard

### Step 1: Enter Stock Ticker
Type any valid stock ticker symbol (e.g., AAPL, GOOGL, TSLA, MSFT)

### Step 2: Set Forecast Horizon
Choose how many days ahead to predict (1-90 days)

### Step 3: Generate Predictions
Click "Generate Predictions" to:
- Download latest data for the ticker
- Calculate technical indicators
- Evaluate model performance on historical data
- Generate future price forecasts

### Step 4: Analyze Results
The dashboard displays:
- **Performance Metrics**: RMSE and R² on test data
- **Historical Performance**: Actual vs. predicted comparison
- **Future Forecast**: Predicted prices with confidence indicators
- **Detailed Table**: Day-by-day predictions

## 📊 Evaluation Metrics

### RMSE (Root Mean Squared Error)
Measures the average magnitude of prediction errors (in log return space).

**Lower is better** → Indicates more accurate predictions

### R² (Coefficient of Determination)
Measures how well the model explains variance in the data.

- **R² = 1**: Perfect predictions
- **R² = 0**: No better than predicting the mean
- **R² < 0**: Worse than baseline

**Typical ranges for financial data**: 0.3 - 0.7

## 🎓 Educational Value

This project demonstrates:

1. ✅ Proper handling of non-stationary time-series data
2. ✅ Prevention of data leakage in ML pipelines
3. ✅ Feature engineering for financial data
4. ✅ Deep learning for sequence prediction
5. ✅ Model evaluation and validation techniques
6. ✅ Production deployment with Streamlit

## ⚠️ Important Disclaimers

### For Educational Purposes Only

This framework is designed for:
- Learning about time-series prediction
- Understanding LSTM architectures
- Practicing MLOps and deployment
- Academic research and experimentation

### Not Financial Advice

- Stock market predictions are inherently uncertain
- Past performance does not guarantee future results
- This tool should NOT be used as the sole basis for investment decisions
- Always consult qualified financial advisors

### Known Limitations

1. **Simplified Feature Engineering**: Production systems use hundreds of features
2. **No Risk Management**: Doesn't account for volatility, drawdowns, or portfolio theory
3. **Iterative Prediction Limitations**: Future features are estimated, not calculated
4. **Market Regime Changes**: Model may not adapt to unprecedented market conditions
5. **No Transaction Costs**: Real trading involves fees, slippage, and spread

## 🔬 Advanced Usage

### Training on Different Stocks

Modify the `TICKER` parameter in `main.py`:

```python
TICKER = 'GOOGL'  # Google
TICKER = 'TSLA'   # Tesla
TICKER = 'MSFT'   # Microsoft
```

### Adjusting Hyperparameters

Experiment with different configurations:

```python
WINDOW_SIZE = 90      # Longer lookback period
EPOCHS = 100          # More training iterations
BATCH_SIZE = 64       # Larger batch size
```

### Adding More Features

Extend the `FEATURES` list:

```python
FEATURES = [
    'Close', 'Volume', 'RSI', 'MACD', 'EMA_20',
    'ATR',        # Average True Range
    'OBV',        # On-Balance Volume
    'Stoch_K'     # Stochastic Oscillator
]
```

Remember to update `NUM_FEATURES` and recalculate indicators in the data pipeline.

## 🐛 Troubleshooting

### Issue: "No data available for this ticker"
**Solution**: Verify the ticker symbol is correct and has sufficient historical data on Yahoo Finance.

### Issue: Model artifacts not found
**Solution**: Run `main.py` first to generate `lstm_model.h5`, `scaler_features.pkl`, and `scaler_target.pkl`.

### Issue: TensorFlow/CUDA errors
**Solution**: The model works on CPU. If using GPU, ensure compatible CUDA and cuDNN versions are installed.

### Issue: Memory errors during training
**Solution**: Reduce `BATCH_SIZE` or `WINDOW_SIZE` in the configuration.

## 📚 References & Resources

### Academic Papers
- Hochreiter & Schmidhuber (1997): Long Short-Term Memory
- Fischer & Krauss (2018): Deep Learning with LSTM Networks for Financial Market Predictions

### Libraries Used
- **TensorFlow/Keras**: Deep learning framework
- **yfinance**: Yahoo Finance data downloader
- **ta**: Technical analysis library
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library

### Further Reading
- [Time Series Forecasting with LSTM](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Technical Analysis Library Documentation](https://technical-analysis-library-in-python.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Ensemble methods (combining multiple models)
- [ ] Attention mechanisms
- [ ] Additional technical indicators
- [ ] Backtesting framework
- [ ] Real-time data streaming
- [ ] Portfolio optimization
- [ ] Sentiment analysis integration

## 📄 License

This project is provided for educational purposes. Users are responsible for ensuring compliance with applicable regulations when using financial data and predictions.

## 👨‍💻 Author

Akhiranand Boddani
Bhavan's Vivekananda College of Science, Humanities & Commerce Sainikpuri, Secunderabad
Version 1.0 - October 2025

---

**⭐ If you found this project helpful, please consider starring the repository!**
