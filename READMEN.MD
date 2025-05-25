# Enhanced Trading Strategy V9

**Enhanced Trading Strategy V9** is a Streamlit-powered desktop application that combines technical analysis and machine learning to backtest and evaluate stock trading strategies. It provides a robust GUI for users to load data, configure indicators, simulate trades, and analyze strategy performance.

## 🚀 Features

- **Streamlit GUI** for easy configuration and visualization
- **Technical Indicators**:
  - RSI, MACD, ATR, EMA, Bollinger Bands, and more
- **Machine Learning**:
  - XGBoost-based model to predict long/short signals
  - TimeSeriesSplit cross-validation
- **Backtesting Engine**:
  - Handles position sizing, stop-loss, take-profit, and trailing stops
  - Simulates realistic order execution with slippage and commissions
- **Advanced Metrics**:
  - Profit Factor, Sharpe Ratio, Drawdown, Win Rate, and more
- **Interactive Visualizations**:
  - Plotly charts for price action, indicators, equity curve, and drawdown
  - Matplotlib confusion matrix for model evaluation
- **Structured Logging**:
  - Logs saved to `~/trading_strategy_logs/trading_strategy_refactored.log`

## 🛠️ Requirements

- Python 3.8+
- Packages:
  ```bash
  pip install -r requirements.txt
  ```

streamlit run trading_strategy_enhanced_V9.py

Required Libraries
streamlit

pandas

numpy

yfinance

plotly

xgboost

scikit-learn

matplotlib

📊 How It Works
Load and Validate Tickers: Choose from defaults or enter your own.

Configure Strategy Parameters: RSI levels, EMAs, ATR length, etc.

Train ML Models (optional): Use XGBoost to predict entry signals.

Run Backtest: Simulates trades using selected signals (manual or ML).

Analyze Results:

View metrics (Net Profit, Max Drawdown, Win Rate, etc.)

Inspect trade list and durations

Review confusion matrices

Explore interactive charts

🧠 Machine Learning Approach
Target Definitions:

Long: Close > EMA20 (next bar)

Short: Close < EMA20 (next bar)

Features Used:

RSI, MACD, Momentum, ATR, Volume Ratios, EMA Differences

Model:

XGBoostClassifier

GridSearchCV with TimeSeriesSplit for parameter tuning

📈 Visualization
Main Plot: Candlestick with EMAs, trade entries, exits

RSI & MACD: Plotted with thresholds and Bollinger overlays

Equity Curve: With drawdown shaded

Confusion Matrix: Separate for Long and Short models

🧪 Example Usage
bash
Copy
Edit
streamlit run trading_strategy_enhanced_V9.py
Then interact with the sidebar to:

Select ticker(s) like AAPL, TSLA, NVDA

Choose time frame (1d, 1h, 15m, etc.)

Enable or disable ML predictions

View strategy performance in the main panel

📁 Logs & Output
Logs stored at:

bash
Copy
Edit
~/trading_strategy_logs/trading_strategy_refactored.log
Data caching enabled via @st.cache_data

🔍 License
MIT License. See LICENSE file for details.

🙌 Acknowledgements
Streamlit

Yahoo Finance via yfinance

XGBoost

scikit-learn

Plotly
