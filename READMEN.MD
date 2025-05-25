# 📈 Enhanced Trading Strategy Suite

This repository provides a powerful combination of a **Streamlit-based backtesting framework** and a **Tkinter-based IBKR trading bot**. Together, they enable strategy research, machine learning-enhanced signal generation, and optional live trading via Interactive Brokers.

## 🧠 Streamlit App: `trading_strategy_enhanced_V9.py`

A full-featured trading strategy simulator with:

### ✅ Features

- **Technical Indicators**: RSI, MACD, ATR, Bollinger Bands, EMAs, Volume, and Momentum
- **Machine Learning Signals**: Trains XGBoost models to generate long/short signals
- **Interactive GUI**: Configure parameters via Streamlit sidebar
- **Backtest Engine**: Includes:
  - Stop loss, take profit, trailing stop
  - Equity tracking, PnL, drawdown
  - Realistic cost modeling (slippage, commission)
- **Visualization**:
  - Multi-panel Plotly charts for price, RSI, MACD, equity curve, drawdown
  - Matplotlib confusion matrices
- **Performance Metrics**:
  - Net Profit, Win Rate, Drawdown, Sharpe Ratio, Sortino Ratio, Profit Factor, and more
- **Logging & Caching**:
  - Logs stored at `~/trading_strategy_logs/trading_strategy_refactored.log`
  - Caching with `@st.cache_data` for speed

### 🛠 Run the App

```bash
streamlit run trading_strategy_enhanced_V9.py

### How To Run
python IBKR_Trading_Bot_V2.py

### Tech Stack
Python 3.8+

Streamlit: GUI and data interaction

XGBoost: Machine learning signal generation

Plotly + Matplotlib: Visualizations

yfinance: Historical stock data

scikit-learn: Model evaluation

IB API: Real-time trading interface

### Requirements Text
streamlit>=1.30.0
pandas>=1.3.0
numpy>=1.21.0
yfinance>=0.2.0
plotly>=5.9.0
xgboost>=1.7.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
ibapi>=9.81.1.post1

## Project Structure
📁 trading-strategy-suite/
├── trading_strategy_enhanced_V9.py    # Streamlit strategy simulator
├── IBKR_Trading_Bot_V2.py             # Live trading GUI bot using IB API
├── requirements.txt
└── README.md                          # This file


### Acknowledgements
Streamlit

Interactive Brokers API

Yahoo Finance

XGBoost

scikit-learn

Plotly
```
