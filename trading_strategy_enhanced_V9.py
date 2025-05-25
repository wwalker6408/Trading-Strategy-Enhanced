import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import time
import matplotlib.pyplot as plt  # For confusion matrix plot
# from functools import reduce # Alternative for union

# --- Configuration & Constants ---
LOG_DIR = Path.home() / "trading_strategy_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "trading_strategy_refactored.log"
UPDATE_INTERVAL_SECONDS = 300  # 5 minutes
DEFAULT_TICKERS = ["TSLA", "AAPL", "NVDA"]
MAX_DAYS_DAILY = 730
MAX_DAYS_INTRADAY = 60
MIN_DATA_FOR_ML = 100 # Minimum samples required to train ML model
ML_FEATURES = ['rsi', 'macd_line', 'macd_hist', 'momentum', 'ema_diff_20_50', 'vol_ratio', 'atr_norm']

# Configure logging
# Clear previous log file handlers if any exist from previous runs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=str(LOG_FILE),
    filemode='w',  # Overwrite log file each run
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logging.info("--- Enhanced Trading Strategy V9 Started ---")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Enhanced Trading Strategy V9", layout="wide")
st.title("Enhanced Trading Strategy V9")
st.info(f"Debug messages are being logged to: {LOG_FILE}")

# --- Session State Initialization ---
if 'custom_tickers' not in st.session_state:
    st.session_state.custom_tickers = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {} # Cache for trained models {ticker: {'long': model, 'short': model, 'report_long': report, 'report_short': report, 'cm_long': cm, 'cm_short': cm}}

# --- Helper Functions ---

def validate_ticker(ticker):
    """Checks if a ticker symbol is valid and has data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        # Requesting 1 day is usually enough to check existence
        data = stock.history(period="1d")
        if not data.empty:
            logging.debug(f"Validated ticker: {ticker}")
            return True
        else:
            # Check if it's a known issue (e.g., delisted but yfinance still finds it)
            info = stock.info # Fetch info only if history is empty
            if not info or info.get('regularMarketPrice') is None:
                 logging.warning(f"Ticker {ticker} found but has no recent market data (info: {info}).")
                 return False
            logging.warning(f"Ticker {ticker} validation returned empty data but info exists.")
            # Could be valid but no data for '1d', might be okay for longer periods
            # Let's try a slightly longer period for robustness
            data_longer = stock.history(period="5d")
            if not data_longer.empty:
                logging.debug(f"Validated ticker {ticker} with 5d data.")
                return True
            logging.warning(f"Ticker {ticker} has no data even for 5d.")
            return False
    except Exception as e:
        # Catch potential network errors or other yfinance issues
        logging.error(f"Error validating ticker {ticker}: {e}")
        return False

@st.cache_data(ttl=UPDATE_INTERVAL_SECONDS) # Cache data for 5 minutes
def get_stock_data(ticker, timeframe, days_back, _cache_key):
    """Downloads stock data using yfinance with error handling."""
    logging.info(f"Fetching data for {ticker} ({timeframe}, {days_back} days)")
    try:
        end_date = datetime.now()
        # Adjust days_back based on timeframe limits
        if timeframe == "1d":
            days_back = min(days_back, MAX_DAYS_DAILY)
        else:
            days_back = min(days_back, MAX_DAYS_INTRADAY)
        start_date = end_date - timedelta(days=days_back)

        # Use yf.Ticker().history for potentially more robust column handling
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=timeframe)
        # data = yf.download(ticker, start=start_date, end=end_date, interval=timeframe, progress=False) # Alternative

        if data.empty:
            st.error(f"No data returned for {ticker} with specified parameters.")
            logging.error(f"No data returned for {ticker} (Start: {start_date}, End: {end_date}, Interval: {timeframe})")
            return None

        # --- FIX: Handle potential MultiIndex or tuple column names ---
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex: typically take the first level name
            logging.debug(f"Detected MultiIndex columns for {ticker}: {data.columns}. Flattening.")
            # Join tuple elements with an underscore if needed, or just take the first part
            data.columns = [col[0].lower() if isinstance(col, tuple) and len(col) > 0 else str(col).lower() for col in data.columns.get_level_values(0)]
        else:
            # Handle potential tuples even in non-MultiIndex cases, or just simple strings
            logging.debug(f"Standardizing single-level columns for {ticker}: {data.columns}")
            data.columns = [col[0].lower() if isinstance(col, tuple) and len(col) > 0 else str(col).lower() for col in data.columns]

        # Remove potential duplicate columns after flattening/lowercasing
        data = data.loc[:, ~data.columns.duplicated()]
        logging.debug(f"Columns after standardization and deduplication for {ticker}: {data.columns.tolist()}")
        # --- END FIX ---


        # Rename 'adj close' if it exists after standardization
        data = data.rename(columns={'adj close': 'adj_close'})

        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in data.columns]
            st.error(f"Data for {ticker} missing expected columns after standardization: {missing}. Found: {data.columns.tolist()}")
            logging.error(f"Data for {ticker} missing columns after standardization: {missing}. Available: {data.columns.tolist()}")
            # Attempt to map common alternatives if possible (e.g., 'closing price' -> 'close') - more complex
            return None

        # Basic data cleaning - Forward fill then backward fill for NaNs
        initial_nans = data[expected_columns].isna().sum().sum()
        if initial_nans > 0:
            logging.warning(f"Found {initial_nans} NaN values in critical columns for {ticker}. Applying ffill/bfill.")
            # Apply fillna selectively to avoid issues with non-numeric columns if any snuck in
            for col in expected_columns:
                 if col in data.columns:
                     data[col] = data[col].ffill().bfill()


        if data[expected_columns].isna().sum().sum() > 0:
             st.error(f"Data for {ticker} still contains NaNs in critical columns after cleaning. Cannot proceed.")
             logging.error(f"Data for {ticker} still contains NaNs after ffill/bfill.")
             return None

        # Ensure correct data types
        for col in expected_columns:
             try:
                 data[col] = pd.to_numeric(data[col])
             except ValueError as e:
                 st.error(f"Could not convert column '{col}' to numeric for {ticker}. Error: {e}")
                 logging.error(f"Numeric conversion error for {ticker}, column '{col}': {e}")
                 # Optionally drop the column or return None depending on criticality
                 return None


        # Drop rows where critical data might still be missing after coercion (shouldn't happen after previous check)
        data.dropna(subset=expected_columns, inplace=True)

        if data.empty:
            st.error(f"Data for {ticker} became empty after cleaning/validation.")
            logging.error(f"Data for {ticker} empty after cleaning.")
            return None

        logging.debug(f"Successfully fetched and cleaned data for {ticker}. Shape: {data.shape}")
        return data

    except Exception as e:
        st.error(f"Error fetching or processing data for {ticker}: {e}")
        logging.error(f"Error fetching/processing data for {ticker}: {e}", exc_info=True)
        return None

def check_data_update(real_time_update_enabled):
    """Checks if data needs refreshing based on the update interval."""
    current_time = time.time()
    if real_time_update_enabled and (current_time - st.session_state.last_update) >= UPDATE_INTERVAL_SECONDS:
        logging.info("Real-time update triggered. Clearing cache.")
        # Clear specific caches if needed, or use st.cache_data's ttl
        # st.cache_data.clear() # Clears all @st.cache_data
        st.session_state.data_cache.clear() # Clear manual cache if used elsewhere
        st.session_state.ml_models.clear() # Clear ML models as data changed
        st.session_state.last_update = current_time
        st.rerun()

# --- Indicator Calculation Functions ---

def _calculate_rsi(close_prices, length):
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    # Use rolling mean for robustness against initial zeros in avg_loss
    avg_gain = gain.rolling(window=length, min_periods=1).mean()
    avg_loss = loss.rolling(window=length, min_periods=1).mean()
    # Use Wilder's smoothing (alternative - equivalent to EMA with alpha=1/length)
    # avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    # avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan) # Avoid division by zero, result in NaN
    rsi = 100 - (100 / (1 + rs))
    # Fill initial NaNs (due to rolling window) and NaNs from division by zero with 50
    return rsi.fillna(50)

def _calculate_ema(prices, length):
    return prices.ewm(span=length, adjust=False).mean()

def _calculate_macd(prices, fast_len=12, slow_len=26, signal_len=9):
    ema_fast = _calculate_ema(prices, fast_len)
    ema_slow = _calculate_ema(prices, slow_len)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_len, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def _calculate_atr(high, low, close, length):
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    # Use rolling mean for standard ATR
    atr = true_range.rolling(window=length, min_periods=1).mean()
    # atr = true_range.ewm(alpha=1/length, adjust=False).mean() # Alternative EMA-based ATR
    # Backfill initial NaNs, then forward fill any remaining (less likely)
    return atr.fillna(method='bfill').fillna(method='ffill')

def _calculate_bbands(prices, length, deviation):
    ma = prices.rolling(window=length, min_periods=1).mean()
    std_dev = prices.rolling(window=length, min_periods=1).std().fillna(0) # Handle NaN std dev early
    upper_band = ma + (std_dev * deviation)
    lower_band = ma - (std_dev * deviation)
    return upper_band, ma, lower_band

def calculate_indicators(df, params):
    """Calculates all technical indicators and adds them to the DataFrame."""
    logging.debug(f"Calculating indicators. Initial shape: {df.shape}")
    df_out = df.copy()
    try:
        # Standard Indicators
        df_out['rsi'] = _calculate_rsi(df_out['close'], params['rsi_length'])
        df_out['ema20'] = _calculate_ema(df_out['close'], 20)
        df_out['ema50'] = _calculate_ema(df_out['close'], 50)
        df_out['ema200'] = _calculate_ema(df_out['close'], 200)
        df_out['macd_line'], df_out['signal_line'], df_out['macd_hist'] = _calculate_macd(df_out['close'])
        df_out['atr'] = _calculate_atr(df_out['high'], df_out['low'], df_out['close'], params['atr_length'])
        df_out['vol_sma20'] = df_out['volume'].rolling(window=20, min_periods=1).mean()
        df_out['momentum'] = df_out['close'].diff(params.get('momentum_length', 3)) # Use param if available

        # Additional Indicators from original script
        df_out['ma10'] = df_out['close'].rolling(window=10, min_periods=1).mean()
        df_out['ma20'] = df_out['close'].rolling(window=20, min_periods=1).mean()
        df_out['rsi_bb_upper'], df_out['rsi_bb_middle'], df_out['rsi_bb_lower'] = _calculate_bbands(
            df_out['rsi'], params['rsi_bb_length'], params['rsi_bb_dev']
        )

        # Feature Engineering for ML / Conditions
        df_out['vol_ratio'] = (df_out['volume'] / df_out['vol_sma20'].replace(0, np.nan)).fillna(1.0) # Avoid NaN/inf
        df_out['atr_norm'] = (df_out['atr'] / df_out['close'].replace(0, np.nan)).fillna(method='bfill').fillna(method='ffill') # Normalize ATR, handle zero close
        df_out['ema_diff_20_50'] = df_out['ema20'] - df_out['ema50']

        # --- Define Base Conditions ---
        # These are calculated based purely on indicator values for the current bar

        # Volume Condition
        df_out['volume_condition'] = df_out['volume'] > (df_out['vol_sma20'] * params['vol_threshold'])

        # Trend Condition (Simple MA Cross)
        df_out['trend_strength_up'] = df_out['ma10'] > df_out['ma20']

        # RSI Conditions
        df_out['rsi_oversold_area'] = df_out['rsi'] < params['rsi_oversold']
        df_out['rsi_overbought_area'] = df_out['rsi'] > params['rsi_overbought']
        # Consider RSI relative to its Bollinger Bands
        df_out['rsi_below_bb_lower'] = df_out['rsi'] < df_out['rsi_bb_lower']
        df_out['rsi_above_bb_upper'] = df_out['rsi'] > df_out['rsi_bb_upper']

        # EMA Conditions
        df_out['close_above_ema20'] = df_out['close'] > df_out['ema20']
        df_out['close_above_ema50'] = df_out['close'] > df_out['ema50']
        df_out['close_above_ema200'] = df_out['close'] > df_out['ema200'] # Long term trend filter
        df_out['ema20_above_ema50'] = df_out['ema20'] > df_out['ema50'] # Medium term trend

        # MACD Conditions
        df_out['macd_bullish_cross'] = (df_out['macd_line'] > df_out['signal_line']) & (df_out['macd_line'].shift() <= df_out['signal_line'].shift())
        df_out['macd_bearish_cross'] = (df_out['macd_line'] < df_out['signal_line']) & (df_out['macd_line'].shift() >= df_out['signal_line'].shift())
        df_out['macd_positive'] = df_out['macd_line'] > 0

        # --- Define Entry/Exit Logic Conditions ---
        # Combine base conditions into specific signals

        # Original Logic (Simplified interpretation)
        # Long Entry: Close > EMA20 (potentially filtered by other conditions)
        # Short Entry: Close < EMA20 (potentially filtered by other conditions)
        # Long Exit: (RSI > RSI BB Upper AND Close < EMA50) OR Close < EMA200
        # Short Exit: (RSI < RSI BB Lower AND Close > EMA50) OR Close > EMA200

        # Refined Example Conditions (can be adjusted based on strategy goals)
        if params['bypass_conditions']:
            df_out['enter_long_signal'] = df_out['close_above_ema20']
            df_out['enter_short_signal'] = ~df_out['close_above_ema20'] # Close <= EMA20
        else:
            # Example: Require multiple confirmations for entry
            df_out['enter_long_signal'] = (
                df_out['close_above_ema50'] & # Basic uptrend filter
                df_out['ema20_above_ema50'] & # Medium term confirmation
                df_out['rsi'] > 50 # Momentum confirmation (adjust threshold) - Original didn't use 50 explicitly here
                # df_out['macd_positive'] # MACD confirmation - Removed to simplify slightly
                # Add volume condition? df_out['volume_condition']
            )
            df_out['enter_short_signal'] = (
                ~df_out['close_above_ema50'] & # Basic downtrend filter
                ~df_out['ema20_above_ema50'] & # Medium term confirmation
                df_out['rsi'] < 50 # Momentum confirmation
                # ~df_out['macd_positive'] # MACD confirmation - Removed
            )

        # Exit conditions from original logic (ensure columns exist)
        df_out['exit_long_signal'] = (
            (df_out['rsi_above_bb_upper'] & ~df_out['close_above_ema50']) | ~df_out['close_above_ema200']
        )
        df_out['exit_short_signal'] = (
            (df_out['rsi_below_bb_lower'] & df_out['close_above_ema50']) | df_out['close_above_ema200']
        )

        # Handle NaNs resulting from indicator calculations (especially rolling windows/diff)
        initial_rows = len(df_out)
        # Identify columns used in signals/features that might have leading NaNs
        cols_to_check_nan = ['rsi', 'ema20', 'ema50', 'ema200', 'macd_line', 'signal_line', 'atr', 'vol_sma20', 'momentum', 'ma10', 'ma20', 'rsi_bb_upper', 'rsi_bb_lower', 'atr_norm', 'ema_diff_20_50']
        cols_to_check_nan.extend(ML_FEATURES) # Ensure ML features are checked
        df_out = df_out.dropna(subset=[col for col in cols_to_check_nan if col in df_out.columns])
        final_rows = len(df_out)
        if initial_rows > final_rows:
             logging.warning(f"Dropped {initial_rows - final_rows} rows due to NaNs after indicator calculation.")

        if df_out.empty:
            st.error("DataFrame became empty after calculating indicators and dropping NaNs.")
            logging.error("DataFrame empty after indicator calculation NaNs drop.")
            return None

        logging.debug(f"Indicators calculated. Final shape: {df_out.shape}. Columns: {df_out.columns.tolist()}")
        return df_out

    except KeyError as e:
        st.error(f"Missing expected column during indicator calculation: {e}")
        logging.error(f"Missing column in calculate_indicators: {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"Unexpected error calculating indicators: {e}")
        logging.error(f"Error in calculate_indicators: {e}", exc_info=True)
        return None

# --- Machine Learning Functions ---

def train_ml_model(df, ticker, features=ML_FEATURES, test_size=0.2):
    """Trains XGBoost models for long/short signals, addressing lookahead bias."""
    logging.info(f"Starting ML model training for {ticker}")
    df_ml = df.copy()

    # Ensure all specified features actually exist in the dataframe
    available_features = [f for f in features if f in df_ml.columns]
    if len(available_features) != len(features):
        missing_features = [f for f in features if f not in available_features]
        logging.warning(f"Missing features for ML training on {ticker}: {missing_features}. Using available: {available_features}")
        if not available_features:
             st.error(f"No features available for ML training on {ticker}. Cannot train model.")
             logging.error(f"No features available for ML training on {ticker}.")
             return None, None, None, None, None, None, None, None # Added None for feature_importance and features_used

    # Define Targets: Predict if the *next* bar's close will be favorable
    # Long Target: Next Close > Next EMA20 (Buy signal valid if price likely to stay above EMA)
    df_ml['long_target'] = (df_ml['close'].shift(-1) > df_ml['ema20'].shift(-1)).astype(int)
    # Short Target: Next Close < Next EMA20 (Sell signal valid if price likely to stay below EMA)
    df_ml['short_target'] = (df_ml['close'].shift(-1) < df_ml['ema20'].shift(-1)).astype(int)

    # Drop rows with NaN targets (last row) and any NaNs in features
    df_ml_clean = df_ml[available_features + ['long_target', 'short_target']].dropna()

    if len(df_ml_clean) < MIN_DATA_FOR_ML:
        logging.warning(f"Insufficient data ({len(df_ml_clean)} rows) for ML model training on {ticker} after NaN drop. Need {MIN_DATA_FOR_ML}.")
        st.warning(f"Skipping ML for {ticker}: Not enough clean data ({len(df_ml_clean)} rows).")
        return None, None, None, None, None, None, None, None # Added None for feature_importance and features_used

    X = df_ml_clean[available_features]
    y_long = df_ml_clean['long_target']
    y_short = df_ml_clean['short_target']

    # Split data chronologically to prevent lookahead bias during training/testing
    # Train on the first part, test on the later part
    try:
        X_train, X_test, y_long_train, y_long_test = train_test_split(X, y_long, test_size=test_size, shuffle=False)
        # Use the same split indices for the short target
        _ , _, y_short_train, y_short_test = train_test_split(X, y_short, test_size=test_size, shuffle=False)
    except ValueError as e:
         st.error(f"Error splitting data for ML training on {ticker}: {e}. Check data length ({len(X)}).")
         logging.error(f"ML data split error for {ticker}: {e}")
         return None, None, None, None, None, None, None, None

    if len(X_train) == 0 or len(X_test) == 0:
        st.error(f"Data split resulted in empty train or test set for {ticker}. Cannot train ML model.")
        logging.error(f"Empty train/test set for {ticker}: Train={len(X_train)}, Test={len(X_test)}")
        return None, None, None, None, None, None, None, None


    logging.debug(f"ML Data split for {ticker}: Train={len(X_train)}, Test={len(X_test)}")

    # --- XGBoost Model Training with GridSearchCV ---
    xgb_clf = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    # Reduced param grid for faster execution in Streamlit context
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1] # Fixed learning rate for speed
    }
    # TimeSeriesSplit for cross-validation suitable for time series
    tscv = TimeSeriesSplit(n_splits=3)

    # Function to safely run GridSearchCV
    def safe_grid_search(estimator, X_tr, y_tr, param_grid_cv, tscv_cv):
        try:
            grid_search = GridSearchCV(estimator, param_grid_cv, cv=tscv_cv, scoring='f1', n_jobs=-1, error_score='raise')
            grid_search.fit(X_tr, y_tr)
            return grid_search.best_estimator_, grid_search.best_params_
        except ValueError as e:
            # Handle cases like "Only one class present in y_true." during CV folds
            logging.warning(f"GridSearchCV failed for {ticker} (likely due to single class in CV fold): {e}. Training with default params.")
            # Fallback: Train with default parameters
            estimator.fit(X_tr, y_tr)
            return estimator, "Default (GridSearch Failed)"
        except Exception as e:
            logging.error(f"Unexpected error during GridSearchCV for {ticker}: {e}", exc_info=True)
            st.error(f"ML Training Error (GridSearch) for {ticker}: {e}")
            return None, None # Indicate failure


    # Train Long Model
    logging.debug(f"Training Long model for {ticker}...")
    best_long_model, best_params_long = safe_grid_search(xgb_clf, X_train, y_long_train, param_grid, tscv)
    if best_long_model is None: return None, None, None, None, None, None, None, None # Propagate failure

    y_long_pred = best_long_model.predict(X_test)
    report_long = classification_report(y_long_test, y_long_pred, output_dict=True, zero_division=0)
    cm_long = confusion_matrix(y_long_test, y_long_pred, labels=best_long_model.classes_) # Ensure labels match model classes
    logging.info(f"Long Model Training Complete for {ticker}. Best Params: {best_params_long}")
    logging.debug(f"Long Model Test Report:\n{classification_report(y_long_test, y_long_pred, zero_division=0)}")

    # Train Short Model
    logging.debug(f"Training Short model for {ticker}...")
    # Need a new instance of the classifier for the second grid search
    xgb_clf_short = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    best_short_model, best_params_short = safe_grid_search(xgb_clf_short, X_train, y_short_train, param_grid, tscv)
    if best_short_model is None: return None, None, None, None, None, None, None, None

    y_short_pred = best_short_model.predict(X_test)
    report_short = classification_report(y_short_test, y_short_pred, output_dict=True, zero_division=0)
    cm_short = confusion_matrix(y_short_test, y_short_pred, labels=best_short_model.classes_)
    logging.info(f"Short Model Training Complete for {ticker}. Best Params: {best_params_short}")
    logging.debug(f"Short Model Test Report:\n{classification_report(y_short_test, y_short_pred, zero_division=0)}")

    # --- Feature Importance ---
    feature_importance = pd.DataFrame({
        'Feature': available_features, # Use only features actually used
        'Long Importance': best_long_model.feature_importances_,
        'Short Importance': best_short_model.feature_importances_
    }).sort_values(by=['Long Importance', 'Short Importance'], ascending=False)

    return best_long_model, best_short_model, report_long, report_short, cm_long, cm_short, feature_importance, available_features # Return features used

def apply_ml_predictions(df, model_long, model_short, features):
    """Applies trained ML models to generate prediction signals."""
    df_pred = df.copy()
    # Initialize columns first
    df_pred['ml_long_signal'] = False
    df_pred['ml_short_signal'] = False

    if model_long is None or model_short is None or not features:
        logging.warning("ML models or features not available, skipping prediction application.")
        return df_pred

    # Ensure all features are present and in correct order for prediction
    features_for_pred = [f for f in features if f in df_pred.columns]
    if len(features_for_pred) != len(features):
         logging.warning(f"Some features used in training are missing from prediction data: {set(features) - set(features_for_pred)}")
         # Decide how to handle: error out, or predict with available? For now, proceed with available.
         if not features_for_pred:
              logging.error("No features available for prediction.")
              return df_pred # Return with default False signals

    X_pred_full = df_pred[features_for_pred]
    # Handle potential NaNs introduced *after* indicator calculation but before prediction (less likely)
    X_pred_clean_indices = X_pred_full.dropna().index
    X_pred_clean = X_pred_full.loc[X_pred_clean_indices]


    if not X_pred_clean.empty:
        try:
            pred_long = model_long.predict(X_pred_clean)
            pred_short = model_short.predict(X_pred_clean)

            # Assign predictions back to the original DataFrame index using the clean indices
            df_pred.loc[X_pred_clean_indices, 'ml_long_signal'] = (pred_long == 1)
            df_pred.loc[X_pred_clean_indices, 'ml_short_signal'] = (pred_short == 1)

            logging.debug(f"Applied ML predictions. Long signals: {df_pred['ml_long_signal'].sum()}, Short signals: {df_pred['ml_short_signal'].sum()}")

        except Exception as e:
             logging.error(f"Error during ML prediction application: {e}", exc_info=True)
             st.error(f"Error applying ML predictions: {e}")
             # Keep signals as False in case of error
    else:
        logging.warning("No data available for ML prediction after handling NaNs.")

    # Ensure boolean type
    df_pred['ml_long_signal'] = df_pred['ml_long_signal'].astype(bool)
    df_pred['ml_short_signal'] = df_pred['ml_short_signal'].astype(bool)

    return df_pred


# --- Backtesting Engine ---

def calculate_position_size(equity, risk_per_trade_pct, atr, close_price):
    """Calculates position size based on ATR volatility."""
    # This function is defined but not currently used in run_backtest's PnL calculation.
    # To implement fully: calculate shares = (equity * risk_per_trade_pct / 100) / (stop_distance_points)
    # stop_distance_points could be based on ATR (e.g., 2 * atr) or fixed percentage.
    # Then PnL = (exit_price - entry_price) * shares - costs
    if atr is None or atr <= 0 or close_price <= 0: # Check for non-positive values
        logging.warning(f"Invalid inputs for position sizing: ATR={atr}, Close={close_price}. Defaulting size.")
        return 0.01 # Return a minimum size or handle as error
    risk_amount = equity * (risk_per_trade_pct / 100.0)
    # Example: Stop distance = 2 * ATR
    stop_distance_points = 2 * atr
    if stop_distance_points == 0:
         logging.warning("Stop distance is zero, cannot calculate position size.")
         return 0.01 # Min size
    # Calculate value per share/unit
    value_per_point = 1 # Assume 1 for stocks, adjust for futures/forex contracts
    shares = risk_amount / (stop_distance_points * value_per_point)

    # Convert shares to fraction of equity for the current simplified PnL calc (Needs review)
    position_size_fraction = (shares * close_price) / equity

    return max(0.01, min(1.0, position_size_fraction)) # Ensure size is within reasonable bounds (e.g., 1% to 100% of equity exposure)


def run_backtest(df, initial_capital, params):
    """Runs the backtest simulation based on calculated signals."""
    logging.info(f"Starting backtest. Initial capital: {initial_capital}")
    if df is None or df.empty:
        st.error("Cannot run backtest: Input DataFrame is empty.")
        logging.error("Backtest failed: Input DataFrame is empty.")
        return pd.DataFrame(), [] # Return empty results

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            logging.debug("Converted DataFrame index to DatetimeIndex.")
        except Exception as e:
            st.error(f"Failed to convert DataFrame index to DatetimeIndex: {e}")
            logging.error(f"Index conversion failed: {e}")
            return pd.DataFrame(), []


    df_backtest = df.copy()
    n = len(df_backtest)
    if n <= 1:
        st.error("Cannot run backtest: Not enough data rows (<= 1).")
        logging.error("Backtest failed: Not enough data rows (<= 1).")
        return pd.DataFrame(), []

    # Initialize backtest columns
    df_backtest['position'] = 0          # -1: Short, 0: Flat, 1: Long
    df_backtest['entry_price'] = np.nan
    df_backtest['stop_loss_price'] = np.nan
    df_backtest['take_profit_price'] = np.nan
    df_backtest['trailing_stop_price'] = np.nan # For trailing stop
    df_backtest['bars_in_trade'] = 0
    df_backtest['equity'] = initial_capital
    df_backtest['pnl'] = 0.0             # Profit/Loss for the current bar
    df_backtest['trade_result_pct'] = np.nan # P/L % of completed trade (relative to entry)
    df_backtest['trade_result_abs'] = np.nan # P/L $ amount of completed trade


    trades = []
    current_trade = None
    equity = initial_capital

    # Determine which signals to use (ML or standard)
    use_ml = params['use_ml_signals']
    # Ensure ML columns exist before assigning, otherwise default to standard signals
    if use_ml and 'ml_long_signal' in df_backtest.columns and 'ml_short_signal' in df_backtest.columns:
        long_entry_col = 'ml_long_signal'
        short_entry_col = 'ml_short_signal'
        logging.info("Using ML signals for backtest.")
    else:
        long_entry_col = 'enter_long_signal'
        short_entry_col = 'enter_short_signal'
        if use_ml:
             logging.warning("ML signals requested but columns missing. Falling back to indicator signals.")
        else:
             logging.info("Using indicator signals for backtest.")

    long_exit_col = 'exit_long_signal'
    short_exit_col = 'exit_short_signal'

    # Verify signal columns exist
    required_cols = [long_entry_col, short_entry_col, long_exit_col, short_exit_col,
                     'open', 'high', 'low', 'close', 'atr']
    missing_cols = [col for col in required_cols if col not in df_backtest.columns]
    if missing_cols:
        st.error(f"Backtest cannot proceed. Missing required columns: {missing_cols}")
        logging.error(f"Backtest failed. Missing columns: {missing_cols}")
        return pd.DataFrame(), []

    logging.debug(f"Backtest using signals: Long Entry='{long_entry_col}', Short Entry='{short_entry_col}'")

    # --- Backtesting Loop ---
    # Use iterrows for explicit row access, though slower than vectorization
    # Vectorized backtesting is complex to implement correctly with path-dependent logic like trailing stops
    for i in range(1, n): # Start from the second row to allow lookback
        idx = df_backtest.index[i]
        idx_prev = df_backtest.index[i-1]

        # Get previous state
        prev_position = df_backtest.loc[idx_prev, 'position']
        current_equity = df_backtest.loc[idx_prev, 'equity'] # Equity at the START of the bar i
        entry_price = df_backtest.loc[idx_prev, 'entry_price']
        stop_loss_price = df_backtest.loc[idx_prev, 'stop_loss_price']
        take_profit_price = df_backtest.loc[idx_prev, 'take_profit_price']
        trailing_stop_price = df_backtest.loc[idx_prev, 'trailing_stop_price']
        bars_in_trade = df_backtest.loc[idx_prev, 'bars_in_trade']

        # Get current bar data
        current_high = df_backtest.loc[idx, 'high']
        current_low = df_backtest.loc[idx, 'low']
        current_close = df_backtest.loc[idx, 'close']
        # current_atr = df_backtest.loc[idx, 'atr'] # ATR for sizing/stops if needed

        position = prev_position # Start with previous position
        trade_result_pct = np.nan
        trade_result_abs = np.nan
        pnl_bar = 0.0 # PnL generated *during* this bar

        exit_price = np.nan
        exit_reason = None

        # --- Handle Exits ---
        if position == 1: # Currently Long
            bars_in_trade += 1
            # Update Trailing Stop (based on previous close or current bar's close)
            # Using current_close makes it react faster but introduces slight lookahead within the bar
            # Using previous close is safer but lags. Let's use current_close for this example.
            if params['use_trailing_stop'] and not np.isnan(trailing_stop_price):
                potential_ts = current_close * (1 - params['trail_percent'] / 100.0)
                trailing_stop_price = max(trailing_stop_price, potential_ts) # Update if price moves favorably

            # Check Exit Conditions (Order matters: Stop Loss > Take Profit > Trailing Stop > Signal > Time)
            # Assume exits happen based on price touching the level during the bar
            if current_low <= stop_loss_price:
                exit_price = stop_loss_price # Assume worst price execution
                exit_reason = "Stop Loss"
            elif current_high >= take_profit_price:
                exit_price = take_profit_price # Assume best price execution
                exit_reason = "Take Profit"
            elif params['use_trailing_stop'] and not np.isnan(trailing_stop_price) and current_low <= trailing_stop_price:
                exit_price = trailing_stop_price # Assume worst price execution
                exit_reason = "Trailing Stop"
            elif df_backtest.loc[idx, long_exit_col]: # Check signal at the end of the bar
                exit_price = current_close # Exit at close on signal
                exit_reason = "Exit Signal"
            elif bars_in_trade >= params['max_bars_in_trade']:
                exit_price = current_close
                exit_reason = "Time Exit"

        elif position == -1: # Currently Short
            bars_in_trade += 1
             # Update Trailing Stop (if enabled)
            if params['use_trailing_stop'] and not np.isnan(trailing_stop_price):
                potential_ts = current_close * (1 + params['trail_percent'] / 100.0)
                trailing_stop_price = min(trailing_stop_price, potential_ts) # Update if price moves favorably

            # Check Exit Conditions
            if current_high >= stop_loss_price:
                exit_price = stop_loss_price # Assume worst price execution
                exit_reason = "Stop Loss"
            elif current_low <= take_profit_price:
                exit_price = take_profit_price # Assume best price execution
                exit_reason = "Take Profit"
            elif params['use_trailing_stop'] and not np.isnan(trailing_stop_price) and current_high >= trailing_stop_price:
                 exit_price = trailing_stop_price # Assume worst price execution
                 exit_reason = "Trailing Stop"
            elif df_backtest.loc[idx, short_exit_col]: # Check signal at end of bar
                exit_price = current_close
                exit_reason = "Exit Signal"
            elif bars_in_trade >= params['max_bars_in_trade']:
                exit_price = current_close
                exit_reason = "Time Exit"

        # --- Process Trade Exit ---
        if exit_reason:
            # Calculate costs based on exit price
            commission = params['commission_pct'] / 100.0 * abs(exit_price)
            slippage = params['slippage_pct'] / 100.0 * abs(exit_price)

            # Calculate P/L percentage relative to entry price
            if position == 1:
                # Slippage hurts longs on exit (lower effective exit price)
                effective_exit_price = exit_price - slippage
                trade_result_pct = (effective_exit_price - entry_price - commission) / entry_price
            else: # Short
                # Slippage helps shorts on exit (higher effective exit price)
                effective_exit_price = exit_price + slippage
                trade_result_pct = (entry_price - effective_exit_price - commission) / entry_price

            # Calculate absolute PnL based on equity *at entry* (more accurate for % calc)
            # This assumes full equity was invested, which isn't true with position sizing.
            # A truly accurate PnL requires tracking shares/contracts.
            # Simplified PnL for now: Apply percentage result to equity *at the start of the current bar*
            trade_result_abs = current_equity * trade_result_pct
            equity = current_equity + trade_result_abs # Update equity
            pnl_bar = trade_result_abs # PnL for this bar is the result of the closed trade

            logging.debug(f"Trade Closed: {position} at {idx}. Entry: {entry_price:.2f}, Exit: {exit_price:.2f} (Eff: {effective_exit_price:.2f}), Reason: {exit_reason}, Result%: {trade_result_pct*100:.2f}, PnL: {trade_result_abs:.2f}, Equity: {equity:.2f}")

            # Store trade details
            if current_trade:
                current_trade['exit_date'] = idx
                current_trade['exit_price'] = exit_price
                current_trade['exit_reason'] = exit_reason
                current_trade['profit_loss_pct'] = trade_result_pct * 100
                current_trade['profit_loss'] = trade_result_abs # Store actual PnL amount
                trades.append(current_trade)
                current_trade = None
            else:
                 logging.warning(f"Exit occurred at {idx} but no 'current_trade' was active.")


            # Reset state after exit
            position = 0
            bars_in_trade = 0
            entry_price = np.nan
            stop_loss_price = np.nan
            take_profit_price = np.nan
            trailing_stop_price = np.nan

        # --- Handle Entries ---
        # Check entry signals ONLY if position is now flat (i.e., not exited on this bar AND flat previously, or just exited)
        if position == 0:
            can_enter = True # Add logic here if needed (e.g., max concurrent trades)
            entry_cost_pct = (params['commission_pct'] + params['slippage_pct']) / 100.0

            # Check Long Entry Signal at the end of the current bar
            if can_enter and df_backtest.loc[idx, long_entry_col]:
                position = 1
                entry_price = current_close # Enter at close
                stop_loss_price = entry_price * (1 - params['stop_loss'] / 100.0)
                take_profit_price = entry_price * (1 + params['take_profit'] / 100.0)
                # Initial trailing stop based on entry price
                trailing_stop_price = entry_price * (1 - params['trail_percent'] / 100.0) if params['use_trailing_stop'] else np.nan
                bars_in_trade = 1 # Start counting bars

                # Calculate entry costs and apply to equity
                entry_cost_abs = equity * entry_cost_pct # Apply cost to current equity before trade
                equity -= entry_cost_abs
                pnl_bar -= entry_cost_abs # Costs reduce PnL for this bar

                current_trade = {
                    'type': 'Long',
                    'entry_date': idx,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'initial_equity': equity # Track equity *after* costs at entry
                }
                logging.debug(f"Trade Opened: Long at {idx}. Entry: {entry_price:.2f}, SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}, Costs: {entry_cost_abs:.2f}, Equity: {equity:.2f}")


            # Check Short Entry (only if not entering long on the same bar)
            elif can_enter and df_backtest.loc[idx, short_entry_col]:
                position = -1
                entry_price = current_close
                stop_loss_price = entry_price * (1 + params['stop_loss'] / 100.0)
                take_profit_price = entry_price * (1 - params['take_profit'] / 100.0)
                trailing_stop_price = entry_price * (1 + params['trail_percent'] / 100.0) if params['use_trailing_stop'] else np.nan
                bars_in_trade = 1

                # Calculate entry costs and apply to equity
                entry_cost_abs = equity * entry_cost_pct
                equity -= entry_cost_abs
                pnl_bar -= entry_cost_abs

                current_trade = {
                    'type': 'Short',
                    'entry_date': idx,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                     'initial_equity': equity
                }
                logging.debug(f"Trade Opened: Short at {idx}. Entry: {entry_price:.2f}, SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}, Costs: {entry_cost_abs:.2f}, Equity: {equity:.2f}")

        # --- Update DataFrame for the current row (idx) ---
        # If still in a trade (not exited this bar), calculate unrealized PnL for the bar
        if position != 0 and not exit_reason:
            # Calculate change based on close prices
            price_change = current_close - df_backtest.loc[idx_prev, 'close']
            # Simplified PnL - assumes full equity exposure
            unrealized_pnl = (price_change / df_backtest.loc[idx_prev, 'close']) * current_equity * position # position is 1 or -1
            equity = current_equity + unrealized_pnl
            pnl_bar = unrealized_pnl

        # If flat and no entry/exit, equity remains the same
        elif position == 0 and not exit_reason:
            equity = current_equity
            pnl_bar = 0.0


        # Assign values to the current row (idx)
        df_backtest.loc[idx, 'position'] = position
        df_backtest.loc[idx, 'entry_price'] = entry_price # Carries forward if in trade
        df_backtest.loc[idx, 'stop_loss_price'] = stop_loss_price # Carries forward
        df_backtest.loc[idx, 'take_profit_price'] = take_profit_price # Carries forward
        df_backtest.loc[idx, 'trailing_stop_price'] = trailing_stop_price # Updated or carried forward
        df_backtest.loc[idx, 'bars_in_trade'] = bars_in_trade
        df_backtest.loc[idx, 'equity'] = equity
        df_backtest.loc[idx, 'pnl'] = pnl_bar # PnL generated *in* this bar
        df_backtest.loc[idx, 'trade_result_pct'] = trade_result_pct # P/L % of trade closed *on this bar*
        df_backtest.loc[idx, 'trade_result_abs'] = trade_result_abs # P/L $ of trade closed *on this bar*


    logging.info(f"Backtest completed. Final equity: {equity:.2f}. Total trades: {len(trades)}")
    # Final check for NaN equity which indicates a problem
    if df_backtest['equity'].isna().any():
         logging.error("NaN values found in equity curve after backtest!")
         st.error("Internal Error: NaN values detected in equity calculation. Check logs.")
    return df_backtest, trades

# --- Plotting Functions ---

def plot_results(df, trades, ticker, params):
    """Creates the main Plotly chart with price, indicators, signals, and equity."""
    fig = make_subplots(
        rows=4, cols=1, # Added equity row
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2], # Adjusted heights
        subplot_titles=("Price / EMAs / Trades", "RSI", "MACD", "Equity Curve & Drawdown"),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]] # Secondary Y for drawdown
    )

    # Row 1: Price and EMAs
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], name="EMA 20", line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema50'], name="EMA 50", line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema200'], name="EMA 200", line=dict(color='purple', width=1, dash='dash')), row=1, col=1)

    # Add Trade Markers
    if trades: # Check if trades list is not empty
        trades_df = pd.DataFrame(trades)
        # Ensure date columns are datetime objects
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

        long_entries = trades_df[trades_df['type'] == 'Long']
        short_entries = trades_df[trades_df['type'] == 'Short']
        exits = trades_df # All exits

        if not long_entries.empty:
             fig.add_trace(go.Scatter(
                x=long_entries['entry_date'], y=long_entries['entry_price'],
                mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name="Long Entry"
             ), row=1, col=1)
        if not short_entries.empty:
             fig.add_trace(go.Scatter(
                x=short_entries['entry_date'], y=short_entries['entry_price'],
                mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name="Short Entry"
             ), row=1, col=1)
        # Optional: Mark exits distinctly based on profit/loss
        profit_exits = exits[exits['profit_loss'] > 0]
        loss_exits = exits[exits['profit_loss'] <= 0]
        if not profit_exits.empty:
            fig.add_trace(go.Scatter(
                x=profit_exits['exit_date'], y=profit_exits['exit_price'],
                mode='markers', marker=dict(symbol='x', size=8, color='lime'), name="Profit Exit" # Lime green 'x'
            ), row=1, col=1)
        if not loss_exits.empty:
            fig.add_trace(go.Scatter(
                x=loss_exits['exit_date'], y=loss_exits['exit_price'],
                mode='markers', marker=dict(symbol='x', size=8, color='magenta'), name="Loss Exit" # Magenta 'x'
            ), row=1, col=1)


    # Row 2: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name="RSI", line=dict(color='darkcyan', width=1)), row=2, col=1) # Changed color
    fig.add_hline(y=params['rsi_overbought'], line=dict(color='red', width=1, dash='dash'), name="Overbought", row=2, col=1)
    fig.add_hline(y=params['rsi_oversold'], line=dict(color='green', width=1, dash='dash'), name="Oversold", row=2, col=1)
    # Add RSI BBands if they exist
    if 'rsi_bb_upper' in df.columns and 'rsi_bb_lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi_bb_upper'], name="RSI BB Up", line=dict(color='lightgrey', width=1, dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi_bb_lower'], name="RSI BB Low", line=dict(color='lightgrey', width=1, dash='dot')), row=2, col=1)

    # Row 3: MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_line'], name="MACD Line", line=dict(color='blue', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['signal_line'], name="Signal Line", line=dict(color='red', width=1)), row=3, col=1)
    colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name="MACD Hist", marker_color=colors), row=3, col=1)
    fig.add_hline(y=0, line=dict(color='grey', width=1, dash='dash'), name="MACD Zero Line", row=3, col=1)

    # Row 4: Equity Curve & Drawdown
    fig.add_trace(go.Scatter(x=df.index, y=df['equity'], name="Equity", line=dict(color='purple', width=2)), row=4, col=1, secondary_y=False)
    # Calculate Drawdown
    running_max = df['equity'].cummax()
    drawdown = (df['equity'] - running_max) / running_max.replace(0, np.nan) * 100 # Avoid division by zero if equity hits 0
    fig.add_trace(go.Scatter(x=df.index, y=drawdown.fillna(0), name="Drawdown (%)", line=dict(color='red', width=1, dash='dash'), fill='tozeroy'), row=4, col=1, secondary_y=True)


    # Layout Updates
    ml_suffix = '(XGBoost Signals)' if params['use_ml_signals'] and 'ml_long_signal' in df.columns else '(Indicator Signals)'
    fig.update_layout(
        height=900, # Increased height for 4 rows
        xaxis_rangeslider_visible=False,
        title=f"{ticker} - Strategy Backtest {ml_suffix}",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="x unified" # Show tooltips for all subplots on hover
    )
    # Update Y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100]) # Set RSI range
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1, secondary_y=True, showgrid=False, range=[-100, 1]) # Set drawdown range

    st.plotly_chart(fig, use_container_width=True)

def plot_confusion_matrix(cm, labels, title):
    """Plots a confusion matrix using matplotlib, handling potential missing classes."""
    fig, ax = plt.subplots(figsize=(4, 3)) # Smaller figure size
    try:
        # Ensure labels match the dimensions of the confusion matrix
        # If the model only predicted one class, cm might be smaller than expected
        if cm.shape[0] != len(labels) or cm.shape[1] != len(labels):
             logging.warning(f"Confusion matrix dimensions ({cm.shape}) do not match labels ({labels}) for {title}. Plotting with default labels.")
             # Attempt to plot with generic labels based on cm shape
             effective_labels = [f"Class {i}" for i in range(cm.shape[0])]
             if len(effective_labels) < 2: effective_labels = [0, 1] # Default for binary if needed
             disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=effective_labels[:cm.shape[0]])

        else:
             disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False) # No colorbar for smaller plot
        ax.set_title(title, fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.xlabel("Predicted Label", fontsize=9)
        plt.ylabel("True Label", fontsize=9)
        # Prevent tight layout warning and ensure labels fit
        plt.tight_layout(pad=0.5)
    except Exception as e:
         logging.error(f"Failed to plot confusion matrix for {title}: {e}", exc_info=True)
         ax.text(0.5, 0.5, 'Error plotting CM', horizontalalignment='center', verticalalignment='center')

    return fig


# --- Performance Metrics ---

def calculate_metrics(df_backtest, trades, initial_capital):
    """Calculates key performance metrics from backtest results."""
    metrics = {
        "Total Trades": 0, "Winning Trades": 0, "Losing Trades": 0,
        "Win Rate (%)": 0, "Net Profit ($)": 0, "Net Profit (%)": 0,
        "Gross Profit ($)": 0, "Gross Loss ($)": 0, "Profit Factor": np.nan, # Use NaN for undefined
        "Max Drawdown (%)": 0, "Sharpe Ratio": np.nan, "Sortino Ratio": np.nan,
        "Avg Trade Duration (bars)": np.nan, "Avg Trade PnL ($)": np.nan, "Avg Win ($)": np.nan, "Avg Loss ($)": np.nan
    }
    if not trades:
        # Calculate drawdown even if no trades occurred (measures initial capital fluctuation if any)
        equity_curve = df_backtest['equity']
        if not equity_curve.empty:
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max.replace(0, np.nan) # Avoid division by zero
            metrics["Max Drawdown (%)"] = drawdown.min() * 100 if not drawdown.empty else 0
        return metrics # Return default metrics if no trades

    trades_df = pd.DataFrame(trades)
    metrics["Total Trades"] = len(trades_df)

    # Ensure profit_loss is numeric
    trades_df['profit_loss'] = pd.to_numeric(trades_df['profit_loss'], errors='coerce')
    trades_df.dropna(subset=['profit_loss'], inplace=True) # Drop trades if PnL is invalid

    if trades_df.empty:
         logging.warning("Trade list became empty after coercing PnL to numeric.")
         return metrics # Return defaults if all trades had invalid PnL

    wins = trades_df[trades_df['profit_loss'] > 0]
    losses = trades_df[trades_df['profit_loss'] <= 0] # Include zero PnL trades as losses/break-even

    metrics["Winning Trades"] = len(wins)
    metrics["Losing Trades"] = len(losses)
    if metrics["Total Trades"] > 0:
        metrics["Win Rate (%)"] = (metrics["Winning Trades"] / metrics["Total Trades"]) * 100
        metrics["Avg Trade PnL ($)"] = trades_df['profit_loss'].mean()


    metrics["Gross Profit ($)"] = wins['profit_loss'].sum()
    metrics["Gross Loss ($)"] = losses['profit_loss'].sum() # Already negative or zero
    metrics["Net Profit ($)"] = trades_df['profit_loss'].sum() # Simpler sum
    metrics["Net Profit (%)"] = (metrics["Net Profit ($)"] / initial_capital) * 100

    if metrics["Gross Loss ($)"] != 0:
        # Ensure Gross Profit is positive for meaningful Profit Factor
        metrics["Profit Factor"] = abs(metrics["Gross Profit ($)"] / metrics["Gross Loss ($)"])
    elif metrics["Gross Profit ($)"] > 0:
         metrics["Profit Factor"] = float('inf') # Infinite profit factor if no losses but gains exist
    else:
         metrics["Profit Factor"] = 0.0 # Or NaN, debatable. 0 if no profit and no loss.

    # Avg Win / Loss
    if metrics["Winning Trades"] > 0:
        metrics["Avg Win ($)"] = metrics["Gross Profit ($)"] / metrics["Winning Trades"]
    if metrics["Losing Trades"] > 0:
        metrics["Avg Loss ($)"] = metrics["Gross Loss ($)"] / metrics["Losing Trades"]


    # Drawdown
    equity_curve = df_backtest['equity']
    running_max = equity_curve.cummax()
    # Avoid division by zero if equity starts at or hits zero
    drawdown = (equity_curve - running_max) / running_max.replace(0, np.nan)
    metrics["Max Drawdown (%)"] = drawdown.min() * 100 if not drawdown.empty else 0

    # Sharpe & Sortino (using daily returns approximation if needed)
    returns = equity_curve.pct_change().dropna()
    if not returns.empty and returns.std() != 0:
        # Determine annualization factor based on timeframe (approximate)
        time_diff_seconds = df_backtest.index.to_series().diff().median().total_seconds()
        bars_per_year = 31_536_000 / time_diff_seconds if time_diff_seconds > 0 else 252 # Default to daily if unknown
        annualization_factor = np.sqrt(bars_per_year)

        # Assuming risk-free rate is 0 for simplicity
        sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor
        metrics["Sharpe Ratio"] = sharpe_ratio

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        if downside_std is not None and downside_std != 0:
             sortino_ratio = (returns.mean() / downside_std) * annualization_factor
             metrics["Sortino Ratio"] = sortino_ratio

    # Trade Duration
    if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        # Calculate duration only for trades with valid exit dates
        valid_durations = trades_df.dropna(subset=['entry_date', 'exit_date'])
        if not valid_durations.empty:
            durations_seconds = (valid_durations['exit_date'] - valid_durations['entry_date']).dt.total_seconds()
            # Infer bar duration (needs improvement for mixed timeframes)
            time_diffs = df_backtest.index.to_series().diff().median().total_seconds()
            if time_diffs > 0:
                 metrics["Avg Trade Duration (bars)"] = durations_seconds.mean() / time_diffs


    return metrics


# --- Streamlit UI Components ---

def setup_sidebar():
    """Configures and returns parameters from the Streamlit sidebar."""
    st.sidebar.header("Strategy Parameters")

    params = {}

    with st.sidebar.expander("Indicators", expanded=True):
        params['rsi_length'] = st.number_input("RSI Length", min_value=1, value=14)
        params['rsi_overbought'] = st.number_input("RSI Overbought Level", min_value=50, max_value=100, value=70) # Adjusted default
        params['rsi_oversold'] = st.number_input("RSI Oversold Level", min_value=0, max_value=50, value=30) # Adjusted default
        params['rsi_bb_length'] = st.number_input("RSI BB Length", min_value=1, value=20)
        params['rsi_bb_dev'] = st.number_input("RSI BB Deviation", min_value=0.1, value=2.0, step=0.1)
        params['atr_length'] = st.number_input("ATR Length", min_value=1, value=14)
        params['vol_threshold'] = st.number_input("Volume Threshold Multiplier", min_value=0.1, value=1.5, step=0.1) # Adjusted default

    with st.sidebar.expander("Risk Management", expanded=True):
        params['stop_loss'] = st.number_input("Stop Loss %", min_value=0.1, value=3.0, step=0.1)
        params['take_profit'] = st.number_input("Take Profit %", min_value=0.1, value=6.0, step=0.1)
        params['use_trailing_stop'] = st.checkbox("Use Trailing Stop", value=True) # Default True
        params['trail_percent'] = st.number_input("Trailing Stop %", min_value=0.1, value=2.0, step=0.1, disabled=not params['use_trailing_stop']) # Disable if not used
        params['max_bars_in_trade'] = st.number_input("Max Bars in Trade", min_value=1, value=50) # Increased default
        # params['risk_per_trade'] = st.slider("Risk % per Trade (for Vol Sizing - Not fully implemented)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    with st.sidebar.expander("Trading Costs", expanded=False):
        params['commission_pct'] = st.number_input("Commission % (per side)", min_value=0.0, value=0.05, step=0.01, format="%.4f")
        params['slippage_pct'] = st.number_input("Slippage % (per side)", min_value=0.0, value=0.05, step=0.01, format="%.4f")

    with st.sidebar.expander("Execution Logic", expanded=False):
        params['bypass_conditions'] = st.checkbox("Bypass Complex Conditions (Use Simple EMA Cross)", value=False)
        params['use_ml_signals'] = st.checkbox("Use ML Signals (XGBoost - Requires Training)", value=False)

    st.sidebar.header("Data Settings")
    custom_ticker = st.sidebar.text_input("Add Custom Ticker (e.g., GOOG, MSFT)", "").strip().upper()
    # Ensure default tickers are uppercase
    default_tickers_upper = [t.upper() for t in DEFAULT_TICKERS]
    # Combine default and unique custom tickers
    available_tickers = sorted(list(set(default_tickers_upper + st.session_state.custom_tickers)))


    if custom_ticker and custom_ticker not in available_tickers:
        with st.spinner(f"Validating {custom_ticker}..."):
            is_valid = validate_ticker(custom_ticker)
        if is_valid:
            st.session_state.custom_tickers.append(custom_ticker)
            st.session_state.custom_tickers = sorted(list(set(st.session_state.custom_tickers))) # Keep sorted unique
            st.sidebar.success(f"Added {custom_ticker}")
            # Force rerun to update multiselect options
            st.rerun() # Use st.rerun() instead of experimental
        else:
            st.sidebar.error(f"Invalid or no data for ticker: {custom_ticker}")

    # Default selection logic
    default_selection = [t for t in default_tickers_upper if t in available_tickers]
    if not default_selection and available_tickers: # If default not available, select first available
        default_selection = [available_tickers[0]]


    params['tickers'] = st.sidebar.multiselect("Select Tickers", available_tickers, default=default_selection)
    params['timeframe'] = st.sidebar.selectbox("Timeframe", options=["1d", "1h", "30m", "15m", "5m"], index=0) # Added 30m
    params['days_back'] = st.sidebar.number_input("Days of Historical Data", min_value=30, value=365) # Min 30 days
    params['real_time_update'] = st.sidebar.checkbox("Enable Auto-Refresh (5min)", value=False)

    params['initial_capital'] = st.sidebar.number_input("Initial Capital", min_value=1000, value=10000)

    # --- Input Validation ---
    validation_error = False
    if params['rsi_overbought'] <= params['rsi_oversold']:
        st.sidebar.error("RSI Overbought must be greater than Oversold")
        validation_error = True
    if params['timeframe'] == "1d" and params['days_back'] > MAX_DAYS_DAILY:
        st.sidebar.warning(f"Days back limited to {MAX_DAYS_DAILY} for daily data.")
        params['days_back'] = MAX_DAYS_DAILY
    elif params['timeframe'] != "1d" and params['days_back'] > MAX_DAYS_INTRADAY:
        st.sidebar.warning(f"Days back limited to {MAX_DAYS_INTRADAY} for intraday data.")
        params['days_back'] = MAX_DAYS_INTRADAY

    if validation_error:
         st.stop() # Prevent execution with invalid params

    logging.info(f"Sidebar Parameters: {params}")
    return params

def display_metrics(metrics):
    """Displays performance metrics in columns."""
    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", f"{metrics['Total Trades']:.0f}")
        st.metric("Win Rate", f"{metrics['Win Rate (%)']:.2f}%")
        st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}" if metrics['Profit Factor'] is not None and np.isfinite(metrics['Profit Factor']) else "N/A")
    with col2:
        st.metric("Net Profit", f"${metrics['Net Profit ($)']:.2f}")
        st.metric("Net Profit %", f"{metrics['Net Profit (%)']:.2f}%")
        st.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")
    with col3:
        st.metric("Avg Win ($)", f"${metrics['Avg Win ($)']:.2f}" if metrics['Avg Win ($)'] is not None and not np.isnan(metrics['Avg Win ($)']) else "N/A")
        st.metric("Avg Loss ($)", f"${metrics['Avg Loss ($)']:.2f}" if metrics['Avg Loss ($)'] is not None and not np.isnan(metrics['Avg Loss ($)']) else "N/A")
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}" if metrics['Sharpe Ratio'] is not None and not np.isnan(metrics['Sharpe Ratio']) else "N/A")
    with col4:
        st.metric("Avg PnL ($)", f"${metrics['Avg Trade PnL ($)']:.2f}" if metrics['Avg Trade PnL ($)'] is not None and not np.isnan(metrics['Avg Trade PnL ($)']) else "N/A")
        st.metric("Avg Duration (Bars)", f"{metrics['Avg Trade Duration (bars)']:.1f}" if metrics['Avg Trade Duration (bars)'] is not None and not np.isnan(metrics['Avg Trade Duration (bars)']) else "N/A")
        st.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}" if metrics['Sortino Ratio'] is not None and not np.isnan(metrics['Sortino Ratio']) else "N/A")


def display_trades(trades):
    """Displays the list of executed trades."""
    st.subheader("Trade List")
    if not trades:
        st.info("No trades were executed during the backtest period.")
        return

    trade_df = pd.DataFrame(trades)
    # Ensure datetime columns and handle potential NaT
    trade_df['entry_date'] = pd.to_datetime(trade_df['entry_date'], errors='coerce')
    trade_df['exit_date'] = pd.to_datetime(trade_df['exit_date'], errors='coerce')
    trade_df.dropna(subset=['entry_date', 'exit_date', 'profit_loss'], inplace=True) # Drop invalid trades

    if trade_df.empty:
        st.info("No valid trades remain after cleaning.")
        return

    trade_df['duration_hrs'] = (trade_df['exit_date'] - trade_df['entry_date']).dt.total_seconds() / 3600

    display_df = trade_df[['type', 'entry_date', 'exit_date', 'entry_price', 'exit_price',
                           'profit_loss', 'profit_loss_pct', 'exit_reason', 'duration_hrs']].copy()

    # Formatting for display
    display_df['entry_date'] = display_df['entry_date'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['exit_date'] = display_df['exit_date'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['entry_price'] = display_df['entry_price'].round(4) # More precision for price
    display_df['exit_price'] = display_df['exit_price'].round(4)
    display_df['profit_loss'] = display_df['profit_loss'].round(2)
    display_df['profit_loss_pct'] = display_df['profit_loss_pct'].round(2)
    display_df['duration_hrs'] = display_df['duration_hrs'].round(1)

    display_df.columns = ['Type', 'Entry Date', 'Exit Date', 'Entry Price', 'Exit Price',
                          'Profit/Loss ($)', 'Profit/Loss (%)', 'Exit Reason', 'Duration (hrs)']

    def color_profit_loss(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'darkgrey'
        return f'color: {color}'

    styled_df = display_df.style.applymap(color_profit_loss, subset=['Profit/Loss ($)', 'Profit/Loss (%)'])
    st.dataframe(styled_df, use_container_width=True, height=300) # Added height

    # Trade Duration Histogram
    st.subheader("Trade Duration Distribution")
    if not trade_df.empty and 'duration_hrs' in trade_df.columns:
        fig_hist = px.histogram(
            trade_df,
            x='duration_hrs',
            nbins=30,
            title="Trade Duration Distribution (Hours)",
            labels={'duration_hrs': 'Duration (Hours)', 'count': 'Number of Trades'}
        )
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Could not generate duration histogram.")


# --- Main Application Logic ---

def run_analysis_for_ticker(ticker, params):
    """Performs the full analysis pipeline for a single ticker."""
    st.header(f"Analysis for: {ticker}")
    # Use columns for layout
    main_col, report_col = st.columns([3, 1]) # Main chart area, smaller report area

    with main_col:
        with st.spinner(f"Processing {ticker}..."):
            # 1. Get Data
            cache_key_data = f"{ticker}_{params['timeframe']}_{params['days_back']}"
            df_raw = get_stock_data(ticker, params['timeframe'], params['days_back'], cache_key_data)

            if df_raw is None or df_raw.empty:
                st.error(f"Failed to fetch or process data for {ticker}. Skipping.")
                logging.error(f"Data fetching failed for {ticker}.")
                return None, None, None # Return None for df, trades, metrics

            # 2. Calculate Indicators
            df_indicators = calculate_indicators(df_raw, params)
            if df_indicators is None or df_indicators.empty:
                st.error(f"Failed to calculate indicators for {ticker}. Skipping.")
                logging.error(f"Indicator calculation failed for {ticker}.")
                return None, None, None

            # 3. Train ML Model (if selected) - Train *before* backtest
            model_long, model_short, report_long, report_short, cm_long, cm_short, feature_importance, features_used = (None,) * 8
            ml_trained_successfully = False
            if params['use_ml_signals']:
                # Check cache first
                cache_key_ml = f"{ticker}_{params['timeframe']}_{params['days_back']}_{'_'.join(sorted(ML_FEATURES))}" # More specific cache key
                if cache_key_ml in st.session_state.ml_models:
                     logging.info(f"Using cached ML model for {ticker}")
                     cached_model = st.session_state.ml_models[cache_key_ml]
                     model_long, model_short = cached_model['long'], cached_model['short']
                     report_long, report_short = cached_model['report_long'], cached_model['report_short']
                     cm_long, cm_short = cached_model['cm_long'], cached_model['cm_short']
                     feature_importance = cached_model['feature_importance']
                     features_used = cached_model['features']
                     ml_trained_successfully = True # Assume cached model was successful
                else:
                    logging.info(f"Training ML model for {ticker}...")
                    try:
                        # Pass only available features to train_ml_model
                        current_features = [f for f in ML_FEATURES if f in df_indicators.columns]
                        if not current_features:
                             st.error(f"No ML features available in data for {ticker}. Cannot train.")
                             logging.error(f"No ML features for {ticker} before training call.")
                        else:
                             model_long, model_short, report_long, report_short, cm_long, cm_short, feature_importance, features_used = train_ml_model(
                                 df_indicators, ticker, features=current_features # Pass available features
                             )
                             # Cache the trained model and reports only if successful
                             if model_long and model_short:
                                 st.session_state.ml_models[cache_key_ml] = {
                                     'long': model_long, 'short': model_short,
                                     'report_long': report_long, 'report_short': report_short,
                                     'cm_long': cm_long, 'cm_short': cm_short,
                                     'feature_importance': feature_importance,
                                     'features': features_used # Store features actually used
                                 }
                                 ml_trained_successfully = True
                             else:
                                 st.warning(f"ML Training did not complete successfully for {ticker}. Check logs.")
                                 logging.warning(f"ML training returned None for {ticker}.")

                    except Exception as e:
                        st.error(f"ML Model training failed for {ticker}: {e}")
                        logging.error(f"ML training failed for {ticker}: {e}", exc_info=True)
                        # Ensure we don't proceed with ML signals if training failed
                        params['use_ml_signals'] = False # Fallback to indicator signals
                        model_long, model_short = None, None # Ensure models are None

                # Apply predictions using the trained model if successful
                if ml_trained_successfully and model_long and model_short:
                     df_indicators = apply_ml_predictions(df_indicators, model_long, model_short, features_used)
                else:
                     # Ensure columns exist even if ML failed or was skipped
                     df_indicators['ml_long_signal'] = False
                     df_indicators['ml_short_signal'] = False
                     # If ML was selected but failed, inform user and switch off for backtest
                     if params['use_ml_signals']:
                          st.warning(f"ML signal generation failed for {ticker}. Using indicator signals instead.")
                          params['use_ml_signals'] = False # Ensure backtest uses indicator signals


            # 4. Run Backtest using the potentially modified df_indicators and params
            df_backtest, trades = run_backtest(df_indicators, params['initial_capital'], params)
            if df_backtest.empty:
                st.error(f"Backtest failed for {ticker}. No results to display.")
                logging.error(f"Backtest failed for {ticker}.")
                return None, None, None

            # 5. Calculate Metrics
            metrics = calculate_metrics(df_backtest, trades, params['initial_capital'])

            # 6. Display Results in Main Column
            display_metrics(metrics)
            plot_results(df_backtest, trades, ticker, params)
            display_trades(trades)

            # Add download buttons
            st.subheader(f"Export Data - {ticker}")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if trades:
                    csv_trades = pd.DataFrame(trades).to_csv(index=False)
                    st.download_button(
                        label=f"Download Trades CSV",
                        data=csv_trades,
                        file_name=f"trades_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key=f"download_trades_{ticker}" # Unique key
                    )
                else:
                    st.info("No trades to download.")
            with col_dl2:
                csv_backtest = df_backtest.to_csv()
                st.download_button(
                    label=f"Download Backtest Data CSV",
                    data=csv_backtest,
                    file_name=f"backtest_data_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key=f"download_data_{ticker}" # Unique key
                )

    # Display ML-specific results in the right column if used and successful
    with report_col:
        st.write("---") # Separator for the column
        if params['use_ml_signals'] and ml_trained_successfully:
            st.subheader("ML Performance")
            st.write("**Long Signal Model (Test Set):**")
            if report_long:
                 st.dataframe(pd.DataFrame(report_long).transpose().round(3))
                 fig_cm_long = plot_confusion_matrix(cm_long, model_long.classes_, "CM (Long)")
                 st.pyplot(fig_cm_long)
            else:
                 st.write("Report not available.")

            st.write("**Short Signal Model (Test Set):**")
            if report_short:
                 st.dataframe(pd.DataFrame(report_short).transpose().round(3))
                 fig_cm_short = plot_confusion_matrix(cm_short, model_short.classes_, "CM (Short)")
                 st.pyplot(fig_cm_short)
            else:
                 st.write("Report not available.")


            st.subheader("ML Feature Importance")
            if feature_importance is not None and not feature_importance.empty:
                 st.dataframe(feature_importance.round(3))
            else:
                 st.info("Feature importance data is not available.")
        elif params['use_ml_signals'] and not ml_trained_successfully:
             st.warning("ML Signals were selected, but model training/loading failed.")
        else:
             st.info("ML Signals not selected.")


    return df_backtest, trades, metrics


# --- Main Streamlit App Execution ---
if __name__ == "__main__":
    params = setup_sidebar()
    check_data_update(params['real_time_update'])

    if not params['tickers']:
        st.warning("Please select at least one ticker symbol.")
        st.stop()

    # Initialize lists to store results for portfolio analysis
    all_backtest_dfs = {}
    all_trades = {}
    all_metrics = {}

    # --- Run Analysis for Each Ticker ---
    # Use tabs for multiple tickers if selected
    if len(params['tickers']) > 1:
        tabs = st.tabs(params['tickers'])
        for i, ticker in enumerate(params['tickers']):
             with tabs[i]:
                 df_backtest, trades, metrics = run_analysis_for_ticker(ticker, params.copy()) # Pass copy of params
                 if df_backtest is not None:
                     all_backtest_dfs[ticker] = df_backtest
                     all_trades[ticker] = trades
                     all_metrics[ticker] = metrics
    elif len(params['tickers']) == 1:
         ticker = params['tickers'][0]
         df_backtest, trades, metrics = run_analysis_for_ticker(ticker, params.copy())
         if df_backtest is not None:
             all_backtest_dfs[ticker] = df_backtest
             all_trades[ticker] = trades
             all_metrics[ticker] = metrics
    else:
         # This case should be caught by the check above, but added for completeness
         st.warning("No tickers selected.")
         st.stop()


    # --- Portfolio Analysis Section (Displayed after individual tickers) ---
    if len(params['tickers']) > 0 and all_metrics: # Only show if tickers were processed
        st.markdown("---")
        st.header("Portfolio Overview (Aggregated Metrics)")

        portfolio_summary = pd.DataFrame(all_metrics).T # Transpose to have tickers as rows

        # Ensure numeric types before aggregation
        for col in ["Net Profit ($)", "Total Trades", "Win Rate (%)", "Sharpe Ratio", "Max Drawdown (%)"]:
             if col in portfolio_summary.columns:
                 portfolio_summary[col] = pd.to_numeric(portfolio_summary[col], errors='coerce')

        portfolio_summary = portfolio_summary.round(2) # Round for display

        st.subheader("Individual Ticker Summary")
        st.dataframe(portfolio_summary[[
            "Net Profit ($)", "Net Profit (%)", "Win Rate (%)",
            "Profit Factor", "Max Drawdown (%)", "Total Trades", "Sharpe Ratio", "Sortino Ratio"
        ]].dropna(axis=1, how='all')) # Drop columns that are entirely NaN

        # Aggregate Portfolio Metrics (Simple Sum/Average)
        total_net_profit = portfolio_summary["Net Profit ($)"].sum()
        total_trades = portfolio_summary["Total Trades"].sum()
        # Weighted average win rate (by number of trades)
        avg_win_rate = (portfolio_summary["Win Rate (%)"].fillna(0) * portfolio_summary["Total Trades"].fillna(0)).sum() / total_trades if total_trades > 0 else 0
        # Average Sharpe/Drawdown (handle NaNs)
        avg_sharpe = portfolio_summary["Sharpe Ratio"].mean(skipna=True)
        avg_max_drawdown = portfolio_summary["Max Drawdown (%)"].mean(skipna=True)


        st.subheader("Aggregated Portfolio Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Net Profit", f"${total_net_profit:.2f}")
        with col2:
            st.metric("Total Trades", f"{total_trades:.0f}")
            st.metric("Average Win Rate", f"{avg_win_rate:.2f}%")
        with col3:
            st.metric("Average Max Drawdown", f"{avg_max_drawdown:.2f}%" if not np.isnan(avg_max_drawdown) else "N/A")
            st.metric("Average Sharpe Ratio", f"{avg_sharpe:.2f}" if not np.isnan(avg_sharpe) else "N/A")

        # Combined Equity Curve (Simple Average - Requires common index)
        # This remains an approximation, true portfolio backtest is complex
        if len(all_backtest_dfs) > 1: # Only makes sense for multiple tickers
            try:
                # Find the union of all datetime indices
                all_indices = [df.index for df in all_backtest_dfs.values() if isinstance(df.index, pd.DatetimeIndex)]
                if not all_indices:
                     raise ValueError("No valid DatetimeIndex found in backtest results.")

                # --- FIX: Replace union_many with iterative union ---
                common_index = pd.DatetimeIndex([]) # Start with empty index
                if all_indices:
                    common_index = all_indices[0] # Initialize with the first index
                    for idx in all_indices[1:]:
                        common_index = common_index.union(idx) # Iteratively union with others
                # --- END FIX ---


                aligned_equities = []
                initial_cap = params['initial_capital']
                for ticker, df in all_backtest_dfs.items():
                    if isinstance(df.index, pd.DatetimeIndex):
                        # Reindex and forward fill equity curves, fill initial NaNs with capital
                        aligned = df['equity'].reindex(common_index).ffill().fillna(initial_cap)
                        aligned_equities.append(aligned)
                    else:
                         logging.warning(f"Skipping {ticker} for portfolio curve due to non-datetime index.")


                if aligned_equities:
                    # Calculate average equity (equal weight)
                    portfolio_equity = pd.concat(aligned_equities, axis=1).mean(axis=1, skipna=True) # Simple average

                    st.subheader("Average Portfolio Equity Curve")
                    fig_portfolio_equity = go.Figure()
                    fig_portfolio_equity.add_trace(go.Scatter(
                        x=portfolio_equity.index, y=portfolio_equity, name="Avg Portfolio Equity",
                        line=dict(color='navy') # Changed color
                    ))
                    # Add drawdown calculation for the average curve
                    portfolio_running_max = portfolio_equity.cummax()
                    portfolio_drawdown = (portfolio_equity - portfolio_running_max) / portfolio_running_max.replace(0, np.nan) * 100
                    fig_portfolio_equity.add_trace(go.Scatter(
                         x=portfolio_equity.index, y=portfolio_drawdown.fillna(0), name="Avg Drawdown (%)",
                         line=dict(color='crimson', dash='dash'), yaxis="y2", fill='tozeroy' # Changed color
                    ))

                    fig_portfolio_equity.update_layout(
                        height=400,
                        title="Average Portfolio Equity and Drawdown (Equal Weight, Approx.)",
                        yaxis_title="Average Equity ($)",
                        yaxis2=dict(title="Drawdown (%)", overlaying="y", side="right", showgrid=False, range=[-100, 1]),
                        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_portfolio_equity, use_container_width=True)
                else:
                     st.warning("Could not align equity curves for portfolio view (possibly due to index issues).")


            except Exception as e:
                st.warning(f"Could not generate average portfolio equity curve: {e}")
                logging.warning(f"Portfolio equity curve generation failed: {e}", exc_info=True)
        elif len(all_backtest_dfs) == 1:
             st.info("Portfolio equity curve requires more than one ticker.")


    st.markdown("---")
    st.caption("Enhanced Trading Strategy V9 - Refactored Python Implementation")
    logging.info("--- Strategy Execution Completed Successfully ---")

