from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import requests
import ta
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64, os, threading, time

app = Flask(__name__)

MODEL_PATH = "rf_model01.pkl"
HISTORY_CSV = "history.csv"

# simulation / realtime params
CANDLE_INTERVAL = "15m"
PLAYBACK_DELAY_SECONDS = 1      # how fast playback updates (1s per candle)
REALTIME_POLL_MS = 15 * 60 * 1000  # after playback, how often frontend polls (ms) -> 15 minutes

# how many candles -> 30 hours -> 30*4 = 120 (use margin)
CANDLES_TO_FETCH = 130
PLAYBACK_HOURS = 24
PLAYBACK_CANDLES = 24 * 4  # 96

# load model once (will raise if not found)
model = joblib.load(MODEL_PATH)

# ----------------------
# Binance fetch helpers
# ----------------------
def fetch_klines(symbol="BTCUSDT", interval=CANDLE_INTERVAL, limit=CANDLES_TO_FETCH):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])
    # convert types
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)
    return df

# ----------------------
# Safe indicator helpers
# ----------------------
def safe_apply_indicator(df, func, min_periods, *args, **kwargs):
    if len(df) < min_periods:
        return pd.Series([np.nan]*len(df), index=df.index)
    try:
        return func(*args, **kwargs)
    except Exception:
        return pd.Series([np.nan]*len(df), index=df.index)

# ----------------------
# Feature generation function (unchanged)
# ----------------------
def build_features_for_df(df, threshold = 0.8 ):
    df_feat = df.copy()

    # Ensure required columns
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in df_feat.columns:
            raise ValueError(f"Missing column: {c}")

    # ======================================================
    # 1️⃣ RSI + Signal
    # ======================================================
    df_feat['RSI'] = ta.momentum.RSIIndicator(df_feat['Close'], window=14).rsi()
    df_feat['RSI_signal'] = np.where(df_feat['RSI'] > 70, 'Sell',
                              np.where(df_feat['RSI'] < 30, 'Buy', 'Hold'))

    # ======================================================
    # 2️⃣ ATR
    # ======================================================
    df_feat['ATR'] = ta.volatility.AverageTrueRange(
        df_feat['High'], df_feat['Low'], df_feat['Close'], window=14
    ).average_true_range()

    # ======================================================
    # 3️⃣ Moving Averages (EMA, SMA, WMA)
    # ======================================================
    df_feat['EMA_12'] = ta.trend.EMAIndicator(df_feat['Close'], window=12).ema_indicator()
    df_feat['SMA_14'] = ta.trend.SMAIndicator(df_feat['Close'], window=14).sma_indicator()
    df_feat['WMA_14'] = ta.trend.WMAIndicator(df_feat['Close'], window=14).wma()

    df_feat['EMA_relative'] = df_feat['Close'] / df_feat['EMA_12']
    df_feat['MA_relative'] = df_feat['Close'] / df_feat['Close'].rolling(window=14).mean()
    df_feat['SMA_relative'] = df_feat['Close'] / df_feat['SMA_14']
    df_feat['WMA_relative'] = df_feat['Close'] / df_feat['WMA_14']

    # ======================================================
    # 4️⃣ MACD
    # ======================================================
    macd = ta.trend.MACD(df_feat['Close'])
    df_feat['MACD'] = macd.macd()
    df_feat['MACD_signal'] = macd.macd_signal()
    df_feat['MACD_hist'] = macd.macd_diff()
    df_feat['MACD_relative'] = df_feat['MACD'] / df_feat['Close']
    df_feat['MACD_relative_signal'] = df_feat['MACD_signal'] / df_feat['Close']
    df_feat['MACD_relative_histogram'] = df_feat['MACD_hist'] / df_feat['Close']

    # ======================================================
    # 5️⃣ ADX
    # ======================================================
    df_feat['ADX'] = ta.trend.ADXIndicator(
        df_feat['High'], df_feat['Low'], df_feat['Close'], window=14
    ).adx()

    # ======================================================
    # 6️⃣ Stochastic Oscillator
    # ======================================================
    stoch = ta.momentum.StochRSIIndicator(df_feat['Close'], window=14, smooth1=3, smooth2=3)
    df_feat['Stoch_O_k_value'] = stoch.stochrsi_k()
    df_feat['Stoch_O_k_smoothed'] = df_feat['Stoch_O_k_value'].rolling(3).mean()
    df_feat['Stoch_O_d_value'] = stoch.stochrsi_d()
    df_feat['Stoch_O_signal_value'] = df_feat['Stoch_O_k_value'] - df_feat['Stoch_O_d_value']
    df_feat['Stoch_O_signal'] = np.where(df_feat['Stoch_O_signal_value'] > 0, 'Buy', 'Sell')

    # ======================================================
    # 7️⃣ Pivot Points (Classic)
    # ======================================================
    df_feat['pivot_point'] = (df_feat['High'] + df_feat['Low'] + df_feat['Close']) / 3
    df_feat['pivot_support_1'] = 2 * df_feat['pivot_point'] - df_feat['High']
    df_feat['pivot_resistance_1'] = 2 * df_feat['pivot_point'] - df_feat['Low']
    df_feat['pivot_support_2'] = df_feat['pivot_point'] - (df_feat['High'] - df_feat['Low'])
    df_feat['pivot_resistance_2'] = df_feat['pivot_point'] + (df_feat['High'] - df_feat['Low'])
    df_feat['pivot_support_3'] = df_feat['pivot_point'] - 2 * (df_feat['High'] - df_feat['Low'])
    df_feat['pivot_resistance_3'] = df_feat['pivot_point'] + 2 * (df_feat['High'] - df_feat['Low'])

    # ======================================================
    # 8️⃣ Fibonacci Levels & Signal
    # ======================================================
    high_50 = df_feat['High'].rolling(window=50).max()
    low_50 = df_feat['Low'].rolling(window=50).min()
    diff = high_50 - low_50
    fib_236 = high_50 - diff * 0.236
    fib_382 = high_50 - diff * 0.382
    fib_618 = high_50 - diff * 0.618

    df_feat['fibonacci_signal'] = np.where(df_feat['Close'] > fib_236, 'Sell',
                                    np.where(df_feat['Close'] < fib_618, 'Buy', 'Hold'))

    # ======================================================
    # 9️⃣ VWAP (Relative)
    # ======================================================
    vwap_short = ta.volume.VolumeWeightedAveragePrice(
        df_feat['High'], df_feat['Low'], df_feat['Close'], df_feat['Volume'], window=20
    ).volume_weighted_average_price()

    vwap_long = ta.volume.VolumeWeightedAveragePrice(
        df_feat['High'], df_feat['Low'], df_feat['Close'], df_feat['Volume'], window=50
    ).volume_weighted_average_price()

    df_feat['VWAP_relative_short'] = df_feat['Close'] / vwap_short
    df_feat['VWAP_relative_long'] = df_feat['Close'] / vwap_long

    # ======================================================
    # 🔟 Bollinger Bands
    # ======================================================
    bb = ta.volatility.BollingerBands(df_feat['Close'], window=20, window_dev=2)
    df_feat['Bollinger_upper_band'] = bb.bollinger_hband()
    df_feat['Bollinger_lower_band'] = bb.bollinger_lband()
    df_feat['Bollinger_signal'] = np.where(
        df_feat['Close'] > df_feat['Bollinger_upper_band'], 'Sell',
        np.where(df_feat['Close'] < df_feat['Bollinger_lower_band'], 'Buy', 'Hold')
    )

    # ======================================================
    # 1️⃣1️⃣ Ichimoku Cloud
    # ======================================================
    ichi = ta.trend.IchimokuIndicator(df_feat['High'], df_feat['Low'], window1=9, window2=26, window3=52)
    df_feat['ichimoku_c_conversion_line'] = ichi.ichimoku_conversion_line()
    df_feat['ichimoku_c_base_line'] = ichi.ichimoku_base_line()
    df_feat['ichimoku_c_leading_span_a'] = ichi.ichimoku_a()
    df_feat['ichimoku_c_leading_span_b'] = ichi.ichimoku_b()
    df_feat['ichimoku_c_signal'] = np.where(
        df_feat['Close'] > df_feat['ichimoku_c_leading_span_a'], 'Buy',
        np.where(df_feat['Close'] < df_feat['ichimoku_c_leading_span_b'], 'Sell', 'Hold')
    )

    # ======================================================
    # 1️⃣2️⃣ Parabolic SAR (Relative)
    # ======================================================
    psar = ta.trend.PSARIndicator(df_feat['High'], df_feat['Low'], df_feat['Close'])
    df_feat['SAR_relative'] = df_feat['Close'] / psar.psar()

    # ======================================================
    # 1️⃣3️⃣ Market Proxy & Result
    # ======================================================
    df_feat['s&p_move_15m'] = df_feat['Close'].pct_change(periods=3) * 100
    df_feat['Next_Close'] = df_feat['Close'].shift(-1)
    df_feat['Return_%'] = (df_feat['Next_Close'] - df_feat['Close']) / df_feat['Close'] * 100

    df_feat['result'] = df_feat['Return_%'].apply(
        lambda x: 1 if x >= threshold else (0 if x <= -threshold else 2)
    )

    # ======================================================
    # 1️⃣4️⃣ Misc: Exchange + ID
    # ======================================================
    df_feat['exchange'] = 'BubbleStocks'
    df_feat['id'] = np.arange(len(df_feat)) + 100000
    
    
    df_feat.drop([
    'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close time', 'Quote asset volume', 'Number of trades',
    'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore',
    ], axis=1, inplace=True, errors='ignore')


    df_feat.drop(['RSI_signal','Stoch_O_signal','ichimoku_c_signal','fibonacci_signal','Bollinger_signal','exchange', 'Return_%' ],
                 axis=1, inplace=True , errors = 'ignore')
    
    
    scaler = StandardScaler()
    scaler.fit(df_feat)
    df_feat = pd.DataFrame(scaler.transform(df_feat), index=df_feat.index, columns=df_feat .columns)


    df_feat.drop([
    'SMA_14' , 'WMA_14', 'pivot_support_1' , 'pivot_resistance_1' ,'pivot_support_2' ,'pivot_resistance_2',
    'pivot_support_3', 'pivot_resistance_3', 'ichimoku_c_leading_span_a' ,'ichimoku_c_leading_span_b','ichimoku_c_base_line','Bollinger_lower_band',
    'ichimoku_c_conversion_line','EMA_relative','SMA_relative','Return_%'
    ], axis=1, inplace=True, errors='ignore')


    def create_lag(df_feat):
       for feature in df_feat.columns:
        new_column_name = f"{feature}_lag{1}"  
        df_feat[new_column_name] = df_feat[feature].shift(1)
        new_column_name2 = f"{feature}_lag{2}"  
        df_feat[new_column_name2] = df_feat[feature].shift(2)
        
    # Drop label columns that model shouldn't see during prediction
    drop_cols = ['result', 'result_lag1', 'result_lag2']
    df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], inplace=True, errors='ignore')
       
    create_lag(df_feat)
    
    # ======================================================
    # ✅ Clean up
    # ======================================================
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat.dropna(inplace=True)
    print("✅ All features generated successfully.")
    return df_feat


# ----------------------
# actual signal function (unchanged)
# ----------------------
def actual_signal_for_index(df, idx, threshold_percent=0.8):
    if idx + 1 >= len(df):
        return None
    cur = df.iloc[idx]["Close"]
    nxt = df.iloc[idx + 1]["Close"]
    pct = (nxt - cur) / cur * 100.0
    if pct >= threshold_percent:
        return 1
    elif pct <= -threshold_percent:
        return 0
    else:
        return 2

# ----------------------
# History management (unchanged)
# ----------------------
def append_history_row(ts, price, pred, actual):
    row = {"timestamp": pd.to_datetime(ts).isoformat(), "price": float(price), "prediction": int(pred) if pred is not None and pred!=-1 else None,
           "actual": int(actual) if actual is not None and actual!=-1 else None}
    if os.path.exists(HISTORY_CSV):
        hist = pd.read_csv(HISTORY_CSV)
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    cutoff = datetime.utcnow() - timedelta(hours=24)
    hist = hist[hist["timestamp"] >= cutoff]
    hist.to_csv(HISTORY_CSV, index=False)

def read_history():
    if os.path.exists(HISTORY_CSV):
        h = pd.read_csv(HISTORY_CSV)
        h["timestamp"] = pd.to_datetime(h["timestamp"])
        return h
    else:
        return pd.DataFrame(columns=["timestamp","price","prediction","actual"])

# ----------------------
# Global state & Initialization
# ----------------------
symbol_global = "BTCUSDT"

df_full = fetch_klines(symbol=symbol_global, interval=CANDLE_INTERVAL, limit=CANDLES_TO_FETCH)
if len(df_full) < 60:
    raise RuntimeError("Not enough data fetched from Binance. Check network or symbol.")

# Build features for full df at startup ONCE
df_features_full = build_features_for_df(df_full)

# playback indices - indexes relative to df_features_full since lag drops rows
# We will do playback only on last 24h (PLAYBACK_CANDLES)
playback_done = False
playback_index = 0

# Since features df is shorter due to lag, get last 24h slice index start
playback_start_idx = max(0, len(df_features_full) - PLAYBACK_CANDLES)
playback_end_idx = len(df_features_full) - 1
playback_index = playback_start_idx

play_lock = threading.Lock()

# trim history file to last 24h on startup
_ = read_history()

# ----------------------
# API: serve next step (playback or realtime update)
# ----------------------
@app.route("/api/next")
def api_next():
    global playback_index, playback_done, df_full, df_features_full, playback_start_idx, playback_end_idx
    global df_features_full

    with play_lock:
        img_b64 = ""
        acc = None

        if not playback_done:
            if playback_index > playback_end_idx:
                playback_done = True
            else:
                i = playback_index
                # Slice features df for playback window - slice is already last 24h
                # We only incrementally show playback_index from playback_start_idx to playback_end_idx

                # Select feature rows up to current playback_index (simulate incremental playback)
                window_features = df_features_full.iloc[playback_start_idx:i+1].reset_index(drop=True)

                if window_features.empty:
                    pred = None
                else:
                    try:
                        X = window_features.select_dtypes(include=[np.number]).drop(columns=['Close_price'], errors='ignore').iloc[[-1]].values
                        pred = int(model.predict(X)[0])
                    except Exception:
                        pred = None

                # actual signal aligned to raw data for the candle corresponding to the last row in window_features
                # Map feature index to raw df index by matching timestamps
                feature_last_time = window_features.iloc[-1]['Open time']
                raw_idx = df_full.index[df_full['Open time'] == feature_last_time]
                actual = None
                if len(raw_idx) > 0:
                    actual = actual_signal_for_index(df_full, raw_idx[0], threshold_percent=0.8)

                append_history_row(feature_last_time, window_features.iloc[-1]['Close_price'], pred if pred is not None else -1, actual if actual is not None else -1)

                playback_index += 1

        else:
            # REALTIME MODE: check for new candles and update full df + features full again
            try:
                latest = fetch_klines(symbol=symbol_global, interval=CANDLE_INTERVAL, limit=3)
                # new candle check
                if latest["Open time"].iloc[-1] > df_full["Open time"].iloc[-1]:
                    df_full = pd.concat([df_full, latest.tail(1)], ignore_index=True)
                    if len(df_full) > CANDLES_TO_FETCH:
                        df_full = df_full.tail(CANDLES_TO_FETCH).reset_index(drop=True)

                    # recompute features for full 30h window ONCE
                    global df_features_full
                    df_features_full = build_features_for_df(df_full)

                    # update playback window indices
                    playback_start_idx = max(0, len(df_features_full) - PLAYBACK_CANDLES)
                    playback_end_idx = len(df_features_full) - 1
                    playback_index = playback_end_idx

                    # predict for newest closed candle (last row)
                    if len(df_features_full) > 0:
                        try:
                            X = df_features_full.select_dtypes(include=[np.number]).drop(columns=['Close_price'], errors='ignore').iloc[[-1]].values
                            pred = int(model.predict(X)[0])
                        except Exception:
                            pred = None
                    else:
                        pred = None

                    # actual signal on raw df last but one candle (since last candle just closed)
                    actual = actual_signal_for_index(df_full, len(df_full)-2, threshold_percent=0.8)
                    append_history_row(df_full.iloc[-2]["Open time"], df_full.iloc[-2]["Close"], pred if pred is not None else -1, actual if actual is not None else -1)
                else:
                    pred = None
            except Exception:
                pred = None

        # Plotting history last 24h (last 96 rows)
        hist = read_history()
        plot_df = hist.copy()
        if plot_df.empty:
            img_b64 = ""
            step = playback_index
            total = playback_end_idx
            acc = None
        else:
            valid = plot_df[plot_df["actual"].notnull()]
            if len(valid) > 0:
                correct = (valid["prediction"] == valid["actual"]).sum()
                acc = round(correct / len(valid) * 100.0, 2)
            else:
                acc = None

            plot_df = plot_df.tail(PLAYBACK_CANDLES)

            fig, ax = plt.subplots(figsize=(12,5))
            plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])
            ax.plot(plot_df["timestamp"], plot_df["price"], color="cyan", label="Close")
            longs = plot_df[plot_df["prediction"] == 1]
            shorts = plot_df[plot_df["prediction"] == 0]
            holds = plot_df[plot_df["prediction"] == 2]
            if not longs.empty:
                ax.scatter(longs["timestamp"], longs["price"], color="green", marker="^", label="Pred LONG")
            if not shorts.empty:
                ax.scatter(shorts["timestamp"], shorts["price"], color="red", marker="v", label="Pred SHORT")
            if not holds.empty:
                ax.scatter(holds["timestamp"], holds["price"], color="orange", marker="o", label="Pred HOLD")
            actual_long = plot_df[plot_df["actual"] == 1]
            actual_short = plot_df[plot_df["actual"] == 0]
            if not actual_long.empty:
                ax.scatter(actual_long["timestamp"], actual_long["price"], facecolors='none', edgecolors='lime', s=80, linewidths=1.2, label="Actual LONG")
            if not actual_short.empty:
                ax.scatter(actual_short["timestamp"], actual_short["price"], facecolors='none', edgecolors='magenta', s=80, linewidths=1.2, label="Actual SHORT")
            ax.set_facecolor("#0d1117")
            fig.patch.set_facecolor("#0d1117")
            ax.tick_params(colors='white', which='both')
            ax.set_title("Playback / Live: BTCUSDT 15m (predictions are markers)", color='white')
            ax.legend(facecolor="#161b22", framealpha=0.9)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            step = int(playback_index)
            total = int(len(df_features_full)-1)

    return jsonify({
        "img": img_b64,
        "step": int(playback_index),
        "total": int(len(df_features_full)-1),
        "acc": acc,
        "playback_done": bool(playback_done),
        "history_count": int(len(read_history()))
    })

# ----------------------
# Root route returns HTML page with embedded updater JS
# ----------------------
@app.route("/")
def index():
    return render_template("dashboard.html",
                           playback_delay_ms=int(PLAYBACK_DELAY_SECONDS*1000),
                           realtime_poll_ms=REALTIME_POLL_MS)

# ----------------------
# simple API to get history table as rows
# ----------------------
@app.route("/api/history")
def api_history():
    h = read_history()
    if not h.empty:
        h["timestamp"] = h["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(h.tail(200).to_dict(orient="records"))

# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    _ = read_history()
    print("Starting app. Initial candles fetched:", len(df_full))
    app.run(debug=True, port=5000)
