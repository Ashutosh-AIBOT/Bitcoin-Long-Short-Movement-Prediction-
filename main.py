from flask import Flask, render_template
import pandas as pd
import numpy as np
import requests
import ta
import joblib
import io
import matplotlib
matplotlib.use('Agg')  # no GUI backend
import matplotlib.pyplot as plt
import base64
from sklearn.preprocessing import StandardScaler
# -----------------------------
# 🔹 Flask App Setup
# -----------------------------
app = Flask(__name__)
model = joblib.load("rf_model01.pkl")


# -----------------------------
# 🔹 Fetch 15-min Binance data
# -----------------------------
def get_binance_data(symbol="BTCUSDT", interval="1m", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()

    df = pd.DataFrame(data, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])

    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    return df


# ================================
# ⚙️ Feature Generator
# ================================
def generate_features(df, threshold = 0.8 ):
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



# -----------------------------
# 🔹 Prediction + Chart
# -----------------------------
def make_prediction():
    df = get_binance_data()
    df_feat = generate_features(df)
    X = df_feat.select_dtypes(include=[np.number])

    prediction = model.predict(X.tail(1))[0]
    direction = "LONG 🟢" if prediction == 1 else "SHORT 🔴" if prediction == 0 else "HOLD ⚪"

    # Plot chart
    plt.figure(figsize=(8, 4))
    plt.plot(df['Open time'], df['Close'], label='Close Price', color='blue')
    plt.title(f"BTCUSDT 15m Chart — Signal: {direction}")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return direction, chart_url


# -----------------------------
# 🔹 Flask Route
# -----------------------------
@app.route('/')
def dashboard():
    direction, chart = make_prediction()
    return render_template('dashboard.html', signal=direction, chart=chart)


# -----------------------------
# 🔹 Run Flask App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
