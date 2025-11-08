from flask import Flask, render_template
import pandas as pd
import numpy as np
from charts import (
    plot_price_chart,
    plot_accuracy_chart,
    plot_signal_distribution,
    plot_volume_chart,
    plot_moving_averages,
    plot_volatility_chart
)

app = Flask(__name__)

# 🔹 Load Binance data (you’ll later connect live API)
df = pd.read_csv("data.csv")
df["Open time"] = pd.to_datetime(df["Open time"])

# 🔹 Feature Engineering
df["rolling_mean"] = df["Close"].rolling(window=30).mean()
df["rolling_std"] = df["Close"].rolling(window=30).std()
df["rolling_vol"] = df["Close"].pct_change().rolling(window=30).std() * 100
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
df.dropna(inplace=True)

# 🔹 Temporary Dummy predictions (replace later with your ML model)
df["prediction"] = np.random.choice([0, 1, 2], len(df))  # 0=SHORT, 1=LONG, 2=HOLD
df["accuracy"] = np.random.uniform(0.6, 1.0, len(df))

# 🔹 Filter last 200 candles (≈ 50 hours)
df = df.tail(200)

@app.route('/')
def dashboard():
    label_map = {0: 'SHORT', 1: 'LONG', 2: 'HOLD'}
    latest_pred = df["prediction"].iloc[-1]
    prediction_label = label_map.get(latest_pred, "N/A")

    # Main charts
    price_chart = plot_price_chart(df, prediction_label)
    accuracy_chart = plot_accuracy_chart(df)
    dist_chart = plot_signal_distribution(df)

    # Extra charts
    volume_chart = plot_volume_chart(df)
    ma_chart = plot_moving_averages(df)
    volatility_chart = plot_volatility_chart(df)

    return render_template(
        "dashboard.html",
        price_chart=price_chart,
        accuracy_chart=accuracy_chart,
        dist_chart=dist_chart,
        volume_chart=volume_chart,
        ma_chart=ma_chart,
        volatility_chart=volatility_chart,
        prediction_label=prediction_label
    )

if __name__ == "__main__":
    app.run(debug=True)
