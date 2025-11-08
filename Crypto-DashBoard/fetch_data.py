# fetch_data.py
import pandas as pd
import requests
import datetime as dt

def get_binance_data(symbol="BTCUSDT", interval="15m", limit=3000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base", "Taker buy quote", "Ignore"
    ])
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    df = df[["Open time", "Open", "High", "Low", "Close", "Volume"]]
    return df

if __name__ == "__main__":
    df = get_binance_data()
    df.to_csv("data.csv", index=False)
    print(f"✅ Saved {len(df)} candles to data.csv ({df['Open time'].min()} → {df['Open time'].max()})")
