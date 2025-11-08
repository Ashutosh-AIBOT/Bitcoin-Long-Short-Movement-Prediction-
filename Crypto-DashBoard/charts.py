import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Convert figure to base64
# -----------------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return img_base64


# -----------------------------------------------------------------
# 1️⃣ Price Chart
# -----------------------------------------------------------------
def plot_price_chart(df, prediction_label):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Open time'], df['Close'], color='cyan', label="BTCUSDT Close")
    ax.set_title(f"BTCUSDT 15m Chart — Last Prediction: {prediction_label}", color='white')
    ax.legend()
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors='white')
    return fig_to_base64(fig)


# -----------------------------------------------------------------
# 2️⃣ Model Accuracy Chart
# -----------------------------------------------------------------
def plot_accuracy_chart(history_df):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history_df.index, history_df['accuracy'], color='lime', marker='o')
    ax.set_title("Model Accuracy (Static)", color='white')
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors='white')
    return fig_to_base64(fig)


# -----------------------------------------------------------------
# 3️⃣ Prediction Distribution
# -----------------------------------------------------------------
def plot_signal_distribution(history_df):
    counts = history_df['prediction'].value_counts().sort_index()
    labels = {0: 'SHORT', 1: 'LONG', 2: 'HOLD'}
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar([labels.get(x, x) for x in counts.index], counts.values, color=['red', 'green', 'orange'])
    ax.set_title("Prediction Distribution", color='white')
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors='white')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', color='white')
    return fig_to_base64(fig)


# -----------------------------------------------------------------
# 4️⃣ Volume Chart
# -----------------------------------------------------------------
def plot_volume_chart(df):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(df['Open time'], df['Volume'], color='dodgerblue')
    ax.set_title("Trading Volume", color='white')
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors='white')
    return fig_to_base64(fig)


# -----------------------------------------------------------------
# 5️⃣ Moving Averages (SMA/EMA)
# -----------------------------------------------------------------
def plot_moving_averages(df):
    df = df.copy()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Open time'], df['Close'], color='white', label="Close")
    ax.plot(df['Open time'], df['SMA_50'], color='orange', label='SMA 50')
    ax.plot(df['Open time'], df['EMA_20'], color='magenta', label='EMA 20')
    ax.set_title("Moving Averages", color='white')
    ax.legend()
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors='white')
    return fig_to_base64(fig)


# -----------------------------------------------------------------
# 6️⃣ RSI (Relative Strength Index)
# -----------------------------------------------------------------
def plot_rsi(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df['Open time'], df['RSI'], color='yellow', label="RSI (14)")
    ax.axhline(70, color='red', linestyle='--', alpha=0.7)
    ax.axhline(30, color='green', linestyle='--', alpha=0.7)
    ax.set_title("Relative Strength Index (RSI)", color='white')
    ax.legend()
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors='white')
    return fig_to_base64(fig)


# -----------------------------------------------------------------
# 7️⃣ Candle Volatility (High-Low difference)
# -----------------------------------------------------------------
def plot_volatility(df):
    df = df.copy()
    df['Volatility'] = df['High'] - df['Low']

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df['Open time'], df['Volatility'], color='deepskyblue')
    ax.set_title("Candle Volatility (High - Low)", color='white')
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors='white')
    return fig_to_base64(fig)


def plot_volume_chart(df):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(df['Open time'], df['Volume'], color='deepskyblue')
    ax.set_title("Trading Volume", color='white')
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors='white')
    return fig_to_base64(fig)

def plot_moving_averages(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Open time'], df['Close'], color='cyan', label="Close")
    ax.plot(df['Open time'], df['SMA_20'], color='orange', label="SMA 20")
    ax.plot(df['Open time'], df['SMA_50'], color='magenta', label="SMA 50")
    ax.plot(df['Open time'], df['EMA_20'], color='lime', linestyle='--', label="EMA 20")
    ax.set_title("Moving Averages", color='white')
    ax.legend()
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors='white')
    return fig_to_base64(fig)

def plot_volatility_chart(df):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df['Open time'], df['rolling_vol'], color='yellow', label="Volatility (%)")
    ax.set_title("Market Volatility (Rolling 30)", color='white')
    ax.legend()
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.tick_params(colors='white')
    return fig_to_base64(fig)
