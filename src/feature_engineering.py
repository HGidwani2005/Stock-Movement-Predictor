import os
import pandas as pd
import ta

PROCESSED_DIR = "data/processed"

files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]

for file in files:
    print(f"Creating features for {file}...")

    path = os.path.join(PROCESSED_DIR, file)
    df = pd.read_csv(path)

    # Daily Return
    df["Return"] = df["Close"].pct_change()

    # Moving Averages
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    # Exponential Moving Average
    df["EMA10"] = df["Close"].ewm(span=10).mean()

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    # Volatility
    df["Volatility"] = df["Return"].rolling(10).std()

    # Target (1 = tomorrow's price goes up)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Remove rows with NaN values
    df.dropna(inplace=True)

    # Save back
    df.to_csv(path, index=False)

    print(f"Saved {file}")

print("\nFeature engineering completed!")