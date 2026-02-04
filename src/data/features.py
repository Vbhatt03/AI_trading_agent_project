import pandas as pd
import ta
import numpy as np
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent

df = pd.read_csv(script_dir / "spy.csv", skiprows=[1, 2])  # Skip the ticker and empty date rows

# Ensure Close column is numeric
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
df["ma20"] = df["Close"].rolling(20).mean()
df["ma50"] = df["Close"].rolling(50).mean()
df["returns"] = df["Close"].pct_change()
macd = ta.trend.MACD(df["Close"])
df["macd"] = macd.macd()
df["volatility"] = df["returns"].rolling(20).std()
df["momentum"] = df["Close"] - df["Close"].shift(10)
df["volume_change"] = df["Volume"].pct_change()
df.dropna(inplace=True)

df.to_csv(script_dir / "extracted_data" / "spy_features.csv", index=False)
print("Features created")