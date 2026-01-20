import pandas as pd
import ta

df = pd.read_csv("aapl.csv", skiprows=[1, 2])  # Skip the ticker and empty date rows

# Ensure Close column is numeric
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
df["ma20"] = df["Close"].rolling(20).mean()
df["ma50"] = df["Close"].rolling(50).mean()
df["returns"] = df["Close"].pct_change()

df.dropna(inplace=True)

df.to_csv("extracted_data/aapl_features.csv")
print("Features created")
