import yfinance as yf
import pandas as pd

spy = yf.download("SPY", start="2010-01-01", end="2024-01-01")

spy = spy[["Close", "Volume"]]
spy.dropna(inplace=True)
spy.to_csv("src/data/spy.csv")