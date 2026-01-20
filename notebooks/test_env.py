import pandas as pd
from src.env.trading_env import TradingEnv

df = pd.read_csv("data/aapl_features.csv")

env = TradingEnv(df)

obs, _ = env.reset()

print("Initial observation:", obs)

for _ in range(10):
    obs, reward, done, _, _ = env.step(0)
    print("Reward:", reward)

print("Environment works")
