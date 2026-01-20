import sys
from pathlib import Path
import numpy as np
# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from stable_baselines3 import PPO
from src.env.trading_env import TradingEnv

df = pd.read_csv(project_root / "src" / "data" / "extracted_data" / "aapl_features.csv")

env = TradingEnv(df)

model = PPO.load(project_root / "src" / "agents" / "ppo_trading")

obs, _ = env.reset()

total_reward = 0

portfolio_values = []

prices = df["Close"].values


while True:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    portfolio_values.append(env.net_worth)
    total_reward += reward
    
    if done:
        break
portfolio_values = np.array(portfolio_values)

daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
print("Sharpe ratio:", sharpe_ratio)

final_value = portfolio_values[-1]
total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
print("Total return:", total_return)
running_max = np.maximum.accumulate(portfolio_values)
drawdowns = (portfolio_values - running_max) / running_max

max_drawdown = drawdowns.min()
print("Max drawdown:", max_drawdown)

buy_hold_return = (prices[-1] - prices[0]) / prices[0]
print("Buy and Hold return:", buy_hold_return)


