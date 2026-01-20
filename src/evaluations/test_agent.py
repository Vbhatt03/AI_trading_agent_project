import sys
from pathlib import Path

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

while True:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

    if done:
        break

print("Total reward:", total_reward)
