import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from stable_baselines3 import PPO
from src.env.trading_env import TradingEnv

df = pd.read_csv(project_root / "src" / "data" / "extracted_data" / "aapl_features.csv", index_col=0)
df = df.rename(columns={"Price": "Date"})

train = df[df["Date"] < "2022-01-01"]
test = df[df["Date"] >= "2022-01-01"]

train_env = TradingEnv(train)
test_env = TradingEnv(test)

model = PPO("MlpPolicy", train_env, verbose=1)

model.learn(total_timesteps=100000)

model.save(project_root / "src" / "agents" / "ppo_trading")

print("Training complete")
