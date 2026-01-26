import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.current_step = 0

        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.balance = 10000
        self.position = 0
        self.net_worth = self.balance

    def _next_observation(self):
        row = self.df.iloc[self.current_step]

        obs = np.array([
            row["Close"],
            row["rsi"],
            row["ma20"],
            row["ma50"],
            row["returns"],
            row["macd"],
            row["volatility"],
            row["momentum"],
            row["volume_change"],
            self.position
        ], dtype=np.float32)

        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["Close"]

        trade_fraction = 0
        if action == 1:
            trade_fraction = 0.25
        elif action == 2:
            trade_fraction = 0.50
        elif action == 3:
            trade_fraction = 1.0
        elif action == 4:
            trade_fraction = -0.25
        elif action == 5:
            trade_fraction = -0.50
        elif action == 6:
            trade_fraction = -1.0

        position_change = trade_fraction

        transaction_cost = 0.001 * abs(position_change) * current_price

        self.position += position_change
        self.position = max(min(self.position, 1), 0)

        prev_worth = self.net_worth

        self.net_worth = self.balance + self.position * current_price - transaction_cost

        reward = 100 * (self.net_worth - prev_worth) / prev_worth



        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._next_observation(), reward, done, False, {}


    def reset(self, seed=None):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.net_worth = self.balance
        return self._next_observation(), {}
