import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.current_step = 0

        # 0 Hold | 1–3 Buy | 4–6 Sell
        self.action_space = spaces.Discrete(7)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Portfolio
        self.initial_cash = 10000
        self.cash = self.initial_cash
        self.shares = 0.0
        self.net_worth = self.cash

    # ---------------- OBSERVATION ----------------
    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        price = row["Close"]

        # Portfolio exposure % (normalized)
        exposure = (self.shares * price) / max(self.net_worth, 1e-6)

        obs = np.array([
            price,
            row["rsi"],
            row["ma20"],
            row["ma50"],
            row["returns"],
            row["macd"],
            row["volatility"],
            row["momentum"],
            row["volume_change"],
            exposure
        ], dtype=np.float32)

        return obs

    # ---------------- STEP ----------------
    def step(self, action):
        price = self.df.iloc[self.current_step]["Close"]
        prev_worth = self.net_worth

        # Map action → fraction of portfolio to trade
        trade_fraction = 0
        if action == 1: trade_fraction = 0.25
        elif action == 2: trade_fraction = 0.50
        elif action == 3: trade_fraction = 1.0
        elif action == 4: trade_fraction = -0.25
        elif action == 5: trade_fraction = -0.50
        elif action == 6: trade_fraction = -1.0

        trade_value = trade_fraction * prev_worth
        cost_rate = 0.001

        # -------- BUY --------
        if trade_value > 0:
            buy_value = min(trade_value, self.cash)
            if buy_value > 0:
                cost = buy_value * cost_rate
                shares_bought = (buy_value - cost) / price
                self.cash -= buy_value
                self.shares += shares_bought

        # -------- SELL --------
        elif trade_value < 0:
            max_sell_value = self.shares * price
            sell_value = min(abs(trade_value), max_sell_value)
            if sell_value > 0:
                cost = sell_value * cost_rate
                shares_sold = sell_value / price
                self.cash += (sell_value - cost)
                self.shares -= shares_sold

        # Update portfolio
        self.net_worth = self.cash + self.shares * price

        # Reward = portfolio return
        reward = (self.net_worth - prev_worth) / max(prev_worth, 1e-6)

        # Small penalty to stop overtrading
        reward -= 0.0001 * abs(trade_fraction)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._next_observation(), reward, done, False, {}

    # ---------------- RESET ----------------
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.net_worth = self.cash
        return self._next_observation(), {}
