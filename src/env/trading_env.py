import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.current_step = 0

        # Action space: 5 discrete target exposures
        # Action 0: Target 0% (exit all)
        # Action 1: Target 25%
        # Action 2: Target 50%
        # Action 3: Target 75%
        # Action 4: Target 100% (fully invested)
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Portfolio
        self.initial_cash = 10000
        self.cash = self.initial_cash
        self.shares = 0.0
        self.net_worth = self.cash
        
        # Track action history for reversal penalties
        self.last_action = 0
        self.steps_since_action_change = 0

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

    # -------- STEP ----------------
    def step(self, action, conflict_penalty=0.0):
        price = self.df.iloc[self.current_step]["Close"]
        prev_worth = self.net_worth
        current_exposure = (self.shares * price) / max(self.net_worth, 1e-6)

        # Map action → target exposure (declarative, not delta)
        target_exposures = [0.0, 0.25, 0.50, 0.75, 1.0]
        target_exposure = target_exposures[action]

        # Calculate required trade delta to reach target
        target_value = target_exposure * prev_worth
        current_value = self.shares * price
        trade_delta = target_value - current_value

        cost_rate = 0.001

        # Execute trade to reach target exposure
        if trade_delta > 0:  # BUY
            buy_amount = min(trade_delta, self.cash)
            if buy_amount > 0:
                cost = buy_amount * cost_rate
                shares_bought = (buy_amount - cost) / price
                self.cash -= buy_amount
                self.shares += shares_bought

        elif trade_delta < 0:  # SELL
            max_sell_value = self.shares * price
            sell_amount = min(abs(trade_delta), max_sell_value)
            if sell_amount > 0:
                cost = sell_amount * cost_rate
                shares_sold = sell_amount / price
                self.cash += (sell_amount - cost)
                self.shares -= shares_sold

        # Update portfolio
        self.net_worth = self.cash + self.shares * price

        # Reward = portfolio return
        reward = (self.net_worth - prev_worth) / max(prev_worth, 1e-6)

        # Apply LLM conflict penalty if provided
        if conflict_penalty > 0:
            reward -= conflict_penalty * abs(reward)

        # Penalize rapid reversals (Buy→Sell or Sell→Buy in short window)
        is_buy_action = action > 1  # Target > 25%
        was_buy_action = self.last_action > 1
        is_reversal = (is_buy_action != was_buy_action) and (self.last_action != action)
        
        if is_reversal and self.steps_since_action_change < 10:
            reversal_penalty = 0.01 * (1 - self.steps_since_action_change / 10)
            reward -= reversal_penalty

        # Momentum continuation bonus: reward staying invested during positive returns
        # Encourages trend-following and reduces early exits
        if action == self.last_action and action > 1 and reward > 0:
            momentum_bonus = 0.0002 * reward  # Small % of positive reward
            reward += momentum_bonus
        
        # Penalize exiting (0%) during strong positive momentum
        momentum = float(self.df.iloc[self.current_step]["momentum"])
        macd = float(self.df.iloc[self.current_step]["macd"])
        if action == 0 and momentum > 0.01 and macd > 0.5:  # Exiting during strong bullish signal
            early_exit_penalty = 0.003
            reward -= early_exit_penalty

        # Update action tracking
        if action != self.last_action:
            self.steps_since_action_change = 0
            self.last_action = action
        else:
            self.steps_since_action_change += 1

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
