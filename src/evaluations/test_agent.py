import sys
from pathlib import Path
import numpy as np
# For reproducibility
import random

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from src.rag.memory_store import MemoryStore

memory = MemoryStore()

import pandas as pd
from stable_baselines3 import PPO
from src.env.trading_env import TradingEnv

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
from src.llm.explainer import action_to_text, explain_trade
df = pd.read_csv(project_root / "src" / "data" / "extracted_data" / "aapl_features.csv")

env = TradingEnv(df)
obs, _ = env.reset(seed=42)

model = PPO.load(project_root / "src" / "agents" / "ppo_trading")

total_reward = 0

portfolio_values = []

prices = df["Close"].values
reward_buffer = []
step_count=0
volatility = float(obs[6])
while True:
    action, _ = model.predict(obs)
    action_name = action_to_text(action)
    # Only call LLM every 200 steps OR when action is not hold
    if step_count % 200 == 0:
        explanation = explain_trade(obs, int(action))
        print("\n--- LLM Reasoning ---")
        print(explanation)
    obs, reward, done, _, _ = env.step(action)
    trend = "Uptrend" if obs[2] > obs[3] else "Downtrend"

    reward_buffer.append(reward)
    step_count += 1

    if step_count % 20 == 0 and len(reward_buffer) >= 20:
        WINDOW = 50
        future_return = sum(reward_buffer[-WINDOW:])
        vol = max(volatility, 1e-6)
        signal_strength = future_return / vol
        print("YOUR RL AGENT DECIDED TO: " + action_name)
        is_hold = action_name.lower() == "hold position"
        if is_hold:
            if signal_strength < -1:
                outcome_desc = "Holding avoided a drawdown that followed"
            elif signal_strength > 1:
                outcome_desc = "Holding missed a profitable upward move"
            else:
                outcome_desc = "Holding preserved capital in a sideways market"

        else:
            if signal_strength > 5:
                outcome_desc = "Strong bullish follow-through occurred after the action"
            elif signal_strength > 1:
                outcome_desc = "Moderate positive continuation followed"
            elif signal_strength > -1:
                outcome_desc = "Sideways / noisy movement followed"
            elif signal_strength > -5:
                outcome_desc = "Bearish pullback followed the action"
            else:
                outcome_desc = "Sharp adverse move / drawdown followed"


        summary = f"""
        Market Regime:
        {trend}, RSI {obs[1]:.1f}, MACD {obs[5]:.2f}, Volatility {obs[6]:.3f}.

        Agent Action:
        {action_name}

        Outcome:
        {outcome_desc}. 20-step return = {future_return:.4f}.
        """

        memory.add(summary)
        print(f"[MEMORY SIZE]: {len(memory.memories)}")
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


