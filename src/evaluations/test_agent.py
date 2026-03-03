import sys
from pathlib import Path
import numpy as np
# For reproducibility
import random

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from src.rag.memory_store import memory
from src.llm.state_formatter import format_state_for_llm, action_to_text

import pandas as pd
from stable_baselines3 import PPO
from src.env.trading_env import TradingEnv

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
from src.llm.explainer import action_to_text, explain_trade
df = pd.read_csv(project_root / "src" / "data" / "extracted_data" / "spy_features.csv")

env = TradingEnv(df)
obs, _ = env.reset(seed=42)

model = PPO.load(project_root / "src" / "agents" / "ppo_trading")

memory.clear()

model_action_n = getattr(model.action_space, "n", None)
env_action_n = env.action_space.n
if model_action_n is not None and model_action_n != env_action_n:
    print(
        f"[WARN] Model action space ({model_action_n}) != env action space ({env_action_n}). "
        "Applying safe clipping for compatibility."
    )

total_reward = 0

portfolio_values = []

prices = df["Close"].values
reward_buffer = []
step_count=0
volatility = float(obs[6])
while True:
    raw_action, _ = model.predict(obs)
    if isinstance(raw_action, np.ndarray):
        action = int(raw_action.item()) if raw_action.size == 1 else int(raw_action.flat[0])
    else:
        action = int(raw_action)

    if action < 0 or action >= env_action_n:
        clipped_action = int(np.clip(action, 0, env_action_n - 1))
        print(f"[WARN] Invalid action {action} clipped to {clipped_action}")
        action = clipped_action

    action_name = action_to_text(action)
    
    # Step without conflict penalty (evaluate pure model performance)
    obs, reward, done, _, _ = env.step(action, conflict_penalty=0.0)
    trend = "Uptrend" if obs[2] > obs[3] else "Downtrend"

    reward_buffer.append(reward)
    step_count += 1
    
    # Call LLM every 100 steps for monitoring/analysis only
    if step_count % 100 == 0 and len(memory.memories) >= 10:
        explanation = explain_trade(obs, action)
        print("\n--- LLM Reasoning (Analysis Only) ---")
        print(explanation)

    if step_count % 20 == 0 and len(reward_buffer) >= 20:
        WINDOW = 50
        # Calculate true portfolio return instead of summing distorted rewards
        portfolio_values.append(env.net_worth)
        if len(portfolio_values) >= WINDOW:
            true_return = (portfolio_values[-1] - portfolio_values[-WINDOW]) / max(portfolio_values[-WINDOW], 1e-6)
        else:
            true_return = (env.net_worth - portfolio_values[0]) / max(portfolio_values[0], 1e-6)
        
        vol = max(volatility, 1e-6)
        signal_strength = true_return / vol if vol > 1e-6 else 0
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


        rsi = obs[1]
        vol = obs[6]
        exposure = obs[9]

        if rsi > 65 and trend == "Uptrend":
            regime = "Overbought uptrend"
        elif rsi < 35 and trend == "Downtrend":
            regime = "Oversold downtrend"
        elif vol > 0.03:
            regime = "High volatility"
        else:
            regime = "Neutral regime"

        # Get state description for better memory retrieval matching
        state_text, _ = format_state_for_llm(obs)
        
        summary = f"""{state_text}
Agent Action:
{action_name}

Outcome:
{outcome_desc}. {WINDOW}-step return = {true_return:.4f}.
"""


        memory.add(summary)
        print(f"[MEMORY SIZE]: {len(memory.memories)}")
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


