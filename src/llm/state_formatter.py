def format_state_for_llm(obs):
    state = {
        "price": float(obs[0]),
        "rsi": float(obs[1]),
        "ma20": float(obs[2]),
        "ma50": float(obs[3]),
        "returns": float(obs[4]),
        "macd": float(obs[5]),
        "volatility": float(obs[6]),
        "momentum": float(obs[7]),
        "volume_change": float(obs[8]),
        "position": float(obs[9]),
    }

    trend = "uptrend" if state["ma20"] > state["ma50"] else "downtrend"
    momentum_dir = "positive" if state["momentum"] > 0 else "negative"
    macd_signal = "bullish" if state["macd"] > 0 else "bearish"

    description = f"""
Market Indicators:
- Current Price: {state['price']:.2f}
- Trend: {trend}
- RSI: {state['rsi']:.2f}
- MACD: {state['macd']:.3f} ({macd_signal})
- Volatility: {state['volatility']:.4f}
- Short-term Momentum: {state['momentum']:.2f} ({momentum_dir})
- Volume Change: {state['volume_change']:.2%}
- Current Position Size: {state['position']:.2f} (0 = cash, 1 = fully invested)
"""

    return description, state
def action_to_text(action):
    mapping = {
        0: "Hold position",
        1: "Buy 25% more",
        2: "Buy 50% more",
        3: "Buy 100% (full allocation)",
        4: "Sell 25%",
        5: "Sell 50%",
        6: "Sell all holdings"
    }
    # Ensure action is hashable (convert numpy array to int or tuple)
    try:
        import numpy as np
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action = int(action.item())
            else:
                action = tuple(action.tolist())
    except ImportError:
        pass
    except Exception:
        pass
    return mapping.get(action, "Unknown action")
