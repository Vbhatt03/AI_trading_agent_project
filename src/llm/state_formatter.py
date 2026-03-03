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
        0: "Target 0% exposure (exit all)",
        1: "Target 25% exposure",
        2: "Target 50% exposure",
        3: "Target 75% exposure",
        4: "Target 100% exposure (fully invested)"
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
