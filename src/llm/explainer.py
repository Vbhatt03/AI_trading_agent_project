import requests
from src.llm.state_formatter import format_state_for_llm, action_to_text
from src.rag.memory_store import memory


def local_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def explain_trade(obs, action):
    state_text, _ = format_state_for_llm(obs)
    action_text = action_to_text(action)
    retrieved = memory.retrieve(state_text, k=2)
    memory_context = "\n".join(retrieved)
    # price = float(obs[0])
    rsi = float(obs[1])
    short_ma = float(obs[2])
    long_ma = float(obs[3])
    momentum = float(obs[4])
    macd = float(obs[5])
    vol = float(obs[6])
    volume = float(obs[8])
    trend = "Uptrend" if short_ma > long_ma else "Downtrend"

    prompt = f"""
You are evaluating whether the trading agent's action is ALIGNED or in CONFLICT with market conditions.

You MUST follow this reasoning order:

1) FIRST examine the PAST SIMILAR HISTORICAL CASES below.
   These are actual past outcomes from this trading system.
   Treat them as empirical evidence.

2) Determine if current market conditions resemble any past cases.

3) If past cases show negative outcomes after similar actions, this is strong evidence of CONFLICT.
   If past cases show positive outcomes, this is evidence of ALIGNED.

4) ONLY AFTER using historical evidence, use technical indicators (RSI, MACD, trend, etc.) as secondary confirmation.

PAST SIMILAR HISTORICAL CASES:
{memory_context}

CURRENT MARKET STATE:
RSI: {rsi}
MACD: {macd}
Trend: {trend}
Volatility: {vol}
Short-term Momentum: {momentum}
Volume Change: {volume}

AGENT ACTION:
{action}

Respond strictly in this format:

Verdict: ALIGNED or CONFLICT  
Reasoning: Explain primarily using past case outcomes, then indicators.  
Risk: Brief risk assessment.
"""



    explanation = local_llm(prompt)

    return explanation
