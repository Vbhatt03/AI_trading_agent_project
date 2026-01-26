import requests
from src.llm.state_formatter import format_state_for_llm, action_to_text


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


    prompt = f"""
You are a quantitative trading analyst evaluating an AI trading agent's action.

{state_text}

Action taken: {action_text}

Follow these instructions exactly:

1. Decide if the action is ALIGNED or CONFLICT with the indicators.
2. Use ONLY the provided indicators.
3. Be analytical, neutral, and risk-aware.
4. Maximum 3 short sentences in reasoning.
5. Do not add extra formatting or markdown.

Output must be EXACTLY in this format:

Verdict: ALIGNED or CONFLICT
Reasoning: <2-3concise analytical sentences>
Risk: <one short risk sentence>
"""



    explanation = local_llm(prompt)

    return explanation
