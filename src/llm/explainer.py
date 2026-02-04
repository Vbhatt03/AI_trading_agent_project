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
    state_text, current_state_summary = format_state_for_llm(obs)
    action_text = action_to_text(action)
    relevant_memories = memory.retrieve(state_text, k=5)
    
    # If not enough memories, skip LLM call
    if len(relevant_memories) < 3:
        return "Insufficient historical data. Building memory..."
    
    # Number and format memories for explicit citation
    memory_context = ""
    for i, mem in enumerate(relevant_memories, 1):
        memory_context += f"[Case {i}]\n{mem}\n\n"
    
    rsi = float(obs[1])
    short_ma = float(obs[2])
    long_ma = float(obs[3])
    momentum = float(obs[4])
    macd = float(obs[5])
    vol = float(obs[6])
    volume = float(obs[8])
    exposure = float(obs[9])
    trend = "Uptrend" if short_ma > long_ma else "Downtrend"

    prompt = f"""
CRITICAL INSTRUCTIONS:
- You have {len(relevant_memories)} historical cases below
- You MUST cite specific cases by number [Case 1], [Case 2], etc.
- DO NOT make up case names or references
- DO NOT use generic trading knowledge
- If you cannot find similar cases, say "No similar historical cases"

=== HISTORICAL CASES ===
{memory_context}

=== CURRENT STATE ===
RSI {rsi:.1f}, MACD {macd:.2f}, {trend}, Vol {vol:.3f}, Momentum {momentum:.2f}, Exposure {exposure:.2f}

ACTION: {action_text}

RESPOND EXACTLY:
Verdict: ALIGNED or CONFLICT
Similar Cases: List case numbers that match current state
Reasoning: Compare action/outcome from those specific cases
Risk: High/Medium/Low based on case outcomes
"""



    explanation = local_llm(prompt)

    return explanation
