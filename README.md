# **AI Trading Agent using Reinforcement Learning**

A reinforcement learning-based trading system that learns dynamic portfolio allocation strategies from historical market data. The project combines a custom-built trading environment with deep RL (PPO) to simulate and optimize decision-making under realistic market constraints.

---

## **Overview**

This project focuses on building an intelligent trading agent capable of learning optimal strategies through interaction with a simulated financial market. Instead of making simple buy/sell decisions, the agent adjusts **portfolio exposure levels**, enabling more structured and realistic control over investments.

The system incorporates **feature-rich market data**, realistic trading constraints, and behavior-shaping mechanisms to improve learning stability and performance.

---

## **Key Features**

* **Custom Trading Environment**

  * Built using Gymnasium for flexible RL experimentation
  * Models portfolio dynamics, transaction costs, and exposure adjustments

* **Reinforcement Learning Agent**

  * Implemented using **Proximal Policy Optimization (PPO)** via Stable-Baselines3
  * Learns optimal strategies through continuous interaction with the environment

* **Exposure-Based Action Space**

  * Actions represent **target portfolio allocation levels** (0% to 100%)
  * Enables interpretable and structured decision-making

* **Feature-Rich State Representation**

  * Incorporates multiple technical indicators:

    * RSI, Moving Averages (MA20, MA50)
    * MACD, volatility, momentum
    * Returns and volume-based features

* **Reward Design**

  * Based on **portfolio net worth growth**
  * Enhanced with:

    * penalties for frequent position reversals
    * incentives for momentum-following behavior
    * penalties for premature exit during strong trends

* **Modular System Design**

  * Clear separation of environment, training, evaluation, and data pipelines
  * Easily extensible for experimentation and research

---

## **Project Structure**

```
.
├─ notebooks/              # Experimentation and analysis
├─ requirements.txt
└─ src/
   ├─ data/                # Processed datasets
   ├─ env/
   │  └─ trading_env.py    # Custom RL environment
   ├─ evaluations/         # Performance analysis
   ├─ llm/                 # LLM-based modules (optional)
   ├─ rag/                 # Retrieval-augmented components
   └─ training/
      └─ train.py          # PPO training pipeline
```

---

## **Technical Approach**

### **Action Space**

The agent operates on a discrete action space representing target exposure:

| Action | Exposure |
| ------ | -------- |
| 0      | 0%       |
| 1      | 25%      |
| 2      | 50%      |
| 3      | 75%      |
| 4      | 100%     |

---

### **State Representation**

The environment provides a structured feature vector including:

* Price data (Close)
* Technical indicators (RSI, MA20, MA50, MACD)
* Market dynamics (returns, volatility, momentum)
* Volume-based signals
* Current portfolio exposure

---

### **Reward Function**

The agent is trained to maximize portfolio returns:

```
(net_worth_t - net_worth_{t-1}) / net_worth_{t-1}
```

Additional reward shaping includes:

* Penalizing excessive trading (reducing noise)
* Encouraging trend-following behavior
* Discouraging early exit in strong bullish conditions

---

## **Training**

The model is trained using PPO over large timesteps to ensure stable learning.

### Run Training

```bash
python src/training/train.py
```

* Uses historical stock data (e.g., AAPL)
* Splits data into training and testing based on time
* Saves trained model for evaluation and reuse

---

## **System Highlights**

* Models **sequential decision-making** in financial markets
* Focuses on **portfolio-level optimization**, not individual trades
* Incorporates **realistic constraints** like transaction costs
* Designed for experimentation with **advanced AI techniques (RL + LLMs)**

---

## **Tech Stack**

* **Language:** Python
* **RL Framework:** Gymnasium, Stable-Baselines3 (PPO)
* **Libraries:** NumPy, Pandas
* **Extensions:** LLM, RAG (optional modules)