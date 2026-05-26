# Thread 1 — Reinforcement Learning
## Stock Trading Q-Learning | COMM075 Group Gochugaru

---

## Dataset Overview



| Source | Yahoo Finance via yfinance |
 AAPL (Apple Inc.) |
| Training period | 2018-01-01 to 2020-12-31 |
| Test period | 2021-01-01 to 2022-12-31 |
Close price, RSI (14-day), Moving Average ratio (20-day) |
 Tabular Q-Learning stock trading agent |





| `yahoo_env.py` 
| `yahoo_data.py` 
| `yahoo_qagent.py` 
| `yahoo_evo.py` 
| `yahoo_evF.py`
| `yahoo_test.py`
| `yahoo_rl.ipynb` 
| `data_full.csv` 
| `data_train.csv` 
| `data_test.csv` 
---

## MDP Formulation

### State Space (18 states)
```
3 RSI bins × 3 MA ratio bins × 2 holding states = 18 total states

RSI bins:
    0 = oversold    (RSI < 40)
    1 = neutral     (40 ≤ RSI ≤ 60)
    2 = overbought  (RSI > 60)

MA ratio bins:
    0 = below average   (MA ratio < 0.98)
    1 = near average    (0.98 ≤ MA ratio ≤ 1.02)
    2 = above average   (MA ratio > 1.02)

Holding state:
    0 = not holding shares
    1 = currently holding shares

State index = RSI_bin × 6 + MA_bin × 2 + holding
```

### Action Space (3 actions)
```
0 = Hold   — do nothing
1 = Buy    — invest all available cash
2 = Sell   — sell all held shares
```

### Reward Function
```
reward = Δ portfolio_value − transaction_cost
```

---

## Q-Learning Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| α (learning rate) | 0.1 | How much new info overrides old |
| γ (discount factor) | 0.99 | Importance of future rewards |
| ε (initial epsilon) | 1.0 | Start fully random |
| ε_min | 0.01 | Minimum exploration rate |
| ε_decay | 0.995 | Epsilon multiplier per episode |

### Bellman Update Rule
```
Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') − Q(s,a)]
```

---

## Member Contributions to yahoo_rl.ipynb

Each member adds clearly labelled cells to the shared notebook:

```
[Lubaba — EDA]
    - AAPL price chart with train/test boundary
    - RSI timeseries with 40/60 thresholds
    - MA ratio distribution
    - State frequency heatmap (18 states)

[Kim — Agent & Training]
    - Training reward curve per episode
    - Epsilon decay curve
    - Q-table heatmap after training
    - Portfolio value per episode

[Anas — Evaluation]
    - Greedy test episode portfolio tracking
    - Action distribution on test set
    - Overtrading analysis

[Long — Environment Design]
    - State space diagram (all 18 states labelled)
    - Reward distribution histogram
    - MDP justification writeup

[Rahin — Reward & State Analysis]
    - Cumulative reward over training
    - RSI threshold sensitivity (40/60 vs 30/70)
    - Transaction cost sensitivity

[Safayet — Baseline & Final Results]
    - Buy-and-hold portfolio value chart
    - RL vs Buy-and-Hold final comparison table
    - Distribution shift discussion
```



# with proper venv instalation and setup  

```bash
# Step 1 — download data and generate CSVs
python yahoo_data.py

# Step 2 — train the Q-learning agent
python yahoo_evo.py

# Step 3 — evaluate on test set
python yahoo_evF.py

# Step 4 — run buy-and-hold baseline
python yahoo_test.py

# Step 5 — open the notebook and run all cells in order
jupyter notebook yahoo_rl.ipynb
```
