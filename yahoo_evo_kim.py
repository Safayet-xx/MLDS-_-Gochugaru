import numpy as np
import pandas as pd
import sys

sys.path.append(".")
from yahoo_env import StockEnv
from yahoo_qagent import QLearningAgent

print("Loading data")
train_df = pd.read_csv("data_train.csv")
print(f"Training on {len(train_df)} days "
      f"({train_df['Date'].iloc[0]} → {train_df['Date'].iloc[-1]})\n")

env = StockEnv(train_df, initial_cash=10_000)
agent = QLearningAgent(
    n_states=18,
    n_actions=3,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0
)

N_EPISODES = 500
rewards_log = []
portfolio_log = []

print("Starting training\n")
print(f"{'Episode':>8} | {'Avg Reward':>10} | {'Portfolio':>12} | "
      f"{'Epsilon':>8} | {'Trades':>6}")
print("-" * 60)

for episode in range(1, N_EPISODES + 1):
    state = env.reset()
    total_reward = 0
    n_trades = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)

        if action in [1, 2]:
            n_trades += 1

        total_reward += reward
        state = next_state

        if done:
            break

    agent.decay_epsilon()
    agent.snapshot()

    rewards_log.append(total_reward)
    portfolio_log.append(env.portfolio_value)

    if episode % 50 == 0 or episode == 1:
        avg_reward = np.mean(rewards_log[-50:])
        print(f"{episode:>8} | "
              f"{avg_reward:>10.2f} | "
              f"£{env.portfolio_value:>10,.2f} | "
              f"{agent.epsilon:>8.3f} | "
              f"{n_trades:>6}")

print("-" * 60)
print("\nTraining complete!")

np.save("q_table.npy", agent.Q)
print("Q-table saved → q_table.npy")

log_df = pd.DataFrame({
    "episode": range(1, N_EPISODES + 1),
    "reward": rewards_log,
    "portfolio": portfolio_log
})
log_df.to_csv("training_log.csv", index=False)
print("Training log saved → training_log.csv")

print()
agent.print_q_table()

print(f"\nTraining Summary")
print(f"Episodes trained      : {N_EPISODES}")
print(f"Final epsilon         : {agent.epsilon:.4f}")
print(f"Best portfolio value  : £{max(portfolio_log):,.2f}")
print(f"Worst portfolio value : £{min(portfolio_log):,.2f}")
print(f"Final portfolio value : £{portfolio_log[-1]:,.2f}")
print(f"Starting cash         : £10,000.00")

profit = portfolio_log[-1] - 10_000
print(f"Profit / Loss (ep500) : £{profit:+,.2f}  "
      f"({profit / 10_000 * 100:+.1f}%)")