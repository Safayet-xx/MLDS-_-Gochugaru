import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states=18,
        n_actions=3,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0
    ):
        self.Q = np.zeros((n_states, n_actions))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.eps_min = 0.01
        self.eps_decay = 0.995

        self.q_history = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)

        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        future = 0.0 if done else np.max(self.Q[next_state])
        target = reward + self.gamma * future

        self.Q[state, action] += self.alpha * (
            target - self.Q[state, action]
        )

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def snapshot(self):
        self.q_history.append(self.Q.copy())

    def print_q_table(self):
        rsi_labels = ["RSI-low ", "RSI-mid ", "RSI-high"]
        ma_labels = ["MA-below", "MA-near ", "MA-above"]
        hold_labels = ["no-shares", "holding  "]

        print("\n── Q-Table (Hold | Buy | Sell) ──────────────────────────")
        print(f"{'State':<35} {'Hold':>7} {'Buy':>7} {'Sell':>7}  Best")
        print("-" * 65)

        for rsi_i in range(3):
            for ma_i in range(3):
                for h_i in range(2):
                    state = rsi_i * 6 + ma_i * 2 + h_i
                    q = self.Q[state]
                    best = ["Hold", "Buy ", "Sell"][np.argmax(q)]
                    label = (
                        f"{rsi_labels[rsi_i]} + "
                        f"{ma_labels[ma_i]} + "
                        f"{hold_labels[h_i]}"
                    )

                    print(f"{label:<35} "
                          f"{q[0]:>7.3f} "
                          f"{q[1]:>7.3f} "
                          f"{q[2]:>7.3f}  {best}")

        print("-" * 65)