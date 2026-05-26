import numpy as np
import pandas as pd
class StockEnv:
    N_ACTIONS = 3
    N_STATES  = 18      

    def __init__(self, df, initial_cash=10_000):
        self.df           = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.reset()

    def _discretise(self, rsi, ma_ratio, holding):

        if rsi < 40:
            rsi_bin = 0
        elif rsi > 60:
            rsi_bin = 2
        else:
            rsi_bin = 1

        if ma_ratio < 0.98:
            ma_bin = 0
        elif ma_ratio > 1.02:
            ma_bin = 2
        else:
            ma_bin = 1

        h_bin = 1 if holding else 0
        return rsi_bin * 6 + ma_bin * 2 + h_bin

    def reset(self):
        self.current_step = 0
        self.cash         = self.initial_cash
        self.shares       = 0
        self.prev_worth   = self.initial_cash
        return self._get_state()

    def _get_state(self):
        row = self.df.iloc[self.current_step]
        return self._discretise(
            rsi      = float(row["rsi"]),
            ma_ratio = float(row["ma_ratio"]),
            holding  = self.shares > 0
        )

    def step(self, action):
        row   = self.df.iloc[self.current_step]
        price = float(row["Close"])
        cost  = 0.0

        if action == 1 and self.shares == 0:      # Buy
            self.shares  = self.cash // price     # buy as many as we can
            cost         = self.shares * price * 0.001   # 0.1% broker fee
            self.cash   -= (self.shares * price) + cost

        elif action == 2 and self.shares > 0:     # Sell
            cost         = self.shares * price * 0.001   # 0.1% broker fee
            self.cash   += (self.shares * price) - cost
            self.shares  = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        curr_worth = self.cash + self.shares * price

        reward  = (curr_worth - self.prev_worth) / self.initial_cash * 100
        reward -= cost / self.initial_cash * 100

        self.prev_worth = curr_worth

        if not done:
            next_state = self._get_state()
        else:
            next_state = 0   # episode over, state doesn't matter

        return next_state, reward, done

    @property
    def portfolio_value(self):
        price = float(self.df.iloc[self.current_step]["Close"])
        return self.cash + self.shares * price

    @property
    def profit_pct(self):
        return (self.portfolio_value - self.initial_cash) / self.initial_cash * 100

if __name__ == "__main__":

    print("Loading training data...")
    train_df = pd.read_csv("data_train.csv")
    print(f"Loaded {len(train_df)} rows\n")

    env   = StockEnv(train_df)
    state = env.reset()

    print(f"Initial state index : {state}")
    print(f"Starting cash       : £{env.cash:,.2f}")
    print(f"Starting portfolio  : £{env.portfolio_value:,.2f}")

    test_actions = [1, 0, 0, 2, 0] 
    action_names = {0: "Hold", 1: "Buy ", 2: "Sell"}
    for i, action in enumerate(test_actions):
        next_state, reward, done = env.step(action)

        print(f"Day {i+1} | Action: {action_names[action]} | "
              f"State: {state:2d} → {next_state:2d} | "
              f"Reward: {reward:+.4f} | "
              f"Portfolio: £{env.portfolio_value:,.2f}")

        state = next_state
    print(f"Final portfolio value : £{env.portfolio_value:,.2f}")
    print(f"Profit / Loss         : {env.profit_pct:+.2f}%")
    print(f"\n 3")