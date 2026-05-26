import yfinance as yf
import pandas as pd
import numpy as np

def load_data(ticker="AAPL", start="2015-01-01", end="2023-01-01"):
    print(f"Downloading {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    print(f"Downloaded {len(df)} trading days")

    df["return"] = df["Close"].pct_change()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)          # avoid divide by zero
    df["rsi"] = 100 - (100 / (1 + rs))

    df["ma_ratio"] = df["Close"] / df["Close"].rolling(10).mean()

    df.dropna(inplace=True)
    df.reset_index(inplace=True)          

    print(f"Rows after dropping NaN warm-up: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df[["Date","Close","rsi","ma_ratio"]].head(3))
    print(f"\nLast 3 rows:")
    print(df[["Date","Close","rsi","ma_ratio"]].tail(3))
    return df

def train_test_split(df, train_ratio=0.8):
    split       = int(len(df) * train_ratio)
    train_df    = df.iloc[:split].reset_index(drop=True)
    test_df     = df.iloc[split:].reset_index(drop=True)

    print(f"\nTrain set: {len(train_df)} days "
          f"({train_df['Date'].iloc[0].date()} → "
          f"{train_df['Date'].iloc[-1].date()})")
    print(f"Test set:  {len(test_df)} days "
          f"({test_df['Date'].iloc[0].date()} → "
          f"{test_df['Date'].iloc[-1].date()})")

    return train_df, test_df

if __name__ == "__main__":

    df = load_data()
    train_df, test_df = train_test_split(df)
    df.to_csv("data_full.csv",  index=False)
    train_df.to_csv("data_train.csv", index=False)
    test_df.to_csv("data_test.csv",  index=False)

    print("\nSaved: data_full.csv, data_train.csv, data_test.csv")

    print(f"\nRSI range:      {df['rsi'].min():.1f} → {df['rsi'].max():.1f}  (should be 0–100)")
    print(f"MA ratio range: {df['ma_ratio'].min():.3f} → {df['ma_ratio'].max():.3f}  (should be near 1.0)")
    print(f"\n2")