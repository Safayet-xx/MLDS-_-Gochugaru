

import yfinance as yf
import pandas as pd
import numpy as np

print("Libraries imported successfully")

df = yf.download("AAPL", start="2022-01-01", end="2023-01-01", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"\nDownloaded {len(df)} trading days of AAPL data")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())
print(f"\nAny missing values? {df.isnull().sum().sum()}")
print("\n 1")