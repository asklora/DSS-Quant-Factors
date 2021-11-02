import global_vals
import pandas as pd
import datetime as dt

with global_vals.engine.connect() as conn:
    df = pd.read_sql(f"SELECT * FROM orders WHERE placed AND user_id=1423", conn)
global_vals.engine.dispose()

df['created'] = pd.to_datetime(df['created']).dt.tz_localize(None)
df = df.loc[df['created'] < dt.datetime(2021,11,1,0,0,0)]

df.loc[df['side']=="buy", "qty"] = -df["qty"]
df.loc[df['side']=="buy", "amount"] = -df["amount"]

sum = df.groupby("ticker")["qty", "amount"].sum()

ticker = list(sum.loc[sum['qty']!=0].index)

with global_vals.engine.connect() as conn:
    price = pd.read_sql(f"SELECT ticker, close FROM master_ohlcvtr WHERE ticker in {tuple(ticker)} AND trading_day='2021-10-29'", conn)
global_vals.engine.dispose()

price = price.set_index('ticker')
sum = sum.merge(price, left_index=True, right_index=True, how='outer')

sum['if_sell'] = sum['qty']*sum['close']
sum['if_sell'] = sum['if_sell'].fillna(0)
sum['total'] = sum['amount'] - sum['if_sell']
print(sum["total"].sum())

df = df.set_index(["ticker","filled_at","bot_id","side"])
print(df)