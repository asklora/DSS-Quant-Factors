import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from scipy.stats import skew

df = pd.read_csv('tri_z.csv')
df_s = df.loc[(df['ticker']=='0981.HK')&(df['volume_rate_z']>2)]
df_s['trading_day'] = pd.to_datetime(df['trading_day']).dt.strftime('%Y-%m')
print(df_s['trading_day'].unique())
print(df.describe())

with global_vals.engine_ali.connect() as conn:
    uni = pd.read_sql(f'SELECT ticker, currency_code FROM {global_vals.dl_value_universe_table}', conn)
global_vals.engine_ali.dispose()

df = df.merge(uni, on=['ticker'])

inf_df = df.loc[df['volume_rate_z']==-np.inf]
df = df.replace([np.inf, -np.inf], np.nan)

import matplotlib.pyplot as plt
plt.hist(df['volume_rate_z'], bins=100)
plt.show()
