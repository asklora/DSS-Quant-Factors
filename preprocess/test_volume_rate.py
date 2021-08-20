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
# df.loc[df['ticker']=='AAPL.O'].to_csv('tri_z_aapl.csv', index=False)

print(df.describe())

with global_vals.engine_ali.connect() as conn:
    uni = pd.read_sql(f'SELECT ticker, currency_code FROM {global_vals.dl_value_universe_table}', conn)
global_vals.engine_ali.dispose()

df = df.merge(uni, on=['ticker'])

inf_df = df.loc[df['volume_rate_z']==-np.inf]
df = df.replace([np.inf, -np.inf], np.nan)

df.sort_values(by=['volume_rate_z'], ascending=False).head(100).to_csv('tri_z_top.csv', index=False)

import matplotlib.pyplot as plt
plt.hist(df['volume_rate_z'], bins=100)
plt.show()
