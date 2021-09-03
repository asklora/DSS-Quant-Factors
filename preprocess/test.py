import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from scipy.stats import skew

with global_vals.engine_ali.connect() as conn:
    f = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}', conn)
    fp = pd.read_sql(f'SELECT * FROM {global_vals.formula_factors_table}_prod', conn)
global_vals.engine_ali.dispose()

f['t'] = 'f'
fp['t'] = 'fp'

# f = f.set_index('name')
# fp = fp.set_index('name').drop(['scaler'], axis=1)

x = fp.append(f).sort_values('name')
print(x)