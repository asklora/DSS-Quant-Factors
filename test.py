import pandas as pd
import global_vals
from sqlalchemy.dialects.postgresql import TEXT
import numpy as np
from sqlalchemy import create_engine
import datetime as dt
from dateutil.relativedelta import relativedelta

# test_url = "postgres://loratech:loraTECH123@pgm-3ns7dw6lqemk36rgpo.pg.rds.aliyuncs.com:5432/postgres"
# test_engine = create_engine(test_url, max_overflow=-1, isolation_level="AUTOCOMMIT")

import os
import re

os.chdir('./results_analysis')
df_list = []
for f in os.listdir():
    if re.match('^#rf_reg_pred_v2_weekly._20211014_debug_testy_.*.xlsx$', f):
        s = pd.read_excel(f, sheet_name='average')[['max_ret','actual']].mean().to_frame().transpose()
        s.index = [f]
        df_list.append(s)

df = pd.concat(df_list, axis=0).reset_index()
df['add'] = df['max_ret'] - df['actual']
print(df)

df.sort_values(['index']).to_csv('1w4w.csv')