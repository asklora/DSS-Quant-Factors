import pandas as pd
from global_vars import db_url_write
from sqlalchemy.dialects.postgresql import TEXT
import numpy as np
from sqlalchemy import create_engine
import datetime as dt
from dateutil.relativedelta import relativedelta
from general.sql_process import upsert_data_to_database

engine = create_engine(db_url_write, max_overflow=-1, isolation_level="AUTOCOMMIT")
df = pd.DataFrame({"test": {"tc": "tv"}})
upsert_data_to_database(df, "test", how="append", db_url=db_url_write)

exit(1)

# import os
# import re
# os.chdir('./results_analysis')
# df_list = []
# for f in os.listdir():
#     if re.match('^#rf_reg_pred_weekly._20211014_debug_testy_.*.xlsx$', f):
#         s = pd.read_excel(f, sheet_name='average')[['max_ret','actual']].mean().to_frame().transpose()
#         s.index = [f]
#         df_list.append(s)
#
# df = pd.concat(df_list, axis=0).reset_index()
# df['add'] = df['max_ret'] - df['actual']
# print(df)
#
# df.sort_values(['index']).to_csv('1w4w.csv')