import pandas as pd
import global_vals
from sqlalchemy.dialects.postgresql import TEXT
import numpy as np
from sqlalchemy import create_engine
import datetime as dt
from dateutil.relativedelta import relativedelta

# test_url = "postgres://loratech:loraTECH123@pgm-3ns7dw6lqemk36rgpo.pg.rds.aliyuncs.com:5432/postgres"
# test_engine = create_engine(test_url, max_overflow=-1, isolation_level="AUTOCOMMIT")

filter_field = ["EPS1TR12", "WC05480", "WC18100A", "WC18262A", "WC08005",
                "WC18309A", "WC18311A", "WC18199A", "WC08372", "WC05510", "WC08636A",
                "BPS1FD12", "EBD1FD12", "EVT1FD12", "EPS1FD12", "SAL1FD12", "CAP1FD12",
                "WC02999", "WC02001", "WC03101", "WC03501", "WC18312A", "WC02101",
                "WC18264", "WC18267", "WC01451", "WC18810", "WC02401", "WC18274",
                "WC07211", "i0eps"]

with global_vals.engine_ali_prod.connect() as conn:
    df = pd.read_sql('SELECT * FROM ingestion_name', conn)
    df.loc[df['dsws_name'].str[-1]=='A', 'replace_fn1']= df.loc[df['dsws_name'].str[-1]=='A', 'dsws_name'].str[:-1]
    print(df)
    extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
    df.to_sql('ingestion_name', **extra)
global_vals.engine_ali_prod.dispose()
exit(1)

df = df.loc[df['currency_code']=='USD'].sort_values('ai_score', ascending=False).head(20)
df = df.loc[(df['wts_rating']==10)&(df['dlp_1m']==10)]
print(df)

# import os
# import re
# os.chdir('./results_analysis')
# df_list = []
# for f in os.listdir():
#     if re.match('^#rf_reg_pred_v2_weekly._20211014_debug_testy_.*.xlsx$', f):
#         s = pd.read_excel(f, sheet_name='average')[['max_ret','actual']].mean().to_frame().transpose()
#         s.index = [f]
#         df_list.append(s)
#
# df = pd.concat(df_list, axis=0).reset_index()
# df['add'] = df['max_ret'] - df['actual']
# print(df)
#
# df.sort_values(['index']).to_csv('1w4w.csv')