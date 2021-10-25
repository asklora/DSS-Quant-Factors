import pandas as pd
import global_vals
from sqlalchemy.dialects.postgresql import TEXT
import numpy as np
from sqlalchemy import create_engine
import datetime as dt
from dateutil.relativedelta import relativedelta

# test_url = "postgres://loratech:loraTECH123@pgm-3ns7dw6lqemk36rgpo.pg.rds.aliyuncs.com:5432/postgres"
# test_engine = create_engine(test_url, max_overflow=-1, isolation_level="AUTOCOMMIT")

with global_vals.engine_ali_prod.connect() as conn:
    df = pd.read_sql('SELECT * FROM ai_value_formula_ratios', conn)
    ingestion_name = pd.read_sql('SELECT dsws_name, our_name FROM ingestion_name', conn)
    ingestion_name['dsws_name'] = ingestion_name['dsws_name'].str.lower()
    ingestion_name['dsws_name'] = ingestion_name['dsws_name'].apply(lambda x: 'fn_'+str(int(x[2:-1])) if x[:2]=='wc' else x)
    ingestion_name = ingestion_name.set_index('dsws_name')['our_name'].to_dict()
    for i in ['field_num','field_denom']:
        df[i] = df[i].replace(ingestion_name)
    extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi'}
    df.to_sql('ai_value_formula_ratios', **extra)
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