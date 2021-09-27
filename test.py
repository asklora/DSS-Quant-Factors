import pandas as pd
import global_vals
from sqlalchemy.dialects.postgresql import TEXT

df = pd.read_excel('iso_cur.xlsx', 'Sheet1').iloc[:,:2]
df.columns = ['numeric_code', 'alpha3_code']
df['numeric_code'] = df['numeric_code'].astype(str).str.zfill(3)

print(df)

with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
    extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000,
             'dtype': {'numeric_code':TEXT, 'alpha3_code':TEXT}}
    df.to_sql("iso_currency_code", **extra)
global_vals.engine_ali.dispose()