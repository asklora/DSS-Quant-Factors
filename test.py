import pandas as pd
import global_vals
from sqlalchemy.dialects.postgresql import TEXT
import numpy as np

with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
    df = pd.read_sql('SELECT * FROM universe_newcode', conn)
    map = pd.read_sql('SELECT * FROM iso_currency_code', conn)
    df = df.merge(map, on=['nation_code'], how='left', suffixes=('','_ws'))
    df = df.drop(['nation_code','nation_name'], axis=1)
    extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize': 10000}
    df.to_sql("universe_newcode", **extra)
global_vals.engine_ali.dispose()