import numpy as np
import pandas as pd
from sqlalchemy import text
import global_vals
import datetime as dt


def count_sample_number(tri):
    ''' count number of samples for each period & each indstry / currency
        -> Select to use 6-digit code = on average 37 samples '''

    with global_vals.engine_ali.connect() as conn:
        universe = pd.read_sql(f"SELECT ticker, currency_code, icb_code FROM {global_vals.dl_value_universe_table}",
                               conn)
    global_vals.engine_ali.dispose()

    tri = tri.merge(universe, on=['ticker'], how='left')
    tri['icb_code'] = tri['icb_code'].replace({'10102010':'101021','10102015':'101022','10102020':'101023',
                                               '10102030':'101024','10102035':'101024'})   # split industry 101020 - software (100+ samples)
    tri['icb_code'] = tri['icb_code'].astype(str).str[:6]

    c1 = tri.groupby(['trading_day', 'icb_code']).count()['stock_return_y'].unstack(level=1)

    c2 = tri.groupby(['trading_day', 'currency_code']).count()['stock_return_y'].unstack(level=1)
    df = pd.concat([c1, c2], axis=1).stack().reset_index()
    df.columns = ['period_end', 'group', 'num_ticker']

    with global_vals.engine_ali.connect() as conn:
        extra = {'con': conn, 'index': False, 'if_exists': 'replace', 'method': 'multi', 'chunksize':1000}
        df.to_sql('icb_code_count', **extra)
    global_vals.engine_ali.dispose()

if __name__ == "__main__":
    tri = pd.read_csv('cache_tri_ratio.csv')
    count_sample_number(tri)
    # exit(0)