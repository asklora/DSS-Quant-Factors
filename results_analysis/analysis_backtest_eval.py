import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
import datetime as dt
import numpy as np
from global_vars import *
from general.sql_process import read_query

def download_model(weeks_to_expire='%%', average_days='%%', name_sql=None):
    ''' evaluation runtime calculated metrics '''

    if name_sql:
        query = f"SELECT * FROM {production_factor_rank_backtest_eval_table} WHERE name_sql like '{name_sql}%%'"
    else:
        query = f"SELECT * FROM {production_factor_rank_backtest_eval_table} WHERE name_sql like 'w{weeks_to_expire}_d{average_days}_%%'"

    df = read_query(query, db_url_read).fillna(0)
    df['net_ret'] = df['max_ret'] - df['min_ret']

    iter_unique_col = ['name_sql', 'group', 'y_type', 'trading_day']  # keep 'cv_number' in last one for averaging
    diff_config_col = ['tree_type', 'use_pca', 'n_splits']
    df_avg = df.groupby(iter_unique_col[:-1] + diff_config_col).mean().reset_index()

    return

if __name__ == "__main__":
    # download_model(weeks_to_expire=26, average_days=28)
    download_model(weeks_to_expire=26)