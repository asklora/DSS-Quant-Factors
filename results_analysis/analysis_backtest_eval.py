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

    df_avg_time = df.groupby(iter_unique_col[:-1] + diff_config_col).mean().reset_index()
    df_best_avg_time = df_avg_time.sort_values(by=["net_ret"], ascending=False).groupby(iter_unique_col[:-1]).first().reset_index()
    df_best_avg_time_y = df_best_avg_time.groupby(iter_unique_col[:-2])['net_ret'].agg(['mean', 'count']).reset_index()
    df_best_avg_time_y['start_time'] = pd.to_datetime(df_best_avg_time_y['name_sql'].str.split('_', expand=True)[2],
                                                      format='%Y%m%d%H%M%S', errors='coerce')
    # df_best_avg_time_y: rank name_sql given best configuration -> to select name_sql for analysis_backtest

    return

if __name__ == "__main__":
    # download_model(weeks_to_expire=26, average_days=28)
    download_model(weeks_to_expire=8)