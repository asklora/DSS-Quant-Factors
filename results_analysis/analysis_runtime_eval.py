import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
import datetime as dt
import numpy as np
from global_vars import *
from general.sql_process import read_query

def download_model(weeks_to_expire, average_days):
    ''' evaluation runtime calculated metrics '''

    query = f"SELECT * FROM {result_score_table} WHERE name_sql like 'w{weeks_to_expire}_d{average_days}_%%'"
    df = read_query(query, db_url_read)

    iter_unique_col = ['name_sql', 'group_code', 'y_type', 'testing_period', 'cv_number']  # keep 'cv_number' in last one for averaging
    diff_config_col = ['tree_type', 'use_pca', 'qcut_q']

    # 1. remove duplicate samples from running twice when testing
    df = df.drop_duplicates(subset=iter_unique_col + diff_config_col, keep='last')

    # 2. find best in cv groups
    df_best = df.sort_values(by=['r2_valid'], ascending=False).groupby(iter_unique_col[:-1] + diff_config_col).first()

    # 3. calculate average accuracy across testing_period
    df_best_avg = df_best.groupby(iter_unique_col[:-2] + diff_config_col).mean().filter(regex=f'^r2_').reset_index()
    df_best_avg_q = df_best_avg.groupby(['qcut_q']).mean()

    # 4. correlation between net_ret & metrics
    metrics_col = df.filter(regex="^mae_|^mse_|^r2_").columns.to_list()
    def net_ret_corr(g):
        corr = g[metrics_col+['net_ret']].corr(method='pearson').loc['net_ret']
        return corr
    df_corr = df.groupby(iter_unique_col[:-2] + diff_config_col).apply(net_ret_corr)
    df_corr_avg = df_corr.mean()
    df_best_corr = df_best.groupby(iter_unique_col[:-2] + diff_config_col).apply(net_ret_corr)
    df_best_corr_avg = df_best_corr.mean()
    df_corr_avg_all = pd.DataFrame([df_corr_avg, df_best_corr_avg]).transpose().reset_index()
    return

if __name__ == "__main__":
    download_model(weeks_to_expire=4, average_days='%%')