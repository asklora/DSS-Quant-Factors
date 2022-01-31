import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
import datetime as dt
import numpy as np
from global_vars import *
from general.sql_process import read_query

def download_model(weeks_to_expire=None, average_days=None, start_uid=None, name_sql=None):
    ''' evaluation runtime calculated metrics '''

    if name_sql:
        query = f"SELECT * FROM {result_score_table} WHERE name_sql='{name_sql}' "
    else:
        query = f"SELECT * FROM {result_score_table} WHERE name_sql like 'w{weeks_to_expire}_d{average_days}_%%' "
    if start_uid:
        query += f"AND to_timestamp(left(uid, 20), 'YYYYMMDDHH24MISSUS') > to_timestamp('{start_uid}', 'YYYYMMDDHH24MISSUS')"
    df = read_query(query, db_url_read)

    iter_unique_col = ['name_sql', 'group_code', 'y_type', 'testing_period', 'cv_number']  # keep 'cv_number' in last one for averaging
    diff_config_col = ['tree_type', 'use_pca', 'qcut_q', 'n_splits', 'valid_method']

    # 1. remove duplicate samples from running twice when testing
    df = df.drop_duplicates(subset=iter_unique_col + diff_config_col, keep='last')

    # 2. find best in cv groups
    df_best = df.sort_values(by=['r2_valid'], ascending=False).groupby(iter_unique_col[:-1] + diff_config_col).first()
    df_best_avg_cv = df_best.groupby(['testing_period', 'cv_number']).mean().reset_index()
    df_best_avg_q = df_best.groupby(['qcut_q']).mean()
    df_best_avg_n_splits = df_best.groupby(['n_splits']).mean()

    df_corr_cv = df['cv_number'].corr(df['net_ret'])
    df_corr_n_splits = df['n_splits'].corr(df['net_ret'])

    # 3. calculate average accuracy across testing_period
    df_best_avg = df_best.groupby(iter_unique_col[:-2] + diff_config_col).mean().filter(regex=f'^r2_|net_ret').reset_index()

    # 4. correlation between net_ret & metrics
    def net_ret_corr(g):
        corr1 = -g.filter(regex="^mae_|^mse_").apply(lambda x: x.corr(g['net_ret'], method='pearson'))
        corr2 = g.filter(regex="^r2_").apply(lambda x: x.corr(g['net_ret'], method='pearson'))
        return corr1.append(corr2)
    df_corr = df.groupby(iter_unique_col[:-2] + diff_config_col).apply(net_ret_corr).reset_index()
    df_best_corr = df_best.groupby(iter_unique_col[:-2] + diff_config_col).apply(net_ret_corr).reset_index()
    df_corr_avg_all = pd.DataFrame([df_corr.mean(), df_best_corr.mean()]).transpose().reset_index()
    df_corr_avg_all = df_corr_avg_all.sort_values(by=[1], ascending=False)
    return

if __name__ == "__main__":
    # download_model(weeks_to_expire=4, average_days='%%', start_uid='20220130135806140236')
    download_model(name_sql='w4_d7_20220131132617_debug')