import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
import datetime as dt
import numpy as np
from global_vars import *
from general.sql_process import read_query

def download_model(weeks_to_expire=None, average_days=None, start_uid=None, name_sql=None):
    ''' evaluation runtime calculated metrics '''

    try:
        df = pd.read_csv(name_sql + '.csv')
    except:
        if name_sql:
            query = f"SELECT * FROM {result_score_table} WHERE name_sql='{name_sql}' "
        else:
            query = f"SELECT * FROM {result_score_table} WHERE name_sql like 'w{weeks_to_expire}_d{average_days}_%%' "
        if start_uid:
            query += f"AND to_timestamp(left(uid, 20), 'YYYYMMDDHH24MISSUS') > to_timestamp('{start_uid}', 'YYYYMMDDHH24MISSUS')"
        df = read_query(query, db_url_read)
        df.to_csv(name_sql+'.csv', index=False)
    df['uid_hpot'] = df['uid'].str[:20]

    iter_unique_col = ['name_sql', 'group_code', 'y_type', 'testing_period', 'cv_number']  # keep 'cv_number' in last one for averaging
    diff_config_col = ['tree_type', 'use_pca', 'qcut_q', 'n_splits', 'valid_method', 'use_average', 'down_mkt_pct']

    # 1. remove duplicate samples from running twice when testing
    df = df.drop_duplicates(subset=iter_unique_col + diff_config_col, keep='last').fillna(0)

    # 1.1. check train_pred_std == 0
    corr = {"train_pred_std": {}}
    for i in df.columns.to_list():
        unique_var = df[i].unique()
        if len(unique_var) in [2, 3]:
            corr["train_pred_std"][i] = df.groupby(i)['train_pred_std'].apply(lambda x: round(sum(x < 1e-5)/len(x), 3)).to_dict()
    corr = pd.DataFrame(corr)
    print(corr)

    # 1.2. check iterations not done
    df1 = df.groupby(diff_config_col)['r2_valid'].count().reset_index()
    df2 = df.groupby(diff_config_col + ['group_code', 'y_type'])['r2_valid'].count()
    df3 = df.groupby(diff_config_col + ['group_code', 'y_type', 'testing_period'])['r2_valid'].count()

    # 2. find best in cv groups
    df_best_all = df.sort_values(by=['r2_valid'], ascending=False).groupby('uid_hpot').first()

    # 3. filter for not used config
    # df_best_all = df_best_all.loc[~df_best_all['use_average']]
    # df_best_all = df_best_all.loc[df_best_all['qcut_q']==10]

    df_pillar_all = []
    # for each pillar
    for name, df_best in df_best_all.groupby(['y_type']):
        df_best_avg_config = {}
        df_corr_config = {}
        for i in diff_config_col:
            df_best_avg_config[i] = df_best.groupby([i]).mean().reset_index()
            try:
                df_corr_config[i] = df_best[i].corr(df_best['net_ret'])
            except Exception as e:
                print(e)

        # 3. calculate average accuracy across testing_period
        df_best_avg = df_best.groupby(iter_unique_col[:-2] + diff_config_col).mean().filter(regex=f'^r2_|net_ret')

        # 4. correlation between net_ret & metrics
        def net_ret_corr(g):
            corr1 = -g.filter(regex="^mae_|^mse_").apply(lambda x: x.corr(g['net_ret'], method='pearson'))
            corr2 = g.filter(regex="^r2_").apply(lambda x: x.corr(g['net_ret'], method='pearson'))
            return corr1.append(corr2)
        df_corr = df.groupby(iter_unique_col[:-2] + diff_config_col).apply(net_ret_corr).filter(regex=f'^r2_')
        df_corr.columns = [x + '_corr' for x in df_corr.columns.to_list()]

        df_pillar = df_best_avg.merge(df_corr, left_index=True, right_index=True).reset_index()
        df_pillar['y_type'] = name
        print(df_pillar)
        df_pillar_all.append(df_pillar)

    df_pillar_all = pd.concat(df_pillar_all, axis=0)
    return

if __name__ == "__main__":
    # download_model(weeks_to_expire=4, average_days='%%', start_uid='20220130135806140236')
    # download_model(name_sql='w4_d7_20220204170205_debug')
    # download_model(name_sql='w4_d7_20220204144656_debug')
    # download_model(name_sql='w4_d7_20220204181443_debug')
    download_model(name_sql='w8_d7_20220217120110_debug')
    # download_model(name_sql='w26_d7_20220207144412_debug')
    # download_model(name_sql='w26_d7_20220207153438_debug')