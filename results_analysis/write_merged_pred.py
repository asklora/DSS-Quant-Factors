from sqlalchemy import text
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
from pandas.tseries.offsets import MonthEnd
import numpy as np
import os
import re

import global_vals

def download_stock_pred(q, iter_name, history=False):
    ''' organize production / last period prediction and write weight to DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.pred, P.actual, P.y_type, P.group as group_code, S.neg_factor, S.testing_period, S.cv_number FROM {global_vals.result_pred_table}_rf_reg P "
                     f"INNER JOIN {global_vals.result_score_table}_rf_reg S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql like '{iter_name}%' and tree_type ='rf' and use_pca = 0.2 ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'cv_number'], keep='last')
    if history:
        period = result_all['testing_period'].unique()
        tbl_suffix = '_history'
    else:
        period = [result_all['testing_period'].max()]
        tbl_suffix = ''

    for p in period:
        df = result_all.loc[result_all['testing_period']==p]   # keep only last testing i.e. for production
        neg_factor = df[['group_code','neg_factor']].drop_duplicates().set_index('group_code')['neg_factor'].to_dict()

        # use average predictions from different validation sets
        df = df.groupby(['testing_period','y_type','group_code'])['pred'].mean().unstack()

        # classify predictions to n-bins
        df_cut = df.apply(pd.qcut, q=q, labels=False).stack().dropna().reset_index()
        df_cut.columns = ['period_end','factor_name','group','factor_weight']
        df_cut['factor_rank'] = df.rank(ascending=True).stack().dropna().values

        # set format of the sql table
        # df_cut['factor_name'] = df_cut['factor_name'].str[2:]
        df_cut['period_end'] = df_cut['period_end'] + MonthEnd(1)
        df_cut['factor_weight'] = df_cut['factor_weight'].astype(int)
        df_cut['long_large'] = False
        df_cut['last_update'] = dt.datetime.now()

        for k, v in neg_factor.items():
            df_cut.loc[(df_cut['group']==k)&(df_cut['factor_name'].isin([x[2:] for x in v.split(',')])), 'long_large'] = True

        with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
            extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
            try:
                conn.execute(f"DELETE FROM {global_vals.production_factor_rank_table}{tbl_suffix} "
                             f"WHERE period_end='{dt.datetime.strftime(df_cut['period_end'][0], '%Y-%m-%d')}'")   # remove same period prediction if exists
            except:
                pass
            df_cut.sort_values(['group','factor_rank']).to_sql(global_vals.production_factor_rank_table+tbl_suffix, **extra)
        global_vals.engine_ali.dispose()

if __name__ == "__main__":
    download_stock_pred(3, 'pca_trimold2', history=True)

