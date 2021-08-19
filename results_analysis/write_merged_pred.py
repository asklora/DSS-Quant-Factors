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

def download_stock_pred(q, iter_name):
    ''' organize production / last period prediction and write weight to DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.pred, P.actual, P.y_type, P.group as group_code, S.testing_period, S.cv_number FROM {global_vals.result_pred_table}_rf_reg P "
                     f"INNER JOIN {global_vals.result_score_table}_rf_reg S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql like '{iter_name}%' ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'cv_number'], keep='last')
    result_all = result_all.loc[result_all['testing_period']==result_all['testing_period'].max()]   # keep only last testing i.e. for production

    # use average predictions from different validation sets
    result_all = result_all.groupby(['testing_period','y_type','group_code'])['pred'].mean().unstack()

    # classify predictions to n-bins
    result_all = result_all.apply(pd.qcut, q=q, labels=False).stack().dropna().reset_index()

    # set format of the sql table
    result_all.columns = ['period_end','factor_name','group','factor_weight']
    result_all['factor_name'] = result_all['factor_name'].str[2:]
    result_all['period_end'] = result_all['period_end'] + MonthEnd(1)
    result_all['factor_weight'] = result_all['factor_weight'].astype(int)
    result_all['last_update'] = dt.datetime.now()

    with global_vals.engine_ali.connect() as conn:  # write stock_pred for the best hyperopt records to sql
        extra = {'con': conn, 'index': False, 'if_exists': 'append', 'method': 'multi', 'chunksize': 10000}
        conn.execute(f"DELETE FROM {global_vals.production_factor_rank_table} "
                     f"WHERE period_end='{dt.datetime.strftime(result_all['period_end'][0], '%Y-%m-%d')}'")   # remove same period prediction if exists
        result_all.sort_values(['group','factor_weight']).to_sql(global_vals.production_factor_rank_table, **extra)
    global_vals.engine_ali.dispose()

if __name__ == "__main__":
    download_stock_pred(3, 'pca_mse_moretree123')

