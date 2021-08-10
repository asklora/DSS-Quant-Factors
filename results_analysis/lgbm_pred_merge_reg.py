from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
import numpy as np

import global_vals

r_name = 'lastweekavg_change3'

iter_name = r_name

def download_stock_pred():
    ''' download training history and training prediction from DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.*, S.group_code, S.testing_period, S.cv_number FROM {global_vals.result_pred_table}_lgbm_reg P "
                     f"INNER JOIN {global_vals.result_score_table}_lgbm_reg S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql='{r_name}' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'cv_number','group'], keep='last')
    result_all = result_all.drop(['cv_number'], axis=1)

    return result_all

def combine_mode_group(df):
    ''' calculate accuracy score by each industry/currency group '''

    result_dict = {}
    for name, g in df.groupby(['group_code', 'y_type', 'group']):
        result_dict[name] = {}
        result_dict[name]['mae'] = mean_absolute_error(g['pred'], g['actual'])
        result_dict[name]['mse'] = mean_squared_error(g['pred'], g['actual'])
        result_dict[name]['r2'] = r2_score(g['pred'], g['actual'])

    df = pd.DataFrame(result_dict).transpose().reset_index()
    df.columns = ['group_code', 'y_type','group'] + df.columns.to_list()[3:]

    with global_vals.engine_ali.connect() as conn:
        icb_name = pd.read_sql(f"SELECT DISTINCT code_6 as group, name_6 as name FROM icb_code_explanation", conn)  # download training history
        icb_count = pd.read_sql(f"SELECT \"group\", avg(num_ticker) as num_ticker FROM icb_code_count GROUP BY \"group\"", conn)  # download training history
    global_vals.engine_ali.dispose()

    df = df.merge(icb_name, on=['group'])
    df = df.merge(icb_count, on=['group'])

    return df

def combine_mode_time(df):
    ''' calculate accuracy score by each industry/currency group '''

    result_dict = {}
    for name, g in df.groupby(['group_code', 'y_type', 'testing_period']):
        result_dict[name] = {}
        result_dict[name]['mae'] = mean_absolute_error(g['pred'], g['actual'])
        result_dict[name]['mse'] = mean_squared_error(g['pred'], g['actual'])
        result_dict[name]['r2'] = r2_score(g['pred'], g['actual'])

    df = pd.DataFrame(result_dict).transpose().reset_index()
    df.columns = ['group_code', 'y_type','testing_period'] + df.columns.to_list()[3:]

    return df

def calc_pred_class():
    ''' Calculte the accuracy_score if combine CV by mean, median, mode '''

    df = download_stock_pred()

    df = df.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
    df = df.dropna(how='any')

    result_time = combine_mode_time(df)
    result_group = combine_mode_group(df)

    with pd.ExcelWriter(f'score/result_pred_reg_{iter_name}.xlsx') as writer:
        result_time.groupby(['group_code', 'y_type']).mean().to_excel(writer, sheet_name='average')
        result_group.to_excel(writer, sheet_name='mode_group', index=False)
        result_time.to_excel(writer, sheet_name='mode_time', index=False)

if __name__ == "__main__":

    calc_pred_class()

