from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
import numpy as np

import global_vals

# r_name = 'indoverfit'
# r_name = 'testing'
# r_name = 'laskweekavg'
r_name = 'lastweekavg'
# r_name = 'lastweekavg_newmacros'
r_name = 'biweekly'
r_name = 'biweekly_new1'
r_name = 'biweekly_ma'

iter_name = r_name

def download_stock_pred():
    ''' download training history and training prediction from DB '''

    with global_vals.engine_ali.connect() as conn:
        query = text(f"SELECT P.*, S.group_code, S.testing_period, S.cv_number FROM {global_vals.result_pred_table}_lgbm_class P "
                     f"INNER JOIN {global_vals.result_score_table}_lgbm_class S ON S.finish_timing = P.finish_timing "
                     f"WHERE S.name_sql='{r_name}' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
        result_all = pd.read_sql(query, conn)       # download training history
    global_vals.engine_ali.dispose()

    # remove duplicate samples from running twice when testing
    result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'cv_number','group'], keep='last')
    result_all = result_all.drop(['cv_number'], axis=1)

    # combine cross validation results by mode
    result_all = result_all.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
    result_all = result_all.dropna(how='any')

    return result_all

def select_best_group():
    ''' select group with historically high prediction accuracy '''

    df = download_stock_pred()

    # test selection process based on last testing_period
    testing_period = df['testing_period'].max()
    df_selection = df.loc[df['testing_period'] < testing_period]

    # # 1. Remove factor with majority proportion of prediction as 1
    # prc1 = {}
    # for name, g in df.groupby(['y_type']):
    #     prc1[name] = g['pred'].value_counts().to_dict()[1]/len(g)
    # x = pd.DataFrame(prc1, index=['index']).transpose().sort_values(by=['index'], ascending=False)
    # print(x)

    # calculate historic accuracy with all sample prior to prediction period
    group_acc = {}
    for name, g in df_selection.groupby(['group_code', 'group', 'y_type']):
        group_acc[name] = accuracy_score(g['pred'], g['actual'])
    group_acc_df = pd.DataFrame(group_acc, index=['0']).transpose().reset_index()
    group_acc_df.columns = ['group_code', 'y_type','group','accuracy']

    # 1. select factor used (accuracy high enough)
    avg = group_acc_df.groupby(['y_type']).mean()
    print(avg)


    # select top (2) countries for
    for name, g in group_acc_df.groupby(['y_type']):
        pass

    print(df)

def combine_mode_group(df):
    ''' calculate accuracy score by each industry/currency group '''

    result_dict = {}
    for name, g in df.groupby(['group_code', 'y_type', 'group']):
        result_dict[name] = accuracy_score(g['pred'], g['actual'])

    df = pd.DataFrame(result_dict, index=['0']).transpose().reset_index()
    df.columns = ['group_code', 'y_type','group','accuracy']

    with global_vals.engine_ali.connect() as conn:
        icb_name = pd.read_sql(f"SELECT DISTINCT code_6 as group, name_6 as name FROM icb_code_explanation", conn)  # download training history
        icb_count = pd.read_sql(f"SELECT \"group\", avg(num_ticker) as num_ticker FROM icb_code_count GROUP BY \"group\"", conn)  # download training history
    global_vals.engine_ali.dispose()

    df = df.merge(icb_name, on=['group'])
    df = df.merge(icb_count, on=['group'])

    return df

if __name__ == "__main__":

    select_best_group()
