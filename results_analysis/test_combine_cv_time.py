import matplotlib.pyplot as plt
from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, \
    multilabel_confusion_matrix, roc_curve
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from pandas.tseries.offsets import MonthEnd

import global_vals

restart = True
model = 'lgbm'
period = 'weekavg' # biweekly / weekavg
r_name = 'lastweekavg_cv_pivotmacro'

iter_name = r_name


def download_stock_pred():
    ''' download training history and training prediction from DB '''

    try:
        if restart:
            raise Exception('------------> Restart')
        result_all = pd.read_csv('cache_result_all.csv')
    except Exception as e:
        print(e)
        with global_vals.engine_ali.connect() as conn:
            query = text(f"SELECT P.group, P.pred, P.actual, S.group_code, S.y_type, S.testing_period, S.cv_number FROM {global_vals.result_pred_table}_{model}_class P "
                         f"INNER JOIN {global_vals.result_score_table}_{model}_class S ON S.finish_timing = P.finish_timing "
                         f"WHERE S.name_sql='{r_name}' AND P.actual IS NOT NULL ORDER BY S.finish_timing")
            result_all = pd.read_sql(query, conn)       # download training history
        global_vals.engine_ali.dispose()
        print(result_all.shape)

        # remove duplicate samples from running twice when testing
        result_all = result_all.drop_duplicates(subset=['group_code', 'testing_period', 'y_type', 'cv_number','group'], keep='last')

        # convert pred/actual class to int & combine 5-CV with mode
        result_all = result_all.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
        result_all = result_all.dropna(subset=['actual'])
        result_all.to_csv('cache_result_all.csv', index=False)

    result_all[['pred','actual']] = result_all[['pred','actual']].astype(int)

    # remove last period no enough data to measure reliably
    result_all = result_all.loc[result_all['testing_period']<result_all['testing_period'].max()]

    # label period_end as the time where we assumed to train the model (try to incorporate all information before period_end)
    if period == 'biweekly':
        result_all['period_end'] = pd.to_datetime(result_all['testing_period']).apply(lambda x: x + relativedelta(weeks=2))
    else:
        result_all['period_end'] = pd.to_datetime(result_all['testing_period']) + MonthEnd(1)

    # map the original premium to the prediction result
    result_all = add_org_premium(result_all)

    return result_all

def add_org_premium(df):
    ''' map the original premium to the prediction result '''

    factor = df['y_type'].unique()
    with global_vals.engine_ali.connect() as conn:
        actual_ret = pd.read_sql(f"SELECT \"group\", period_end, {','.join(factor)} "
                                 f"FROM {global_vals.factor_premium_table}_{period}", conn)  # download training history
    global_vals.engine_ali.dispose()

    actual_ret = actual_ret.set_index(['group', 'period_end']).stack().reset_index()
    actual_ret.columns = ['group', 'period_end', 'y_type', 'premium']

    df = df.merge(actual_ret, on=['group', 'period_end', 'y_type'], how='left')
    return df

if __name__ == "__main__":
    download_stock_pred()