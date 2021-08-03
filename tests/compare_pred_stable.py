from sqlalchemy import text
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix
import datetime as dt
import numpy as np
import global_vals

def download_stock_pred(r_name):
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

    result_all[['pred','actual']] = result_all[['pred','actual']].astype(int)

    result_all = result_all.groupby(['group_code', 'testing_period', 'y_type', 'group']).apply(pd.DataFrame.mode).reset_index(drop=True)
    result_all = result_all.dropna(how='any')

    return result_all

def combine_stable():
    df = download_stock_pred('biweekly_ma')
    df1 = download_stock_pred('biweekly_ma_re')
    ddf = df.merge(df1, on=['group','y_type','testing_period'], suffixes=['_1','_2'])

    result_dict = {}
    for name, g in ddf.groupby(['group_code_1', 'testing_period', 'y_type']):
        result_dict[name] = {}
        result_dict[name]['accu'] = accuracy_score(g['pred_1'], g['pred_2'])
        result_dict[name]['mae'] = mean_absolute_error(g['pred_1'], g['pred_2'])
        result_dict[name]['std'] = np.sqrt(mean_squared_error(g['pred_1'], g['pred_2']))

    r = pd.DataFrame(result_dict).transpose().reset_index()
    r.columns = ['group_code', 'testing_period', 'y_type'] + r.columns.to_list()[3:]

    r.to_csv('compare_stable_3.csv', index=False)


if __name__ == "__main__":
    combine_stable()

